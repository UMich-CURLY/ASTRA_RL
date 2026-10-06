import torch

# nu  = [u, v, w, p, q, r] (surge, sway, heave, roll, pitch, yaw)
# tau = [X, Y, Z, K, M, N] (forces and torques)


class FossenDynamics:
    def __init__(self, FOSSEN_CFG, WIND_CFG):
        FossenDynamics.WIND_CFG = WIND_CFG
        self._mass = FOSSEN_CFG["mass"]
        self._Iz = FOSSEN_CFG["Iz"]
        
        # linear drag coefficients
        self._lin_drag_list    = [abs(FOSSEN_CFG["X_u"]),  abs(FOSSEN_CFG["Y_v"]),  abs(FOSSEN_CFG["Z_w"]),
                                  abs(FOSSEN_CFG["K_p"]),  abs(FOSSEN_CFG["M_q"]),  abs(FOSSEN_CFG["N_r"])]
        # quadratic drag coefficients
        self._nonlin_drag_list = [abs(FOSSEN_CFG["X_uu"]), abs(FOSSEN_CFG["Y_vv"]), abs(FOSSEN_CFG["Z_ww"]),
                                  abs(FOSSEN_CFG["K_pp"]), abs(FOSSEN_CFG["M_qq"]), abs(FOSSEN_CFG["N_rr"])]

        # permanent buffers on GPU (6,)
        self._lin_drag = None
        self._nonlin_drag = None
        self._buffers_initialized = False


    def get_dynamics_params(self, env=None):
        if env is None or not hasattr(env, "sysid_params") or env.sysid_params is None:
            return self._mass, self._Iz, self._lin_drag, self._nonlin_drag

        # mass = env.sysid_params["mass"]
        Iz = env.sysid_params["Iz"]

        lin_drag = self._lin_drag.expand(env.num_envs, -1).clone()
        lin_drag[:, 0] = torch.as_tensor(env.sysid_params["X_u"])
        lin_drag[:, 1] = torch.as_tensor(env.sysid_params["Y_v"])
        lin_drag[:, 5] = torch.as_tensor(env.sysid_params["N_r"])

        nonlin_drag = self._nonlin_drag.expand(env.num_envs, -1).clone()
        nonlin_drag[:, 0] = torch.as_tensor(env.sysid_params["X_uu"])
        nonlin_drag[:, 1] = torch.as_tensor(env.sysid_params["Y_vv"])
        nonlin_drag[:, 5] = torch.as_tensor(env.sysid_params["N_rr"])

        return self._mass, Iz, lin_drag, nonlin_drag


    def compute_damping_forces(self, nu_rel, env=None):
        """Compute hydrodynamic linear and nonlinear drag for all envs
        nu_rel:  (N, 6) vel_rel tensor [u, v, w, p, q, r] for N envs
        env:     Optional Environment instance carrying sysid_params
        returns: (N, 6) force   tensor [X, Y, Z, K, M, N] for N envs
        """
        # initialize buffers on the first execution
        if not self._buffers_initialized:
            self._lin_drag    = torch.tensor(self._lin_drag_list,    device=nu_rel.device, dtype=nu_rel.dtype)
            self._nonlin_drag = torch.tensor(self._nonlin_drag_list, device=nu_rel.device, dtype=nu_rel.dtype)
            self._buffers_initialized = True

        _, _, lin_drag, nonlin_drag = self.get_dynamics_params(env)
        tau_linear = -lin_drag * nu_rel
        tau_nonlinear = -nonlin_drag * torch.abs(nu_rel) * nu_rel

        return tau_linear + tau_nonlinear


    @staticmethod
    def compute_wind_forces(nu, wind_vel_b):
        # rel velocity
        u_rw = nu[:, 0:1] - wind_vel_b[:, 0:1]
        v_rw = nu[:, 1:2] - wind_vel_b[:, 1:2]

        # apparent wind speed and angle
        V_rw = torch.sqrt(u_rw**2 + v_rw**2 + 1e-16)
        gamma_rw = -torch.atan2(v_rw, u_rw)

        cx = -FossenDynamics.WIND_CFG["C_x"] * torch.cos(gamma_rw)
        cy =  FossenDynamics.WIND_CFG["C_y"] * torch.sin(gamma_rw)
        cn =  FossenDynamics.WIND_CFG["C_n"] * torch.sin(2 * gamma_rw)

        q = 0.5 * FossenDynamics.WIND_CFG["rho_air"] * (V_rw**2)

        # 6-dof wrench tensor
        tau_wind = torch.zeros_like(nu)
        tau_wind[:, 0:1] = q * FossenDynamics.WIND_CFG["A_fw"] * cx  # surge
        tau_wind[:, 1:2] = q * FossenDynamics.WIND_CFG["A_lw"] * cy  # sway
        tau_wind[:, 5:6] = q * FossenDynamics.WIND_CFG["A_lw"] * FossenDynamics.WIND_CFG["L_oa"] * cn  # yaw

        return tau_wind  # (N, 6)
