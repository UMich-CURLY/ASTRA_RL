import torch
from isaaclab.utils.math import quat_apply_inverse

from .water import Water


class Hydrodynamics:
    def __init__(self, env, robot, dynamics, hull_volume, HYDRODYNAMICS_CFG, device):
        self.device = device
        self.env = env
        self._robot = robot

        self.N = robot.data.root_pos_w.shape[0]

        self.dynamics = dynamics   # Fossen dynamics
        self.volume = hull_volume  # volume and CoB of hull

        # water and buoyancy properties
        self.rho_gravity = HYDRODYNAMICS_CFG["RHO"] * 9.81
        self.total_rest_vol = self.volume.get_total_rest_vol()  # [M^3]
        b_neutral = HYDRODYNAMICS_CFG["MASS"] / (HYDRODYNAMICS_CFG["RHO"]* self.total_rest_vol)
        self.buoyancy_scale = self.rho_gravity * b_neutral

        # pre-allocate world-z tensor
        self.z_w = torch.tensor([[0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32).expand(self.N, 3).contiguous()

        # CoM offset
        self.com_offset_body = torch.tensor(HYDRODYNAMICS_CFG["COM_OFFSET_BODY"],
                                            device=self.device,
                                            dtype=torch.float32)


    def _get_tau_buoyancy(self, root_quat_w):
        """Estimate the tau_buoyancy for all envs
        returns: (N, 6) buoyancy forces and torques tensor for N envs
        """
        # transform z_w to z_b
        z_b = quat_apply_inverse(root_quat_w, self.z_w)  # (N, 3)

        # get submerged volume and centroid in body frame [volume: (N, H), centroid: (N, H, 3)]
        volume, centroid = self.volume.submerged_vol_centroid_b(self._robot.data)

        # world force is buoyancy force in the Z direction
        vol_sum = volume.sum(dim=1, keepdim=True)  # (N,)
        force_b = (vol_sum * self.buoyancy_scale) * z_b  # (N, 3)

        # compute torque in body frame
        r = centroid - self.com_offset_body  # (N, H, 3)
        weighted_r_sum = torch.sum(volume.unsqueeze(-1) * r, dim=1)  # (N, 3)
        torque_b = torch.cross(weighted_r_sum, z_b, dim=-1) * self.rho_gravity  # (N, 3)

        # total force and torque is sum over all hulls
        return torch.cat([force_b, torque_b], dim=-1)  # (N, 6)


    def compute_hydrodynamic_wrenchs(self) -> torch.Tensor:
        """Compute hydrodynamic wrenchs for all envs
        returns: (N, 6) forces and torques tensor [X, Y, Z, K, M, N] for N envs
        """
        # save root_quat_w to avoid repeated calls
        root_quat_w = self._robot.data.root_quat_w

        # compute the relative nu (nu_body - nu_wave)
        nu_wave_w = Water.get_nu_wave(self._robot.data.root_pos_w, device=self.device)
        nu_rel_lin = self._robot.data.root_link_lin_vel_b - quat_apply_inverse(root_quat_w, nu_wave_w[:, :3])
        nu_rel = torch.cat([nu_rel_lin, self._robot.data.root_link_ang_vel_b], dim=-1)

        # compute buoyancy and hydro damping forces
        tau_buoyancy = self._get_tau_buoyancy(root_quat_w)  # (N, 6)
        tau_damping = self.dynamics.compute_damping_forces(nu_rel, self.env)  # (N, 6)
        self.env.tau_hydro = (tau_buoyancy + tau_damping)

        # get tau_wind (N, 6) (if it exists)
        tau_wind = getattr(self.env, "tau_wind", torch.zeros((self.N, 6), device=self.device, dtype=torch.float32))

        # combine all forces to a single wrench
        return tau_buoyancy + tau_damping + tau_wind  # (N, 6)
