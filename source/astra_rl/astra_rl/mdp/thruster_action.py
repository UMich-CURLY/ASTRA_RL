import torch
import omni.usd

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from isaacsim.util.debug_draw import _debug_draw
from isaaclab.sim import SimulationContext


# https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm
class ThrusterAction(ActionTerm):
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm.__init__
    def __init__(self, cfg, env):
        super().__init__(cfg, env)

        self._env = env

        # pre-allocate env indices and zero torque tensors
        self._env_indices = torch.arange(self.num_envs, device=self.device)
        self._zero_torques = torch.zeros((self.num_envs, 1, 3), dtype=torch.float32, device=self.device)
        self._wrench_composer = self._asset.instantaneous_wrench_composer

        # required for ActionTerm methods
        self._num_thrusters = len(cfg.prim_paths)
        self._raw_actions = torch.zeros((self.num_envs, self._num_thrusters), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # ActionTerm debug visualization
        # https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.util.debug_draw/docs/index.html
        self.draw = _debug_draw.acquire_debug_draw_interface()

        # thruster directions
        self._dirs = torch.tensor(cfg.directions, device=self.device)

        # thruster min/max
        self._force_min_max = torch.tensor(cfg.force_min_max, device=self.device)
        self.min_forces = self._force_min_max[:, 0]
        self.max_forces = self._force_min_max[:, 1]

        # action history
        self._max_buffer_size = 10

        # fixed-size history buffer and its associated index
        self._history_buffer = torch.zeros(
            (self._max_buffer_size, self.num_envs, self._num_thrusters),
            device=self.device
        )  # (history_steps, N, T)
        self._buffer_idx = 0

        # motor state buffer for spool-up
        self._motor_state = torch.zeros((self.num_envs, self._num_thrusters), device=self.device, dtype=torch.float32)

        # sysID dynamics parameters
        self._c_lag_range = cfg.c_lag_range
        self._c_u_range = cfg.c_u_range
        self._c_rev_range = cfg.c_rev_range
        self._c_stbd_range = cfg.c_stbd_range

        self.c_lag = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.c_u = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.c_rev = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self.c_stbd = torch.ones(self.num_envs, device=self.device, dtype=torch.float32)
        self._sample_sysid_params()

        # Newtons force deadband
        self.force_deadband = cfg.force_deadband

        # store stage and declare list for USD positions
        stage = omni.usd.get_context().get_stage()

        # get the first env's root path
        env_0_path = self._asset.root_physx_view.prim_paths[0]
        env_namespace = env_0_path.rsplit("/Robot", 1)[0]

        # replace {ENV_REGEX_NS} with this specific env namespace (i.e. env_0, env_1, etc.)
        expanded_prim_paths = [p.replace("{ENV_REGEX_NS}", env_namespace) for p in cfg.prim_paths]
        env_pos = []

        # iterate through all prim paths
        for p in expanded_prim_paths:
            prim = stage.GetPrimAtPath(p)

            # error checking for invalid prim paths
            if not prim or not prim.IsValid():
                raise ValueError(f"[ThrusterAction] Prim path '{p}' not found.")

            # get the local position of the xform prim
            translation = prim.GetAttribute("xformOp:translate").Get()
            env_pos.append(list(translation))

        # store the first env's thruster positions to use in apply_actions()
        base_pos = torch.tensor(env_pos, dtype=torch.float32, device=self.device)
        self._pos = base_pos.expand(self.num_envs, self._num_thrusters, 3)


    # ActionTerm attribute: dimension of the action term
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm.action_dim
    @property
    def action_dim(self):
        return self._num_thrusters


    # ActionTerm attribute: the input/raw actions sent to the term
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm.raw_actions
    @property
    def raw_actions(self):
        return self._raw_actions


    # ActionTerm attribute: the actions computed by the term after applying processing
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm.processed_actions
    @property
    def processed_actions(self):
        return self._processed_actions


    @property
    def processed_action(self) -> torch.Tensor:
        return self._history_buffer[self._buffer_idx].clone()


    @property
    def prev_processed_action(self) -> torch.Tensor:
        prev_idx = (self._buffer_idx - 1) % (self._max_buffer_size)
        return self._history_buffer[prev_idx].clone()


    @property
    def prev2_processed_action(self) -> torch.Tensor:
        prev2_idx = (self._buffer_idx - 2) % (self._max_buffer_size)
        return self._history_buffer[prev2_idx].clone()


    # ActionTerm method: processes the actions sent to the environment
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm.process_actions
    def process_actions(self, actions):
        self._raw_actions[:] = actions

        # process raw actions (clip or tanh works)
        torch.mul(actions, 0.2, out=self._processed_actions)
        self._processed_actions.tanh_()

        # asymmetric scaling
        multipliers = torch.where(
            self._processed_actions < 0, 
            torch.abs(self.min_forces), 
            self.max_forces
        )
        self._processed_actions.mul_(multipliers)

        # apply Newtons force deadband
        self._processed_actions.masked_fill_(
            torch.abs(self._processed_actions) < self.force_deadband, 
            0.0
        )

        # increment circular head pointer and insert latest action
        self._buffer_idx = (self._buffer_idx + 1) % (self._max_buffer_size)
        self._history_buffer[self._buffer_idx] = self._processed_actions


    # ActionTerm method: applies the actions to the asset managed by the term
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTerm.apply_actions
    def apply_actions(self):
        # target command (T_cmd) direct from processed actions
        target_thrust = self._processed_actions

        # alpha = dt / c_lag
        alpha = torch.clamp(self._env.physics_dt / torch.clamp(self.c_lag, min=1e-4), max=1.0).unsqueeze(1)

        # discrete low-pass filter: T_del,k = T_del,k-1 + alpha * (T_cmd,k - T_del,k-1)
        self._motor_state = self._motor_state + alpha * (target_thrust - self._motor_state)

        # Retrieve body frame linear and angular velocities. 
        # Support both RigidObjectData (root_lin_vel_b) and ArticulationData (root_link_lin_vel_b)
        if hasattr(self._asset.data, "root_link_lin_vel_b"):
            u = self._asset.data.root_link_lin_vel_b[:, 0:1]  # Surge velocity (N, 1)
            r = self._asset.data.root_link_ang_vel_b[:, 2:3]  # Yaw velocity (N, 1)
        else:
            u = self._asset.data.root_lin_vel_b[:, 0:1]       # Surge velocity (N, 1)
            r = self._asset.data.root_ang_vel_b[:, 2:3]       # Yaw velocity (N, 1)

        # Thruster offsets in Y
        y_pos = self._pos[:, :, 1]  # (N, T)
        u_local = u - r * y_pos     # (N, T)
        
        # Surge efficiency drop
        eta_fwd = torch.clamp(1.0 - self.c_u.unsqueeze(1) * u_local, min=0.0)

        # Build forward base efficiencies (handling port vs stbd if 2 thrusters exist)
        c_fwd = torch.ones((self.num_envs, 1), device=self.device, dtype=torch.float32)
        if self._num_thrusters == 2:
            c_base_fwd = torch.cat([c_fwd, c_fwd * self.c_stbd.unsqueeze(1)], dim=1)
        else:
            c_base_fwd = torch.ones((self.num_envs, self._num_thrusters), device=self.device, dtype=torch.float32)

        # Smooth sigmoid transition between forward and reverse thrust efficiency drops
        eps = 1e-3
        smooth_mask = torch.sigmoid(self._motor_state / eps)

        F_act = self._motor_state * (
            smooth_mask * (c_base_fwd * eta_fwd) + 
            (1.0 - smooth_mask) * (c_base_fwd * self.c_rev.unsqueeze(1))
        )

        # shape active forces to (num_envs, _num_thrusters, _dirs)
        forces = F_act.unsqueeze(-1) * self._dirs  # (N, T, 3)

        # save visualization in env buffer if FULL_RENDERING is enabled
        sim_context = SimulationContext.instance()
        if sim_context.render_mode == sim_context.RenderMode.FULL_RENDERING:
            N, T, _ = forces.shape

            q_w = self._asset.data.root_quat_w[:, 0:1].unsqueeze(1)    # (N, 1, 1)
            q_xyz = self._asset.data.root_quat_w[:, 1:].unsqueeze(1)   # (N, 1, 3)

            # transform forces to the world frame
            t_forces = torch.cross(q_xyz, forces, dim=-1) * 2.0
            forces_w = forces + q_w * t_forces + torch.cross(q_xyz, t_forces, dim=-1)

            # transform thruster positions to world frame
            t_pos = torch.cross(q_xyz, self._pos, dim=-1) * 2.0
            positions_w = self._pos + q_w * t_pos + torch.cross(q_xyz, t_pos, dim=-1) + self._asset.data.root_pos_w[:, None, :]

            # endpoints of visualization
            endpoints = positions_w + forces_w * (1.0 / self._force_min_max.max())  # (N, T, 3)

            # store for visualization in environment
            # https://docs.isaacsim.omniverse.nvidia.com/4.5.0/py/source/extensions/isaacsim.util.debug_draw/docs/index.html
            self._env.visualization_buffer.append({
                'positions': positions_w.reshape(-1, 3).tolist(),  # (N*T, 3)
                'endpoints': endpoints.reshape(-1, 3).tolist(),    # (N*T, 3)
                'color': [(1.0, 0.0, 0.0, 1.0)] * N*T,             # red
                'size': [3.0] * N*T,                               # size
            })

            # set global forces and positions used in the wrench composer
            apply_forces = forces_w
            r_pos = positions_w - self._asset.data.root_pos_w[:, None, :]
            is_global_flag = True
        else:
            # set local forces and positions used in the wrench composer
            apply_forces = forces
            r_pos = self._pos
            is_global_flag = False

        net_forces = apply_forces.sum(dim=1, keepdim=True)  # (N, 1, 3)
        net_torques = torch.cross(r_pos, apply_forces, dim=-1).sum(dim=1, keepdim=True)  # (N, 1, 3)

        self._wrench_composer.add_forces_and_torques(
            forces=net_forces,
            torques=net_torques,
            positions=None,
            is_global=is_global_flag
        )


    # ActionTerm method: resets the manager term
    def reset(self, env_ids=None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._history_buffer.zero_()
            self._motor_state.zero_()
            self._sample_sysid_params()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._history_buffer[:, env_ids] = 0.0
            self._motor_state[env_ids] = 0.0
            self._sample_sysid_params(env_ids)


    # samples sysID thrust parameters per environment
    def _sample_sysid_params(self, env_ids=None):
        if env_ids is None:
            env_ids = self._env_indices

        def _sample(range_tuple):
            low, high = range_tuple
            return torch.rand(env_ids.numel(), device=self.device) * (high - low) + low

        self.c_lag[env_ids] = _sample(self._c_lag_range)
        self.c_u[env_ids] = _sample(self._c_u_range)
        self.c_rev[env_ids] = _sample(self._c_rev_range)
        self.c_stbd[env_ids] = _sample(self._c_stbd_range)


# https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ActionTermCfg
@configclass
class ThrusterActionCfg(ActionTermCfg):
    class_type = ThrusterAction
    asset_name: str = "robot"

    # prim path of the thrusters's xform (used for position)
    prim_paths: tuple[str, ...] = (
        "{ENV_REGEX_NS}/Robot/BlueBoat/port_prop",
        "{ENV_REGEX_NS}/Robot/BlueBoat/stbd_prop",
    )

    # direction of force of each thruster
    directions: tuple[tuple[float, float, float], ...] = (
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
    )

    # min and max force (N) of each thruster (min is always neg)
    force_min_max: tuple[tuple[float, float], ...] = (
        (-10.0, 20.0),
        (-10.0, 20.0),
    )

    # [DR] sysID thruster coefficients
    c_lag_range: tuple[float, float]  = (0.10, 0.20)
    c_u_range: tuple[float, float]    = (0.781187*0.9, 0.781187*1.1)
    c_rev_range: tuple[float, float]  = (0.565634*0.9, 0.565634*1.1)
    c_stbd_range: tuple[float, float] = (0.95, 1.05)


    # minimum +- allowed Newtons
    force_deadband: float = 5.0
