# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: np.ndarray | None
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
from pathlib import Path

import omni.usd
from pxr import UsdGeom
from isaaclab.managers import CommandManager, CurriculumManager, RewardManager, TerminationManager
from isaaclab.ui.widgets import ManagerLiveVisualizer
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
import isaaclab.sim as sim_utils

from ....water import Water
from ....hydrodynamics import Hydrodynamics
from ....hull_volume import HullVolume
from ....fossen import FossenDynamics
from .robot_cfg import FOSSEN_CFG, WIND_CFG, HULLS_CFG, HYDRODYNAMICS_CFG


# https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.html#isaaclab.envs.ManagerBasedRLEnv
class Environment(ManagerBasedRLEnv):
    def __init__(self, cfg: ManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        """ Default Sys Id """
        # [SYSID]
        self.num_of_sysID_params = 9  # this MUST equal SysIdActionCfg's param
        self.sysid_params = {
            "Iz": FOSSEN_CFG["Iz"],
            "X_u": FOSSEN_CFG["X_u"],
            "Y_v": FOSSEN_CFG["Y_v"],
            "N_r": FOSSEN_CFG["N_r"],
            "X_uu": FOSSEN_CFG["X_uu"],
            "Y_vv": FOSSEN_CFG["Y_vv"],
            "N_rr": FOSSEN_CFG["N_rr"],
            "c_rev":  1.0,  # 0.4356  0.538754
            "c_u":    1.0,  # 0.6016  0.647737
            "c_lag":  0.15,
        }

        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._robot = self.scene["robot"]

        """ Hydrodynamics """
        self._dynamics = FossenDynamics(FOSSEN_CFG=FOSSEN_CFG, WIND_CFG=WIND_CFG)
        self._hull_volume = HullVolume(robot=self._robot, stage=omni.usd.get_context().get_stage(),
                                       HULLS_CFG=HULLS_CFG, device=self.device)
        self.hydrodynamics = Hydrodynamics(env=self, robot=self._robot, dynamics=self._dynamics, hull_volume=self._hull_volume,
                                           HYDRODYNAMICS_CFG=HYDRODYNAMICS_CFG, device=self.device)
        print('[ENV] Hydrodynamics Initialized')

        """ Water """
        Water.initialize(device=self.device)
        print('[ENV] Water Initialized')

        """ Misc. buffers used across classes """
        self.tau_total = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        self.tau_hydro = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        self.tau_wind  = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        self.tau_thrust = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        self.tau_perturb = torch.zeros((self.num_envs, 6), device=self.device, dtype=torch.float32)
        self.visualization_buffer = []

        """ Thrusters & Latency Buffer """
        self._num_thrusters = 2
        self._max_buffer_size = 100
        self._history_buffer = torch.zeros(
            (self._max_buffer_size, self.num_envs, self._num_thrusters),
            device=self.device,
            dtype=torch.float32
        )
        self._buffer_idx = 0

        self._motor_state = torch.zeros(
            (self.num_envs, self._num_thrusters), 
            device=self.device, 
            dtype=torch.float32
        )

        prop_paths: tuple[str, ...] = ("{ENV_REGEX_NS}/Robot/BlueBoat/port_prop",
                                       "{ENV_REGEX_NS}/Robot/BlueBoat/stbd_prop")

        self._dirs = torch.tensor([
            [1.0, 0.0, 0.0],  # port
            [1.0, 0.0, 0.0]   # stbd
        ], device=self.device, dtype=torch.float32)

        stage = omni.usd.get_context().get_stage()
        env_ns = "/World/envs/env_0"

        port_prim = stage.GetPrimAtPath(prop_paths[0].replace("{ENV_REGEX_NS}", env_ns))
        stbd_prim = stage.GetPrimAtPath(prop_paths[1].replace("{ENV_REGEX_NS}", env_ns))
        root_prim = stage.GetPrimAtPath(f"{env_ns}/Robot/BlueBoat")

        port_matrix = UsdGeom.Xformable(port_prim).ComputeLocalToWorldTransform(0.0)
        stbd_matrix = UsdGeom.Xformable(stbd_prim).ComputeLocalToWorldTransform(0.0)
        root_matrix = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(0.0)

        root_inv = root_matrix.GetInverse()
        port_local = port_matrix * root_inv
        stbd_local = stbd_matrix * root_inv

        self._pos = torch.tensor([
            list(port_local.ExtractTranslation()),
            list(stbd_local.ExtractTranslation())
        ], device=self.device, dtype=torch.float32)

        """ Load Offline Trajectories """
        script_dir = Path(__file__).parent
        # [SYSID]
        dataset_files = ['zigzag_dataset.npz'] 

        self.offline_datasets = []
        for file_name in dataset_files:
            file_path = script_dir / file_name
            if not file_path.exists():
                print(f"[ENV] Warning: {file_name} not found.")
                continue

            dataset = np.load(file_path)
            raw_time = torch.tensor(dataset["time"], device=self.device, dtype=torch.float32)
            time_tensor = raw_time - raw_time[0]

            traj = dataset["trajectory"]
            traj[:, 2] = np.unwrap(traj[:, 2])

            self.offline_datasets.append({
                "time": time_tensor,
                "duration": time_tensor[-1],
                "thrust": torch.tensor(dataset["thrust"], device=self.device, dtype=torch.float32),
                "trajectory": torch.tensor(traj, device=self.device, dtype=torch.float32)
            })

        print(f"[ENV] Loaded {len(self.offline_datasets)} offline trajectories.")

        if len(self.offline_datasets) > 0:
            self._env_traj_idx = torch.randint(
                0, len(self.offline_datasets), (self.num_envs,), device=self.device
            )
        else:
            self._env_traj_idx = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        """ Not sure if necessary... """
        # https://docs.pytorch.org/docs/2.12/generated/torch.set_float32_matmul_precision.html
        torch.set_float32_matmul_precision('high')
        # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.enable_external_forces_every_iteration
        sim_utils.PhysxCfg.enable_external_forces_every_iteration = True

    def step(self, action: torch.Tensor):
        action = action.to(self.device)

        if not hasattr(self, "_latched_sysid_actions") or self._latched_sysid_actions.shape[0] != action.shape[0]:
            self._latched_sysid_actions = torch.zeros((self.num_envs, self.num_of_sysID_params), device=self.device, dtype=torch.float32)

        is_first_step = (self.episode_length_buf == 0)
        if is_first_step.any():
            self._latched_sysid_actions[is_first_step] = action[is_first_step, :self.num_of_sysID_params]

        modified_action = torch.cat([self._latched_sysid_actions, action[:, self.num_of_sysID_params:]], dim=-1)

        # process thrust and trajectory
        current_thrust = torch.zeros_like(self.command_manager.get_term("thrust").command_buf)
        current_trajectory = torch.zeros_like(self.command_manager.get_term("trajectory").command_buf)

        for traj_id, ds in enumerate(self.offline_datasets):
            env_mask = (self._env_traj_idx == traj_id)
            if not env_mask.any():
                continue

            sim_time = (self.episode_length_buf[env_mask] * self.step_dt) % ds["duration"]

            idx = torch.searchsorted(ds["time"], sim_time)
            idx = torch.clamp(idx, 1, len(ds["time"]) - 1)

            t0 = ds["time"][idx - 1]
            t1 = ds["time"][idx]
            weight = ((sim_time - t0) / (t1 - t0)).unsqueeze(1)

            thrust_0 = ds["thrust"][idx - 1]
            thrust_1 = ds["thrust"][idx]
            current_thrust[env_mask] = thrust_0 + weight * (thrust_1 - thrust_0)

            traj_0 = ds["trajectory"][idx - 1]
            traj_1 = ds["trajectory"][idx]
            traj_lerp = traj_0 + weight * (traj_1 - traj_0)
            traj_lerp[:, 2] = (traj_lerp[:, 2] + torch.pi) % (2 * torch.pi) - torch.pi

            current_trajectory[env_mask] = traj_lerp

        self.command_manager.get_term("thrust").command_buf[:] = current_thrust
        self.command_manager.get_term("trajectory").command_buf[:] = current_trajectory

        # process partially latched actions
        self.action_manager.process_action(modified_action)

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            """ Water """
            Water.on_physics_step(dt=self.physics_dt)

            """ Thrust & Latency Buffer """
            raw_thrust_cmd = self.command_manager.get_command("thrust")[:, :self._num_thrusters]

            self._buffer_idx = (self._buffer_idx + 1) % self._max_buffer_size
            self._history_buffer[self._buffer_idx] = raw_thrust_cmd

            latency_val = 1.0
            if not isinstance(latency_val, torch.Tensor):
                latency_val = torch.tensor(latency_val, device=self.device)
            latency_steps = torch.clamp(latency_val.long(), min=0, max=self._max_buffer_size - 1)

            lookup_indices = (self._buffer_idx - latency_steps) % self._max_buffer_size
            delayed_actions = self._history_buffer[lookup_indices, torch.arange(self.num_envs, device=self.device)]

            def _prep_param(key):
                raw = self.sysid_params[key]
                if isinstance(raw, torch.Tensor): val = raw.to(device=self.device, dtype=torch.float32)
                else: val = torch.tensor(raw, device=self.device, dtype=torch.float32)
                if val.ndim == 0: val = val.expand(self.num_envs)
                if val.ndim == 1: val = val.unsqueeze(1)
                return val

            c_fwd = torch.ones((self.num_envs, 1), device=self.device, dtype=torch.float32)
            # c_stbd = _prep_param("c_stbd")
            c_rev  = _prep_param("c_rev")
            c_u    = _prep_param("c_u")
            c_lag  = _prep_param("c_lag")

            alpha = torch.clamp(self.physics_dt / torch.clamp(c_lag, min=1e-4), max=1.0)
            self._motor_state = self._motor_state + alpha * (delayed_actions - self._motor_state)

            u = self._robot.data.root_link_lin_vel_b[:, 0:1]  # Surge velocity (N, 1)
            v = self._robot.data.root_link_lin_vel_b[:, 1:2]  # Sway velocity (N, 1)
            r = self._robot.data.root_link_ang_vel_b[:, 2:3]  # Yaw velocity (N, 1)
            eps = 1e-3

            y_pos = self._pos[:, 1].unsqueeze(0)
            u_local = u - r * y_pos
            eta_fwd = torch.clamp(1.0 - c_u * u_local, min=0.0)

            # c_base_fwd = torch.cat([c_fwd, c_fwd * c_stbd], dim=1)
            c_base_fwd = torch.cat([c_fwd, c_fwd], dim=1)
            smooth_mask = torch.sigmoid(self._motor_state / eps)

            F_act = self._motor_state * (
                smooth_mask * (c_base_fwd * eta_fwd) + 
                (1.0 - smooth_mask) * (c_base_fwd * c_rev)
            )

            local_forces = F_act.unsqueeze(-1) * self._dirs  # (N, 2, 3)
            r_pos = self._pos.unsqueeze(0).expand(self.num_envs, self._num_thrusters, 3)
            local_torques = torch.cross(r_pos, local_forces, dim=-1)

            net_forces = local_forces.sum(dim=1)
            net_torques = local_torques.sum(dim=1)

            self.tau_thrust[:, :3] = net_forces
            self.tau_thrust[:, 3:] = net_torques

            """ Visualizing local forces projected to World Frame """
            sim_context = sim_utils.SimulationContext.instance()
            if sim_context.render_mode == sim_utils.SimulationContext.RenderMode.FULL_RENDERING:
                q_w = self._robot.data.root_link_quat_w[:, 0:1].unsqueeze(1)
                q_xyz = self._robot.data.root_link_quat_w[:, 1:].unsqueeze(1)

                t_pos = torch.cross(q_xyz, r_pos, dim=-1) * 2.0
                positions_w = r_pos + q_w * t_pos + torch.cross(q_xyz, t_pos, dim=-1) + self._robot.data.root_link_pos_w[:, None, :]

                t_forces = torch.cross(q_xyz, local_forces, dim=-1) * 2.0
                forces_w = local_forces + q_w * t_forces + torch.cross(q_xyz, t_forces, dim=-1)

                scale = 0.01
                endpoints = positions_w + (forces_w * scale)  # (N, T, 3)

                self.visualization_buffer.append({
                    'positions': positions_w.reshape(-1, 3).tolist(),  # (N*T, 3)
                    'endpoints': endpoints.reshape(-1, 3).tolist(),    # (N*T, 3)
                    'color': [(1.0, 0.0, 0.0, 1.0)] * (self.num_envs * self._num_thrusters), # red
                    'size': [4.0] * (self.num_envs * self._num_thrusters),
                })

            """ Apply Combined Wrenches """
            if Water.is_waves_enabled:
                """ not using @torch.compile """
                self.tau_hydro = self.hydrodynamics.compute_hydrodynamic_wrenchs()
            else:
                """ using @torch.compile """
                self.tau_hydro = self.compute_forces_and_torques()

            self.tau_total = self.tau_hydro + self.tau_thrust + self.tau_perturb
            self._robot.permanent_wrench_composer.set_forces_and_torques(
                forces=self.tau_total[:, :3].unsqueeze(1),   # (N, 1, 3)
                torques=self.tau_total[:, 3:].unsqueeze(1),  # (N, 1, 3)
                is_global=False,
            )

            """ Visualize forces """
            if self.sim.render_mode == sim_utils.SimulationContext.RenderMode.FULL_RENDERING:
                self.visualize_force_buffers()

            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()

            # simulate
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            # reset for completed environments
            self._reset_idx(reset_env_ids)

            # Assign a new random trajectory to the resetting environments
            if len(self.offline_datasets) > 0:
                self._env_traj_idx[reset_env_ids] = torch.randint(
                    0, len(self.offline_datasets), (len(reset_env_ids),), device=self.device
                )

            self._history_buffer[:, reset_env_ids, :] = 0.0
            self._motor_state[reset_env_ids] = 0.0
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute(update_history=True)

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras


    # @torch.compile = massive speed increase, but does not work with waves!
    # ...uses JIT compilation but could cause other hidden problems...
    @torch.compile
    def compute_forces_and_torques(self):
        return self.hydrodynamics.compute_hydrodynamic_wrenchs()


    def visualize_force_buffers(self):
        from isaacsim.util.debug_draw import _debug_draw
        import isaaclab.utils.math as math_utils

        draw = _debug_draw.acquire_debug_draw_interface()

        # clear old visualization
        draw.clear_lines()

        start_points = self._robot.data.root_link_pos_w  # (N, 3)

        quat_w = self._robot.data.root_link_quat_w  # (N, 4)
        hydro_forces_w = math_utils.quat_apply(quat_w, self.tau_hydro[:, :3])  # (N, 3)
        wind_forces_w = math_utils.quat_apply(quat_w, self.tau_wind[:, :3])    # (N, 3)

        hydro_scale = 0.001
        wind_scale = 0.1

        # hydro force
        hydro_endpoints = start_points + (hydro_forces_w * hydro_scale)
        draw.draw_lines(
            start_points.cpu().tolist(),
            hydro_endpoints.cpu().tolist(),
            [(0.0, 1.0, 1.0, 0.50)] * self.num_envs,  # cyan
            [5.0] * self.num_envs,
        )

        # wind force
        wind_endpoints = start_points + (wind_forces_w * wind_scale)
        draw.draw_lines(
            start_points.cpu().tolist(),
            wind_endpoints.cpu().tolist(),
            [(0.0, 0.0, 1.0, 1.0)] * self.num_envs,  # blue
            [2.0] * self.num_envs,
        )

        # draw all visualization buffer items
        for vis_data in self.visualization_buffer:
            draw.draw_lines(
                vis_data['positions'],
                vis_data['endpoints'],
                vis_data['color'],
                vis_data['size'],
            )

        # clear visualization buffer for next frame
        self.visualization_buffer.clear()
