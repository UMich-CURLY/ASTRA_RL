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

import omni.usd
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
        self.visualization_buffer = []

        """ Not sure if necessary... """
        # https://docs.pytorch.org/docs/2.12/generated/torch.set_float32_matmul_precision.html
        torch.set_float32_matmul_precision('high')
        # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.html#isaaclab.sim.PhysxCfg.enable_external_forces_every_iteration
        sim_utils.PhysxCfg.enable_external_forces_every_iteration = True

    def step(self, action: torch.Tensor):
        # process actions
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        #self.benchmark_phys_step()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1

            """ Water """
            Water.on_physics_step(dt=self.physics_dt)

            if Water.is_waves_enabled:
                """ not using @torch.compile """
                self.tau_total = self.hydrodynamics.compute_hydrodynamic_wrenchs()
                self._robot.permanent_wrench_composer.set_forces_and_torques(
                    forces=self.tau_total[..., :3].unsqueeze(1),   # (N, 1, 3)
                    torques=self.tau_total[..., 3:].unsqueeze(1),  # (N, 1, 3)
                    is_global=False,
                )
            else:
                """ using @torch.compile """
                self.tau_total = self.compute_forces_and_torques()
                self._robot.permanent_wrench_composer.set_forces_and_torques(
                    forces=self.tau_total[..., :3].unsqueeze(1),   # (N, 1, 3)
                    torques=self.tau_total[..., 3:].unsqueeze(1),  # (N, 1, 3)
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

            self._reset_idx(reset_env_ids)

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
        
        # scale factors
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
