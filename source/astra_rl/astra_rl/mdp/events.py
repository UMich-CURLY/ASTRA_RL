# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils
from isaaclab.utils.interpolation import LinearInterpolation
from isaacsim.util.debug_draw import _debug_draw

from ..water import Water
from ..fossen import FossenDynamics
from ..tasks.manager_based.station_keeping.robot_cfg import HULLS_CFG  # isort:skip

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


_wind_interpolator = None


def reset_robot_on_water(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]

    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)

    # change pose based on water
    water_z = Water.get_wave_z(positions[:, 0:1], positions[:, 1:2], env.device)
    positions[:, 2] += water_z.squeeze(-1)

    rest_offset_z = sum(hull["REST_WATERLINE_Z"] for hull in HULLS_CFG) / len(HULLS_CFG)
    positions[:, 2] -= rest_offset_z

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_robot_sys_id(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]
    num_resets = len(env_ids)

    if hasattr(env, "_offline_trajectory") and env._offline_trajectory is not None:
        first_frame = env._offline_trajectory[0]
        x_0, y_0, yaw_0 = first_frame[0], first_frame[1], first_frame[2]
        vx_0, vy_0, wz_0 = first_frame[3], first_frame[4], first_frame[5]
    else:
        x_0 = y_0 = yaw_0 = 0.0
        vx_0 = vy_0 = wz_0 = 0.0

    positions = asset.data.default_root_state[env_ids, 0:3].clone()
    positions[:, 0] = env.scene.env_origins[env_ids, 0] + x_0
    positions[:, 1] = env.scene.env_origins[env_ids, 1] + y_0

    water_z = Water.get_wave_z(positions[:, 0:1], positions[:, 1:2], env.device)
    rest_offset_z = sum(hull["REST_WATERLINE_Z"] for hull in HULLS_CFG) / len(HULLS_CFG)
    positions[:, 2] = water_z.squeeze(-1) - rest_offset_z

    yaw_tensor = torch.full((num_resets,), yaw_0, device=asset.device, dtype=torch.float32)
    orientations = math_utils.quat_from_euler_xyz(
        torch.zeros_like(yaw_tensor),
        torch.zeros_like(yaw_tensor),
        yaw_tensor
    )

    velocities = torch.zeros((num_resets, 6), device=asset.device, dtype=torch.float32)
    lin_vel_b = torch.tensor([vx_0, vy_0, 0.0], device=asset.device, dtype=torch.float32).expand(num_resets, 3)
    
    velocities[:, 0:3] = math_utils.quat_apply(orientations, lin_vel_b)
    velocities[:, 5] = wz_0

    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def set_wind_params(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wind_speed_range: tuple[float, float] = (0.0, 0.0),
    wind_dir_range: tuple[float, float] = (0.0, 0.0),
    wind_dir_range2: tuple[float, float] | None = None,
):
    device = env.device
    N = len(env_ids)

    # initialize buffers if they doesn't exist
    if not hasattr(env, 'wind_speed_mean') or not hasattr(env, 'wind_dir_mean'):
        env.wind_speed_mean = torch.zeros(env.num_envs, device=device)
        env.wind_dir_mean   = torch.zeros(env.num_envs, device=device)
    if not hasattr(env, 'wind_speed_current') or not hasattr(env, 'wind_dir_current'):
        env.wind_speed_current = torch.zeros(env.num_envs, device=device)
        env.wind_dir_current   = torch.zeros(env.num_envs, device=device)

    # uniformly distribute wind_speed and wind_dir
    wind_speed_samples = torch.rand(N, device=device) * (wind_speed_range[1] - wind_speed_range[0]) + wind_speed_range[0]
    wind_dir_samples   = torch.rand(N, device=device) * (wind_dir_range[1]   - wind_dir_range[0])   + wind_dir_range[0]

    # if using wind_dir_range2, split the choice 50/50 per env
    if wind_dir_range2 is not None:
        wind_dir_sample2 = torch.rand(N, device=device) * (wind_dir_range2[1] - wind_dir_range2[0]) + wind_dir_range2[0]
        use_range2_mask = torch.rand(N, device=device) < 0.5
        wind_dir_samples = torch.where(use_range2_mask, wind_dir_sample2, wind_dir_samples)

    env.wind_speed_mean[env_ids] = wind_speed_samples
    env.wind_dir_mean[env_ids]   = math_utils.wrap_to_pi(wind_dir_samples)

    # intialize current wind for the Ornstein–Uhlenbeck process 
    env.wind_speed_current[env_ids] = env.wind_speed_mean[env_ids]
    env.wind_dir_current[env_ids]   = env.wind_dir_mean[env_ids]


def apply_wind_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    wind_speed_std: float = 0.0,
    wind_dir_std: float = 0.0,
    OU_theta: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset = env.scene[asset_cfg.name]
    device = asset.device
    N = len(env_ids)

    # Ornstein–Uhlenbeck process
    wind_speeds_step = env.wind_speed_current[env_ids] + OU_theta * (env.wind_speed_mean[env_ids] - env.wind_speed_current[env_ids])
    wind_dirs_step   = env.wind_dir_current[env_ids]   + OU_theta * math_utils.wrap_to_pi(env.wind_dir_mean[env_ids] - env.wind_dir_current[env_ids])

    # add standard normal noise to the wind speed and direction
    wind_speeds_scalar = (torch.randn(N, device=device) * wind_speed_std) + wind_speeds_step
    wind_speeds_scalar = torch.clamp(wind_speeds_scalar, min=0.0)  # non-negative speed
    wind_dirs = (torch.randn(N, device=device) * wind_dir_std) + wind_dirs_step
    wind_dirs = math_utils.wrap_to_pi(wind_dirs)  # keep wind dir [-pi, pi]

    # update current wind speeds and directions
    env.wind_speed_current[env_ids] = wind_speeds_scalar
    env.wind_dir_current[env_ids]   = wind_dirs

    # store raw wind speed and dir in env buffer (for observation)
    if not hasattr(env, 'wind_speed_scalar') or not hasattr(env, 'wind_dir'):
        env.wind_speed_scalar = torch.zeros(env.num_envs, 1, device=device)
        env.wind_dir = torch.zeros(env.num_envs, 1, device=device)
    env.wind_speed_scalar[env_ids] = wind_speeds_scalar.unsqueeze(-1)
    env.wind_dir[env_ids] = wind_dirs.unsqueeze(-1)

    # wind direction = wind comes from, we want to push in the opposite direction
    wind_force_dir = math_utils.wrap_to_pi(wind_dirs + torch.pi)

    # rads to speed vectors in world frame (x,y,heading)
    wind_speed_x = wind_speeds_scalar * torch.cos(wind_force_dir)
    wind_speed_y = wind_speeds_scalar * torch.sin(wind_force_dir)
    wind_speed_z = torch.zeros_like(wind_speed_x)
    wind_speeds_w = torch.stack([wind_speed_x, wind_speed_y, wind_speed_z], dim=-1)  # (N, 3)

    # transform speeds from world frame to body frame
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectData.root_link_quat_w
    quat_w = asset.data.root_link_quat_w[env_ids]  # (N, 4) [w, x, y, z]
    wind_speeds_b = math_utils.quat_apply_inverse(quat_w, wind_speeds_w)  # (N, 3)

    lin_vel_b = asset.data.root_lin_vel_b[env_ids]  # (N, 3)
    ang_vel_b = asset.data.root_ang_vel_b[env_ids]  # (N, 3)
    nu = torch.cat([lin_vel_b, ang_vel_b], dim=-1)  # (N, 6)

    # compute body frame Fossen aerodynamic tau_wind and store in env buffer
    if not hasattr(env, 'tau_wind'):
        env.tau_wind_= torch.zeros(env.num_envs, 1, device=device)
    env.tau_wind[env_ids] = FossenDynamics.compute_wind_forces(nu, wind_speeds_b)  # (N, 6) [x, y, z, k, m, n]


def randomize_hydrodynamics(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    param_ranges: dict[str, tuple[float, float]],
):
    if not hasattr(env, "sysid_params"):
        return

    for param_name, (min_val, max_val) in param_ranges.items():
        if param_name in env.sysid_params:
            rand_vals = torch.rand(len(env_ids), device=env.device) * (max_val - min_val) + min_val
            env.sysid_params[param_name][env_ids] = rand_vals
