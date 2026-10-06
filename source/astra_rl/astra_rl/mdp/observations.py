# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, wrap_to_pi


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Robot state from root link


def robot_pos_w_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Position of the robot in the x,y axis in the env frame [m]"""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectData.root_link_pos_w
    robot_pos_w = env.scene["robot"].data.root_link_pos_w
    robot_origin = env.scene.env_origins[:, :2]
    return robot_pos_w[:, :2] - robot_origin  # [num_envs, 2]


def robot_vel_w_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Velocity of the robot in the x,y axis in the world frame [m/s]"""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectData.root_link_lin_vel_b
    robot_vel_w = env.scene["robot"].data.root_link_lin_vel_w
    return robot_vel_w[:, :2]  # [num_envs, 2]


def robot_vel_b_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Velocity of the robot in the x,y axis in the body frame [m/s]"""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectData.root_link_lin_vel_w
    robot_vel_b = env.scene["robot"].data.root_link_lin_vel_b
    return robot_vel_b[:, :2]  # [num_envs, 2]


# Robot state from imu


def imu_heading(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Heading angle (-pi, pi] in the sensor frame [rad]"""
    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    yaw = euler_xyz_from_quat(imu_orientation)[2]
    return yaw.unsqueeze(-1)  # [num_envs, 1]


def imu_heading_sin_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sin/Cos of heading angle (-pi, pi] in the sensor frame [rad]"""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/_modules/isaaclab/envs/mdp/observations.html#imu_orientation
    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    yaw = euler_xyz_from_quat(imu_orientation)[2]
    yaw_sin = yaw.sin().unsqueeze(-1)  # [num_envs, 1]
    yaw_cos = yaw.cos().unsqueeze(-1)  # [num_envs, 1]
    return torch.cat([yaw_sin, yaw_cos], dim=-1)  # [num_envs, 2]


def imu_heading_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Angular velocity of heading in the sensor frame [rad/s]"""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/_modules/isaaclab/envs/mdp/observations.html#imu_ang_vel
    imu_ang_vel = env.scene[asset_cfg.name].data.ang_vel_b
    return imu_ang_vel[:, 2:3]  # [num_envs, 1]


def imu_acc_b_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Linear acceleration of the robot in the x,y axis in the sensor frame [m/s^2]"""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/_modules/isaaclab/envs/mdp/observations.html#imu_lin_acc
    imu_lin_acc = env.scene[asset_cfg.name].data.lin_acc_b
    return imu_lin_acc[:, :2]  # [num_envs, 2]


# Task specific observations


def lidar_ray_hits_depth(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.MultiMeshRayCasterData
    ray_hits_w = env.scene[asset_cfg.name].data.ray_hits_w  # (N, B, 3)
    sensor_pos_w = env.scene[asset_cfg.name].data.pos_w

    relative_vectors = ray_hits_w - sensor_pos_w.unsqueeze(1)  # (N, B, 3)
    distances = torch.linalg.norm(relative_vectors, dim=-1)  # Shape: (N, B)

    max_distance = env.scene[asset_cfg.name].cfg.max_distance
    clamped_distances = torch.nan_to_num(distances, posinf=max_distance, neginf=0.0, nan=max_distance)

    return clamped_distances


def target_pos_error_w_xy(
    env: ManagerBasedRLEnv,
    target: tuple[float, float]
) -> torch.Tensor:
    """Position and distance error to target x,y (w) in the world frame [m]"""
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    target_t = torch.as_tensor(target, device=pos_w.device, dtype=pos_w.dtype)
    error_w = target_t - pos_w  # error in world frame (N, 2)

    # norm distance
    dist = torch.norm(error_w, dim=-1, keepdim=True)

    # [dx_w, dy_w, dist]
    return torch.cat([error_w[:, 0], error_w[:, 1], dist], dim=-1)


def target_pos_error_b_xy_dist(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: tuple[float, float]
) -> torch.Tensor:
    """Position and distance error to target x,y (w) in the robot body frame [m]"""
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    target_t = torch.as_tensor(target, device=pos_w.device, dtype=pos_w.dtype)
    error_w = target_t - pos_w  # error in world frame (N, 2)

    # rotate error to body frame using imu yaw
    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    yaw = euler_xyz_from_quat(imu_orientation)[2]
    
    # 2D rotation
    cos_yaw = torch.cos(-yaw)  # (N,)
    sin_yaw = torch.sin(-yaw)  # (N,)
    dx_b = (error_w[:, 0] * cos_yaw - error_w[:, 1] * sin_yaw).unsqueeze(-1)
    dy_b = (error_w[:, 0] * sin_yaw + error_w[:, 1] * cos_yaw).unsqueeze(-1)

    # norm distance
    dist = torch.norm(error_w, dim=-1, keepdim=True)

    # [dx_b, dy_b, dist]
    return torch.cat([dx_b, dy_b, dist], dim=-1)


def target_heading_error_sin_cos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target: float,
    noise_gaussian_std: float,
) -> torch.Tensor:
    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    yaw = euler_xyz_from_quat(imu_orientation)[2]

    error = target - yaw
    heading_error = torch.atan2(torch.sin(error), torch.cos(error))
    env.unwrapped.abs_heading_error_mean = heading_error.abs().mean().item()

    if noise_gaussian_std > 0.0:
        error += torch.randn_like(error) * noise_gaussian_std

    return torch.stack([torch.sin(error), torch.cos(error)], dim=-1)


# Action space observations


def last_processed_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    action_term = env.action_manager.get_term("thrusters")
    return action_term.processed_action


def last_processed_actions(env: ManagerBasedRLEnv, history_length: int, stride: int) -> torch.Tensor:
    action_term = env.action_manager.get_term("thrusters")
    
    if history_length <= 1:
        return action_term.processed_action

    buffer_max_size = action_term._max_buffer_size
    current_idx = action_term._buffer_idx
    history_buffer = action_term._history_buffer  # (buffer_max_size, N, T)

    steps_back = torch.arange(history_length, device=history_buffer.device) * stride
    indices = (current_idx - steps_back) % buffer_max_size
    history_tensor = history_buffer[indices]  # (history_length, N, T)

    history_tensor = history_tensor.permute(1, 0, 2)  # (N, history_length, T)
    return history_tensor.flatten(start_dim=1)  # (N, history_length*T)


# Environmental observations


def total_hydro_wind_force_b_2d(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Total hydro wind force (x,y,yaw) in the body frame [N]"""
    tau_total = getattr(env, "tau_total", torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32))
    return torch.stack([tau_total[:, 0], tau_total[:, 1], tau_total[:, 5]], dim=-1)


def wind_force_b_2d(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Wind force (x,y,yaw) (after processing raw force) in the body frame [N]"""
    tau_wind = getattr(env, "tau_wind", torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32))
    return torch.stack([tau_wind[:, 0], tau_wind[:, 1], tau_wind[:, 5]], dim=-1)


def wind_force_w_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Wind force (after processing raw force) in the world frame [N]"""
    tau_wind = getattr(env, "tau_wind", torch.zeros((env.num_envs, 6), device=env.device, dtype=torch.float32))
    tau_wind = tau_wind[:, :2]

    yaw = env.scene["robot"].data.heading_w
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    force_w_x = tau_wind[:, 0] * cos_yaw - tau_wind[:, 1] * sin_yaw
    force_w_y = tau_wind[:, 0] * sin_yaw + tau_wind[:, 1] * cos_yaw

    return torch.stack([force_w_x, force_w_y], dim=-1)


def wind_opp_direction_b_sin_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Downwind direction (going to; not coming from) in the body frame as sin/cos [rad]"""
    wind_dir_w = wrap_to_pi(getattr(env, "wind_dir", torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)))
    
    # reverse wind direction (get direction the force is being applied)
    wind_dir_w = wrap_to_pi(wind_dir_w + torch.pi)
    
    # rotate world frame to body frame using imu yaw
    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    yaw = euler_xyz_from_quat(imu_orientation)[2]
    wind_dir_b = wrap_to_pi(wind_dir_w - yaw.unsqueeze(-1))

    return torch.cat([torch.sin(wind_dir_b), torch.cos(wind_dir_b)], dim=-1)


def wind_direction_b_sin_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Wind direction (coming from) in the body frame as sin/cos [rad]"""
    wind_dir_w = wrap_to_pi(getattr(env, "wind_dir", torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.float32)))
    
    # rotate world frame to body frame using imu yaw
    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    yaw = euler_xyz_from_quat(imu_orientation)[2]
    wind_dir_b = wrap_to_pi(wind_dir_w - yaw.unsqueeze(-1))

    return torch.cat([torch.sin(wind_dir_b), torch.cos(wind_dir_b)], dim=-1)


# Trajectory tracking observations


def recorded_pos_error_b_xy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    pos_target_w = env.command_manager.get_command("trajectory")[:, :2]  # [x, y]
    err_pos_w = pos_target_w - pos_w  # (N, 2)

    # rotate error to body frame using imu yaw
    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    yaw = euler_xyz_from_quat(imu_orientation)[2]
    cos_yaw = torch.cos(-yaw)  # (N,)
    sin_yaw = torch.sin(-yaw)  # (N,)

    dx_b = err_pos_w[:, 0] * cos_yaw - err_pos_w[:, 1] * sin_yaw  # Shape: (N,)
    dy_b = err_pos_w[:, 0] * sin_yaw + err_pos_w[:, 1] * cos_yaw  # Shape: (N,)
    return torch.stack([dx_b, dy_b], dim=-1)  # (N, 2)


def recorded_vel_error_b_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    vel_b = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_target_b = env.command_manager.get_command("trajectory")[:, 3:5]  # [u, v]
    return vel_target_b - vel_b  # (N, 2)


def recorded_heading_error_sin_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, noise_gaussian_std: float) -> torch.Tensor:
    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    yaw = euler_xyz_from_quat(imu_orientation)[2]
    if noise_gaussian_std > 0.0:
        yaw += torch.randn_like(yaw) * noise_gaussian_std
    yaw_target = env.command_manager.get_command("trajectory")[:, 2]  # [yaw]
    err_yaw = torch.atan2(torch.sin(yaw_target - yaw), torch.cos(yaw_target - yaw))
    env.unwrapped.abs_heading_error_mean = err_yaw.abs().mean().item()
    return torch.stack([torch.sin(err_yaw), torch.cos(err_yaw)], dim=-1)  # (N, 2)


def recorded_heading_rate_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    imu_r = env.scene[asset_cfg.name].data.ang_vel_b[:, 2:3]
    r_target = env.command_manager.get_command("trajectory")[:, 5:6]  # [r]
    return r_target - imu_r  # (N, 1)


# Path following observations


def local_path_waypoints_dxdy(env: ManagerBasedRLEnv, num_lookahead: int, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    waypoints_w = env.command_manager.get_command("path")  # (N, K, 3)
    num_envs, num_waypoints, _ = waypoints_w.shape
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2]

    dists = torch.norm(waypoints_w[:, :, :2] - pos_w.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(dists, dim=-1)  # (N,)

    batch_indices = torch.arange(num_envs, device=env.device, dtype=torch.long)
    offsets = torch.arange(num_lookahead, device=env.device, dtype=torch.long)

    lookahead_idx = (closest_idx.unsqueeze(-1) + offsets).clamp(max=num_waypoints - 1).long()

    target_pts_w = waypoints_w[batch_indices.unsqueeze(-1), lookahead_idx, :2]  # (N, num_lookahead, 2)

    pos_diff = target_pts_w - pos_w.unsqueeze(1)
    yaw = euler_xyz_from_quat(env.scene[asset_cfg.name].data.quat_w)[2]

    cos_yaw = torch.cos(-yaw).unsqueeze(-1)
    sin_yaw = torch.sin(-yaw).unsqueeze(-1)

    dx = pos_diff[..., 0] * cos_yaw - pos_diff[..., 1] * sin_yaw
    dy = pos_diff[..., 0] * sin_yaw + pos_diff[..., 1] * cos_yaw

    return torch.stack([dx, dy], dim=-1).reshape(num_envs, -1)  # (N, num_lookahead*2)


def cross_track_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    waypoints_w = env.command_manager.get_command("path")  # (N, K, 3)
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2]

    dists = torch.norm(waypoints_w[:, :, :2] - pos_w.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(dists, dim=-1)

    num_envs = env.num_envs
    batch_indices = torch.arange(num_envs, device=env.device)
    next_idx = torch.clamp(closest_idx + 1, max=waypoints_w.shape[1] - 1)

    p_a = waypoints_w[batch_indices, closest_idx, :2]
    p_b = waypoints_w[batch_indices, next_idx, :2]

    tangent = p_b - p_a
    tangent_norm = torch.norm(tangent, dim=-1, keepdim=True) + 1e-6
    v = pos_w - p_a

    cross_prod = v[:, 0] * tangent[:, 1] - v[:, 1] * tangent[:, 0]
    signed_cte = cross_prod / tangent_norm.squeeze(-1)

    return signed_cte.unsqueeze(-1)  # (N, 1)


def path_heading_error_sin_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, noise_gaussian_std: float) -> torch.Tensor:
    waypoints_w = env.command_manager.get_command("path")  # (N, K, 3)
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2]

    dists = torch.norm(waypoints_w[:, :, :2] - pos_w.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(dists, dim=-1)

    num_envs = env.num_envs
    batch_indices = torch.arange(num_envs, device=env.device)
    next_idx = torch.clamp(closest_idx + 1, max=waypoints_w.shape[1] - 1)

    p_a = waypoints_w[batch_indices, closest_idx, :2]
    p_b = waypoints_w[batch_indices, next_idx, :2]
    tangent = p_b - p_a

    path_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])

    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    robot_yaw = euler_xyz_from_quat(imu_orientation)[2]

    heading_error = wrap_to_pi(path_yaw - robot_yaw)

    env.unwrapped.abs_heading_error_mean = heading_error.abs().mean().item()

    if noise_gaussian_std > 0.0:
        heading_error += torch.randn_like(heading_error) * noise_gaussian_std

    return torch.stack([torch.sin(heading_error), torch.cos(heading_error)], dim=-1)  # (N, 2)
