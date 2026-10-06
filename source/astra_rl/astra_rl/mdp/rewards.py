# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# For a list of all attributes in "robot.data":
# https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectData

# For a list of all atributes in "env.scene":
# https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveScene

# Example for articulated joints:
def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def base_ang_vel_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    ang_vel_z = robot.data.root_ang_vel_b[:, 2] # Root CoM angular velocity in base frame
    return ang_vel_z.abs()
    #return torch.norm(ang_vel_z, dim=1)  # Shape: (num_envs,)


def lin_vel_x(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    root_lin_vel_w = robot.data.root_lin_vel_w
    return root_lin_vel_w[:, 0]


def target_heading_cos(env: ManagerBasedRLEnv, target: float) -> torch.Tensor:
    robot = env.scene["robot"]
    # yaw heading of the base frame (in radians)
    heading = robot.data.heading_w
    return torch.cos(heading - target)


def target_heading_RBF(
    env: ManagerBasedRLEnv,
    target: float,
    sigma: float
) -> torch.Tensor:
    robot = env.scene["robot"]
    heading = robot.data.heading_w
    angle_error = torch.atan2(torch.sin(heading - target), torch.cos(heading - target))

    return torch.exp(-angle_error.pow(2) / (2 * sigma**2))


def target_heading_RBF_epsilon(
    env: ManagerBasedRLEnv,
    target: float,
    sigma: float,
    epsilon: float
) -> torch.Tensor:
    robot = env.scene["robot"]
    heading = robot.data.heading_w
    angle_error = torch.atan2(torch.sin(heading - target), torch.cos(heading - target))
    abs_error = torch.abs(angle_error)

    excess_error = torch.clamp(abs_error - epsilon, min=0.0)
    return torch.exp(-excess_error.pow(2) / (2 * sigma**2))


def target_quat_z(env: ManagerBasedRLEnv, target: float) -> torch.Tensor:
    robot = env.scene["robot"]
    root_ang_z_w = robot.data.root_quat_w[:, 3]
    return torch.square(root_ang_z_w - target)


def origin_pos_x(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    # (N, (x, y, z)) of the origin in the world frame
    origin = env.scene.env_origins[:, :3]
    # (N, (x, y, z, q_w, q_x, q_y, q_z)) in the world frame
    pos = robot.data.root_link_pose_w[:, :3]
    return (pos - origin)[:, 0]


def target_pos_xy_euclidean(
    env: ManagerBasedRLEnv,
    target: tuple[float, float]
) -> torch.Tensor:
    robot = env.scene["robot"]
    origin = env.scene.env_origins[:, :2]
    pos = robot.data.root_link_pose_w[:, :2]

    target_t = torch.as_tensor(target, device=pos.device, dtype=pos.dtype)
    error = pos - (origin + target_t)
    return torch.norm(error, dim=-1)


def target_pos_xy_RBF(
    env: ManagerBasedRLEnv,
    target: tuple[float, float],
    sigma: float
) -> torch.Tensor:
    robot = env.scene["robot"]
    origin = env.scene.env_origins[:, :2]
    pos = robot.data.root_link_pose_w[:, :2]

    target_t = torch.as_tensor(target, device=pos.device, dtype=pos.dtype)
    error = pos - (origin + target_t)
    dist2 = torch.sum(error.pow(2), dim=1)
    return torch.exp(-dist2 / (2 * sigma**2))


def target_pos_xy_RBF_dist(
    env: ManagerBasedRLEnv,
    target: tuple[float, float],
    sigma: float,
    alpha: float
) -> torch.Tensor:
    robot = env.scene["robot"]
    origin = env.scene.env_origins[:, :2]
    pos = robot.data.root_link_pose_w[:, :2]

    target_t = torch.as_tensor(target, device=pos.device, dtype=pos.dtype)
    error = pos - (origin + target_t)
    dist = torch.norm(error, dim=-1)
    return torch.exp(-dist.pow(2) / (2 * sigma**2)) - alpha * dist


def target_vel_xy_l2(env: ManagerBasedRLEnv, target: tuple[float, float]) -> torch.Tensor:
    vel = env.scene["robot"].data.root_link_vel_w[:, :2]
    target_t = torch.as_tensor(target, device=vel.device, dtype=vel.dtype)
    return torch.sum(torch.square(vel - target_t), dim=1)


def target_acc_xy_l2(env: ManagerBasedRLEnv, target: tuple[float, float, float]) -> torch.Tensor:
    lin_acc = env.scene["robot"].data.body_com_lin_acc_w[:, 0, :2]
    ang_acc = env.scene["robot"].data.body_com_ang_acc_w[:, 0, 2].unsqueeze(-1)
    acc = torch.cat([lin_acc, ang_acc], dim=-1)
    target_t = torch.as_tensor(target, device=lin_acc.device, dtype=lin_acc.dtype)
    return torch.sum(torch.square(acc - target_t), dim=1)


def reached_goal_flag(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "_reached_goal"):
        env._reached_goal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env._reached_goal.float()


# Action shaping rewards


def thruster_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    action_term = env.action_manager.get_term("thrusters")

    u_t = action_term.processed_action / action_term.max_forces

    return torch.sum(torch.square(u_t), dim=1)


def thruster_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    action_term = env.action_manager.get_term("thrusters")
    
    u_t = action_term.processed_action / action_term.max_forces
    u_t_minus_1 = action_term.prev_processed_action / action_term.max_forces
    
    return torch.sum(torch.square(u_t - u_t_minus_1), dim=1)


def thruster_rate2_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    action_term = env.action_manager.get_term("thrusters")
    
    u_t = action_term.processed_action / action_term.max_forces
    u_t_minus_1 = action_term.prev_processed_action / action_term.max_forces
    u_t_minus_2 = action_term.prev2_processed_action / action_term.max_forces
    
    acc = u_t - 2 * u_t_minus_1 + u_t_minus_2
    
    return torch.sum(torch.square(acc), dim=1)


def recorded_pos_xy_RBF(env, sigma: float) -> torch.Tensor:
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    pos_target_w = env.command_manager.get_command("trajectory")[:, :2]  # [x, y]
    dist = torch.norm(pos_target_w - pos_w, p=2, dim=-1)
    return torch.exp(-dist**2 / sigma**2)


def recorded_heading_RBF(env, sigma: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    imu_orientation = env.scene[asset_cfg.name].data.quat_w
    yaw = euler_xyz_from_quat(imu_orientation)[2]
    yaw_target = env.command_manager.get_command("trajectory")[:, 2]  # [yaw]
    err_yaw = torch.atan2(torch.sin(yaw_target - yaw), torch.cos(yaw_target - yaw))
    return torch.exp(-err_yaw**2 / sigma**2)


def recorded_vel_b_xy_RBF(env, sigma: float) -> torch.Tensor:
    vel_b = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    vel_target_b = env.command_manager.get_command("trajectory")[:, 3:5]  # [u, v]
    dist = torch.norm(vel_target_b - vel_b, p=2, dim=-1)
    return torch.exp(-dist**2 / sigma**2)


def recorded_heading_rate_RBF(env, sigma: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    imu_r = env.scene[asset_cfg.name].data.ang_vel_b[:, 2]
    r_target = env.command_manager.get_command("trajectory")[:, 5]  # [r]
    err_r = r_target - imu_r
    return torch.exp(-err_r**2 / sigma**2)


def perturbation_surge_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(env.tau_perturb[:, 0])


def perturbation_sway_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(env.tau_perturb[:, 1])


def perturbation_yaw_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.square(env.tau_perturb[:, 5])


# Path following rewards


def cross_track_error_RBF(env: ManagerBasedRLEnv, sigma: float) -> torch.Tensor:
    waypoints_w = env.command_manager.get_command("path")  # (N, K, 3)
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2]

    dists = torch.norm(waypoints_w[:, :, :2] - pos_w.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(dists, dim=-1)

    batch_indices = torch.arange(env.num_envs, device=env.device)
    next_idx = torch.clamp(closest_idx + 1, max=waypoints_w.shape[1] - 1)

    p_a = waypoints_w[batch_indices, closest_idx, :2]
    p_b = waypoints_w[batch_indices, next_idx, :2]

    tangent = p_b - p_a
    tangent_norm = torch.norm(tangent, dim=-1) + 1e-6
    v = pos_w - p_a

    cross_prod = v[:, 0] * tangent[:, 1] - v[:, 1] * tangent[:, 0]
    cte = cross_prod / tangent_norm

    return torch.exp(-torch.square(cte) / (2 * sigma**2))


def path_heading_error_RBF(env: ManagerBasedRLEnv, sigma: float) -> torch.Tensor:
    robot = env.scene["robot"]
    waypoints_w = env.command_manager.get_command("path")  # (N, K, 3)
    pos_w = robot.data.root_link_pos_w[:, :2]

    dists = torch.norm(waypoints_w[:, :, :2] - pos_w.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(dists, dim=-1)

    batch_indices = torch.arange(env.num_envs, device=env.device)
    next_idx = torch.clamp(closest_idx + 1, max=waypoints_w.shape[1] - 1)

    p_a = waypoints_w[batch_indices, closest_idx, :2]
    p_b = waypoints_w[batch_indices, next_idx, :2]
    tangent = p_b - p_a

    path_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])

    robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2]

    err_yaw = torch.atan2(torch.sin(path_yaw - robot_yaw), torch.cos(path_yaw - robot_yaw))

    return torch.exp(-torch.square(err_yaw) / (2 * sigma**2))


# def path_progress_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
#     waypoints_w = env.command_manager.get_command("path")  # (N, K, 3)
#     pos_w = env.scene["robot"].data.root_link_pos_w[:, :2]
#     vel_w = env.scene["robot"].data.root_link_lin_vel_w[:, :2]

#     dists = torch.norm(waypoints_w[:, :, :2] - pos_w.unsqueeze(1), dim=-1)
#     closest_idx = torch.argmin(dists, dim=-1)

#     batch_indices = torch.arange(env.num_envs, device=env.device)
#     next_idx = torch.clamp(closest_idx + 1, max=waypoints_w.shape[1] - 1)

#     p_a = waypoints_w[batch_indices, closest_idx, :2]
#     p_b = waypoints_w[batch_indices, next_idx, :2]

#     tangent = p_b - p_a
#     tangent_norm = torch.norm(tangent, dim=-1, keepdim=True) + 1e-6
#     tangent_dir = tangent / tangent_norm

#     progress = (vel_w * tangent_dir).sum(dim=-1)

#     return progress


def path_progress_RBF(env: ManagerBasedRLEnv, speed: float, sigma: float) -> torch.Tensor:
    waypoints_w = env.command_manager.get_command("path")  # (N, K, 3)
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2]
    vel_w = env.scene["robot"].data.root_link_lin_vel_w[:, :2]

    dists = torch.norm(waypoints_w[:, :, :2] - pos_w.unsqueeze(1), dim=-1)
    closest_idx = torch.argmin(dists, dim=-1)

    batch_indices = torch.arange(env.num_envs, device=env.device)
    next_idx = torch.clamp(closest_idx + 1, max=waypoints_w.shape[1] - 1)

    p_a = waypoints_w[batch_indices, closest_idx, :2]
    p_b = waypoints_w[batch_indices, next_idx, :2]

    tangent = p_b - p_a
    tangent_norm = torch.norm(tangent, dim=-1, keepdim=True) + 1e-6
    tangent_dir = tangent / tangent_norm

    v_tangent = (vel_w * tangent_dir).sum(dim=-1)

    speed_error = v_tangent - speed
    return torch.exp(-torch.square(speed_error) / (2 * sigma**2))


def path_goal_reached_time_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    goal_reached = env.termination_manager.get_term("path_goal_reached")
    steps_remaining = env.max_episode_length - env.episode_length_buf

    return (goal_reached * steps_remaining).float()
