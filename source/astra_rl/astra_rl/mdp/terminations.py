# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# For a list of all attributes in robot.data:
# https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectData


def heading_limit(env: ManagerBasedRLEnv, range_rad: tuple) -> torch.Tensor:
    """Terminate if robot's heading is out of range"""
    robot = env.scene["robot"]
    # yaw heading of the base frame (in radians)
    heading = robot.data.heading_w[:]
    # return true for elements (robots) outside the allowable range
    return (heading < range_rad[0]) | (heading > range_rad[1])


def backward_distance_limit(env: ManagerBasedRLEnv, distance: float = 2.0) -> torch.Tensor:
    """Terminate if the robot moves backward more than distance_m"""
    robot = env.scene["robot"]
    origin = env.scene.env_origins[:, :3]
    pos = robot.data.root_link_pose_w[:, :3]
    error_x = (pos - origin)[:, 0]
    # return true for robots that have moved too far backward
    return (error_x < -distance)


def origin_distance_limit(env: ManagerBasedRLEnv, distance: float = 2.0) -> torch.Tensor:
    """Terminate if the robot is more than distance_m from the origin"""
    robot = env.scene["robot"]
    origin = env.scene.env_origins[:, :3]
    pos = robot.data.root_link_pose_w[:, :3]
    error = torch.norm(pos - origin, dim=1)
    # return true for robots that have moved too far from the origin
    return (error > distance)


def recorded_dist_limit(env: ManagerBasedRLEnv, distance: float) -> torch.Tensor:
    """Terminate if the robot is more than distance_m from the trajectory."""
    pos_w = env.scene["robot"].data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    pos_target_w = env.command_manager.get_command("trajectory")[:, :2]  # [x, y]
    dist = torch.norm(pos_target_w - pos_w, p=2, dim=-1)
    return (dist > distance)


def reached_goal_termination(
    env: ManagerBasedRLEnv,
    pos_target: tuple,
    heading_target: float,
    pos_epsilon: float,
    heading_epsilon: float,
    max_pos_vel: float,
    max_ang_vel: float
) -> torch.Tensor:
    robot = env.scene["robot"]

    pos = robot.data.root_link_pose_w[:, :2]
    env_origin = env.scene.env_origins[:, :2]  # use env origin instead of world origin
    pos_target_t = torch.as_tensor(pos_target, device=pos.device, dtype=pos.dtype)
    pos_error = torch.norm((pos - env_origin) - pos_target_t, dim=1)

    heading = euler_xyz_from_quat(robot.data.root_link_quat_w)[2]
    heading_error = torch.abs(heading - heading_target)

    # initialize the buffer for maintaining the state
    if not hasattr(env, "_reached_goal"):
        env._reached_goal = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    # reset flag to False at episode start
    reset_env_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
    if len(reset_env_ids) > 0:
        env._reached_goal[reset_env_ids] = False

    current_reached = (pos_error <= pos_epsilon) & (heading_error <= heading_epsilon)
    env._reached_goal = current_reached
    # env._reached_goal = env._reached_goal | current_reached
    return current_reached
    
    # # update flags
    # is_above_target = current_height >= target_height
    # env._jump_success_latch = env._jump_success_latch | is_above_target

    # # detect terminal step (include or exclude timeouts as desired)
    # terminated = env.termination_manager.terminated
    # time_outs = env.termination_manager.time_outs
    # is_terminal_step = terminated | time_outs  # or just `terminated` if you prefer
    # return (env._jump_success_latch & is_terminal_step).float()


# Path following terminations


def cross_track_limit(env: ManagerBasedRLEnv, distance: float) -> torch.Tensor:
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
    tangent_norm = torch.norm(tangent, dim=-1) + 1e-6
    v = pos_w - p_a

    cross_prod = v[:, 0] * tangent[:, 1] - v[:, 1] * tangent[:, 0]
    cte = (cross_prod / tangent_norm).abs()

    return cte > distance


def path_goal_reached(env: ManagerBasedRLEnv, pos_epsilon: float, heading_epsilon: float) -> torch.Tensor:
    robot = env.scene["robot"]
    waypoints_w = env.command_manager.get_command("path")  # (N, K, 3)
    pos_w = robot.data.root_link_pos_w[:, :2]

    final_waypoint = waypoints_w[:, -1, :2]
    prev_waypoint = waypoints_w[:, -2, :2]

    pos_error = torch.norm(final_waypoint - pos_w, dim=-1)

    tangent = final_waypoint - prev_waypoint
    target_yaw = torch.atan2(tangent[:, 1], tangent[:, 0])

    robot_yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2]

    heading_error = wrap_to_pi(target_yaw - robot_yaw).abs()

    if not hasattr(env, "path_goal_reached"):
        env.path_goal_reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    reset_env_ids = (env.episode_length_buf == 0).nonzero(as_tuple=False).flatten()
    if len(reset_env_ids) > 0:
        env.path_goal_reached[reset_env_ids] = False

    current_reached = (pos_error <= pos_epsilon) & (heading_error <= heading_epsilon)
    env.path_goal_reached = current_reached

    return current_reached
