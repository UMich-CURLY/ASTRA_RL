from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, wrap_to_pi
from isaaclab.envs.mdp import *

from ..water import Water


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.curriculums


def override_value(env, env_ids, data, value, num_steps):
    if env.common_step_counter > num_steps:
        return value
    return modify_term_cfg.NO_CHANGE


def override_value_interp(env, env_ids, current_value, term, start_term, end_term, start_value, end_value):
    if callable(term):
        t = term(env)
    else:
        t = term

    if t == 0.0: return modify_term_cfg.NO_CHANGE  # default
    alpha = (t - start_term) / (end_term - start_term + 1e-16)
    alpha = max(0.0, min(1.0, alpha))

    # handle numbers differently than floats
    if isinstance(start_value, (float, int)):
        return start_value + alpha * (end_value - start_value)

    return tuple(s + alpha * (e - s)
                 for s, e in zip(start_value, end_value))


def get_common_step_counter(env):
    return env.common_step_counter


def get_target_dist(env):
    for term_cfg in env.observation_manager._group_obs_term_cfgs["critic"]:
        if term_cfg.func.__name__ == "target_pos_error_b_xy":
            result_tensor = term_cfg.func(env, **term_cfg.params)
            return result_tensor[..., 2].mean().item()


def enable_waves_steps(env, env_ids, num_steps, wave_scale, directional_spreading,
                       wind_direction_rad, wind_speed_ms, fetch_km, gamma, water_depth_m):
    if env.common_step_counter > num_steps:
        Water.is_waves_enabled = True
        Water.wave_scale = wave_scale
        Water.directional_spreading = directional_spreading
        Water.wind_direction_rad = wind_direction_rad
        Water.wind_speed_ms = wind_speed_ms
        Water.fetch_km = fetch_km
        Water.gamma = gamma
        Water.water_depth_m = water_depth_m
        return True
    return False
