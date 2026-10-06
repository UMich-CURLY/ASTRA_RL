# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Path-Following",
    entry_point="astra_rl.tasks.manager_based.path_following.env:Environment",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.path_following_env_cfg:PathFollowingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
