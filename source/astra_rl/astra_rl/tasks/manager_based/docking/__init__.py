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
    id="Docking",
    entry_point="astra_rl.tasks.manager_based.docking.env:Environment",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.docking_env_cfg:DockingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
