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
    id="System-Id",
    entry_point="astra_rl.tasks.manager_based.system_id.env:Environment",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.system_id_env_cfg:SystemIdEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
