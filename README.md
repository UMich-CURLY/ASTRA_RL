# ASTRA RL

<img src="ASTRA_RL_img.png" width="400">

This repository contains the reinforcement learning extension to ASTRA (ASV Simulator for Transferable Realistic Autonomy) and example tasks.

While this work extends ASTRA, the code is not dependent. The dynamics and code are different between ASTRA and ASTRA_RL. We hope to utilize ASTRA's dynamics 1:1 with warp for parallel environments in the future.

All policies are trained using PPO with [SKRL](https://skrl.readthedocs.io/en/latest/) as the RL library.

# Dependencies

> ### Ubuntu 22.04

> ### Isaac Sim 5.1.0

[Installation docs](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html)

> ### Install Isaac Lab

[installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

## Installation
- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/astra_rl

- Verify that the extension is correctly installed by listing the available tasks:

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/list_envs.py
    ```

## Tasks
- Tasks are found in: "source/astra_rl/astra_rl/tasks/manager_based/". Current supported tasks are:
  - path_following
  - station_keeping
  - system_id

> Note: The docking task is unfinished.

- New tasks can be created by created by using existing systems as a template. Files related to the MDP are task-agnostic, while the physics through env.py must be provided per-task.

## Running A Task

    ```bash
    # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python scripts/skrl/train.py --task=<TASK_NAME>
    ```

> Note: Extra parameters from Isaac Lab can be used (ex: --num_envs=32)
