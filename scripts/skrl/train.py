# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    help=(
        "Name of the RL algorithm to use (e.g. AMP, DDPG, IPPO, MAPPO, PPO, SAC, TD3, etc.) "
        "when several algorithms exist for the same task. For a more specific selection, use the argument --agent."
    ),
)
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import gymnasium as gym
import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "2.0.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# import logger
logger = logging.getLogger(__name__)

import astra_rl.tasks  # noqa: F401

# WandB Logging Bug: https://github.com/isaac-sim/IsaacLab/issues/5252
# ---------------------------------------------------------------------------
import wandb
from skrl.agents.torch import Agent

_orig_write_tracking_data = Agent.write_tracking_data

def _patched_write_tracking_data(self, *, timestep: int, timesteps: int) -> None:
    # Collect aggregated metrics BEFORE the original method clears tracking_data.
    wandb_data: dict = {}
    for k, v in self.tracking_data.items():
        if k.endswith("(min)"):
            value = np.min(v)
        elif k.endswith("(max)"):
            value = np.max(v)
        else:
            value = np.mean(v)

        # Avoid NaN/Infs
        if np.isfinite(float(value)):
            wandb_data[k] = float(value)

    # Original behaviour: write to TensorBoard and clear tracking_data.
    _orig_write_tracking_data(self, timestep=timestep, timesteps=timesteps)

    # Additionally push metrics to wandb if a run is active.
    if wandb.run is not None and wandb_data:
        wandb.log(wandb_data, step=timestep)

Agent.write_tracking_data = _patched_write_tracking_data
# ---------------------------------------------------------------------------

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # The Ray Tune workflow extracts experiment name using the logging line below, hence,
    # do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # wandb Observation term logging
    # ---------------------------------------------------------------------------
    class WandbObsLogger(gym.Wrapper):
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if wandb.run is not None:
                log_dict = {}
                # Check if this is a ManagerBasedRLEnv that has an observation manager
                has_obs_mgr = hasattr(self.env.unwrapped, "observation_manager")
                
                if isinstance(obs, dict):
                    for group_key, obs_tensor in obs.items():
                        # Calculate mean over parallel environments
                        mean_obs = obs_tensor.mean(dim=0).cpu().numpy() if hasattr(obs_tensor, "mean") else np.mean(obs_tensor, axis=0)
                        
                        # Reconstruct the terms from the concatenated tensor using Isaac Lab's Observation Manager
                        if has_obs_mgr and group_key in self.env.unwrapped.observation_manager.active_terms:
                            obs_mgr = self.env.unwrapped.observation_manager
                            term_names = list(obs_mgr.active_terms[group_key])
                            term_dims = obs_mgr.group_obs_term_dim[group_key]
                            
                            current_idx = 0
                            for term_name, term_shape in zip(term_names, term_dims):
                                # Calculate how many elements this term contributes to the flat array
                                num_elements = int(np.prod(term_shape))
                                term_values = mean_obs[current_idx : current_idx + num_elements]
                                
                                if num_elements > 1:
                                    for i, val in enumerate(term_values):
                                        log_dict[f"obs/{group_key}/{term_name}_dim_{i}"] = val
                                else:
                                    log_dict[f"obs/{group_key}/{term_name}"] = float(term_values[0])
                                    
                                current_idx += num_elements
                        else:
                            # Fallback if group metadata isn't found
                            if mean_obs.ndim > 0:
                                for idx, val in enumerate(mean_obs.flatten()):
                                    log_dict[f"obs/{group_key}/dim_{idx}"] = val
                            else:
                                log_dict[f"obs/{group_key}"] = float(mean_obs)
                else:
                    # Fallback for purely flattened observations (e.g., DirectRLEnv)
                    mean_obs = obs.mean(dim=0).cpu().numpy() if hasattr(obs, "mean") else np.mean(obs, axis=0)
                    for idx, val in enumerate(mean_obs.flatten()):
                        log_dict[f"obs/dim_{idx}"] = val

                # non-observation metrics logging
                unwrapped_env = self.env.unwrapped

                if hasattr(unwrapped_env, 'wind_speed_scalar'):
                    speeds = unwrapped_env.wind_speed_scalar
                    log_dict["metrics/wind_speeds_mean"] = float(speeds.mean().item())
                    log_dict["metrics/wind_speeds_max"] = float(speeds.max().item())
                    log_dict["metrics/wind_speeds_hist"] = wandb.Histogram(speeds.detach().cpu().numpy())

                if hasattr(unwrapped_env, "wind_dir_current"):
                    dirs = unwrapped_env.wind_dir_current
                    log_dict["metrics/wind_dirs_hist"] = wandb.Histogram(dirs.detach().cpu().numpy())

                if hasattr(unwrapped_env, 'abs_heading_error_mean'):
                    log_dict["metrics/abs_heading_error_mean"] = unwrapped_env.abs_heading_error_mean

                # log perturbations
                if hasattr(unwrapped_env, 'tau_perturb'):
                    surge_perturb = unwrapped_env.tau_perturb[:, 0].detach().cpu().numpy()
                    sway_perturb = unwrapped_env.tau_perturb[:, 1].detach().cpu().numpy()
                    yaw_perturb = unwrapped_env.tau_perturb[:, 5].detach().cpu().numpy()

                    log_dict["metrics/perturb_surge_mean"] = float(surge_perturb.mean())
                    log_dict["metrics/perturb_sway_mean"] = float(sway_perturb.mean())
                    log_dict["metrics/perturb_yaw_mean"] = float(yaw_perturb.mean())

                # log sysId params
                if hasattr(unwrapped_env, "sysid_params"):
                    for param_name, param_val in unwrapped_env.sysid_params.items():
                        if hasattr(param_val, "mean"):
                            log_dict[f"sysid_params/{param_name}"] = float(param_val.mean().item())
                        else:
                            log_dict[f"sysid_params/{param_name}"] = float(param_val)

                # Commit=False keeps the logs synced with the agent's step cycles
                wandb.log(log_dict, commit=False)

            return obs, reward, terminated, truncated, info

    env = WandbObsLogger(env)
    # ---------------------------------------------------------------------------

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # log environment config to wandb
    if wandb.run is not None:
        # Scene config
        wandb.config.update({
            "env_num_envs": env_cfg.scene.num_envs,
            "decimation": env_cfg.decimation,
            "episode_length_s": env_cfg.episode_length_s,
            "sim_dt": env_cfg.sim.dt,
        })
        
        # Reward weights
        if hasattr(env_cfg, 'rewards'):
            reward_weights = {}
            for key, term in env_cfg.rewards.__dict__.items():
                if hasattr(term, 'weight'):
                    reward_weights[f"reward_weight_{key}"] = term.weight
            wandb.config.update(reward_weights)

    # run training
    runner.run()

    # commit partial wandb logs and finish
    if wandb.run is not None:
        wandb.log({}, commit=True)
        wandb.finish()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
