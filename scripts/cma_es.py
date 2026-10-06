import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Optimize SysID parameters using CMA-ES.")
parser.add_argument("--num_envs", type=int, default=256, help="Number of environments / CMA-ES population size.")
parser.add_argument("--task", type=str, default="System-Id", help="Name of the task.")
parser.add_argument("--seed", type=int, default=1, help="Seed used for the environment and CMA-ES.")
parser.add_argument("--max_generations", type=int, default=100, help="Maximum number of CMA-ES generations.")
parser.add_argument("--sigma0", type=float, default=0.2, help="Initial standard deviation for CMA-ES.")
parser.add_argument("--agent", type=str, default="skrl_ppo_cfg_entry_point", help="Task config entry point.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import gymnasium as gym
import torch
import cma
import wandb

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

import isaaclab_tasks  # noqa: F401
# from isaaclab_tasks.utils.hydra import hydra_task_config
from isaaclab_tasks.utils import parse_env_cfg

# import logger
logger = logging.getLogger(__name__)

import astra_rl.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)

    seed = args_cli.seed if args_cli.seed != -1 else random.randint(0, 10000)
    env_cfg.seed = seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    log_root_path = os.path.abspath(os.path.join("logs", "cma_es", args_cli.task or "sys_id"))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_CMA_ES"
    log_dir = os.path.join(log_root_path, log_dir)
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)

    env_cfg.log_dir = log_dir
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)

    print(f"[INFO] Creating environment: {args_cli.task} with {env_cfg.scene.num_envs} envs.")
    env = gym.make(args_cli.task, cfg=env_cfg)
    unwrapped_env = env.unwrapped

    # wandb
    wandb.init(
        project="astra_rl-scripts_cmaes",
        name=os.path.basename(log_dir),
        config={
            "num_envs": env_cfg.scene.num_envs,
            "sim_dt": env_cfg.sim.dt,
            "decimation": env_cfg.decimation,
            "episode_length_s": env_cfg.episode_length_s,
            "sigma0": args_cli.sigma0,
            "max_generations": args_cli.max_generations,
        }
    )

    pop_size = env_cfg.scene.num_envs

    # [SYSID]
    num_params = getattr(unwrapped_env, "num_of_sysID_params", 10)
    # active_param_names = ["Iz", "X_u", "Y_v", "N_r", "X_uu", "Y_vv", "N_rr", "c_rev", "c_u", "c_lag"]
    active_param_names = ["Iz", "X_u", "Y_v", "N_r", "X_uu", "Y_vv", "N_rr", "c_rev", "c_u"]

    x0 = [0.0] * num_params

    cma_opts = {
        'popsize': pop_size,
        'bounds': [-3.0, 3.0],  # -> tanh(bounds)
        'seed': seed,
        'verbose': -1,
    }

    es = cma.CMAEvolutionStrategy(x0, args_cli.sigma0, cma_opts)

    print(f"[INFO] Starting CMA-ES Optimization ({num_params} parameters, population size {pop_size})")
    start_time = time.time()

    generation = 0
    while not es.stop() and generation < args_cli.max_generations:
        generation += 1

        solutions = es.ask()

        actions_tensor = torch.tensor(np.array(solutions), dtype=torch.float32, device=unwrapped_env.device)

        action_dim = unwrapped_env.action_manager.total_action_dim
        if action_dim > num_params:
            padding = torch.zeros((pop_size, action_dim - num_params), device=unwrapped_env.device)
            actions_tensor = torch.cat([actions_tensor, padding], dim=-1)

        obs, _ = env.reset()
        cumulative_rewards = torch.zeros(pop_size, device=unwrapped_env.device)
        dones = torch.zeros(pop_size, dtype=torch.bool, device=unwrapped_env.device)

        while not dones.all():
            obs, reward, terminated, truncated, _ = env.step(actions_tensor)
            cumulative_rewards += reward * (~dones)
            dones = dones | terminated | truncated

        costs = -cumulative_rewards.detach().cpu().numpy()
        es.tell(solutions, costs)

        best_idx = np.argmin(costs)
        best_cost = costs[best_idx]
        mean_cost = np.mean(costs)
        best_params_raw = solutions[best_idx]

        print(f"Gen {generation:03d}/{args_cli.max_generations} | Best Cost: {best_cost:.4f} | Mean Cost: {mean_cost:.4f}")

        log_dict = {
            "cma/best_cost": best_cost,
            "cma/mean_cost": mean_cost,
            "cma/sigma": es.sigma,
        }

        action_term = next(iter(unwrapped_env.action_manager._terms.values()))
        raw_tensor = torch.tensor(best_params_raw, device=unwrapped_env.device)
        norm_actions = torch.tanh(raw_tensor)
        phys_params = action_term.param_min + 0.5 * (norm_actions + 1.0) * (action_term.param_max - action_term.param_min)

        for i, name in enumerate(active_param_names):
            log_dict[f"cma_best_raw/{name}"] = best_params_raw[i]
            log_dict[f"cma_best_physical/{name}"] = phys_params[i].item()

        wandb.log(log_dict)

    print(f"\n[INFO] Optimization finished in {round(time.time() - start_time, 2)} seconds.")
    print(f"[RESULT] Best Fitness (Cost): {es.result.fbest}")

    best_raw_action = torch.tensor(es.result.xbest, dtype=torch.float32, device=unwrapped_env.device).unsqueeze(0)
    padded_best = torch.zeros((1, action_dim), device=unwrapped_env.device)
    padded_best[0, :num_params] = best_raw_action

    env.reset()
    env.step(padded_best.repeat(env_cfg.scene.num_envs, 1))

    print("[RESULT] Best Physical Parameters:")
    for name in active_param_names:
        val = unwrapped_env.sysid_params[name][0].item()
        print(f"  - {name}: {val:.6f}")

    wandb.finish()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
