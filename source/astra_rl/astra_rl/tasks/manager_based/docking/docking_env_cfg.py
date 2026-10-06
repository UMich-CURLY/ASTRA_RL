# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ImuCfg, ContactSensorCfg, MultiMeshRayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg, GaussianNoiseCfg

from .... import mdp

##
# Pre-defined configs
##

from .robot_cfg import RIGID_OBJECT_CFG  # isort:skip
from ....mdp.thruster_action import ThrusterActionCfg

##
# Scene definition
##


@configclass
class DockingSceneCfg(InteractiveSceneCfg):
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.scene.html#isaaclab.scene.InteractiveSceneCfg
    """Configuration for the scene."""

    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectCfg
    robot: RigidObjectCfg = RIGID_OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/BlueBoat/imu/Imu_Sensor")

    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.MultiMeshRayCasterCfg
    ray_caster = MultiMeshRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BlueBoat/lidar",
        update_period=(1/10),  # 10 Hz
        history_length=1,
        offset=MultiMeshRayCasterCfg.OffsetCfg(pos=(-0.00807, 0.0, 0.33472)),  # /Root/BlueBoat/lidar
        max_distance=120,
        mesh_prim_paths=["{ENV_REGEX_NS}/Dock"],
        # Hesai Pandar XT32
        # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sensors.patterns.html#isaaclab.sensors.patterns.LidarPatternCfg
        pattern_cfg=patterns.LidarPatternCfg(
            # XT32 vertical: 32 beams from -16° to +15°
            # channels=32,
            # vertical_fov_range=(-16, 15),
            channels=1,
            vertical_fov_range=(-0, 0),
            # XT32 horizontal: 2000 beams from -180° to +180°
            horizontal_fov_range=(-180, 180),
            horizontal_res=15,
        )
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/BlueBoat",
        update_period=0.0,
        history_length=6,
        debug_vis=False,
    )

    # Dock dim: Inside width: 1.5m, Total width: 2.5m. Inside length: 1.5, Total length: 2
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.AssetBaseCfg
    dock = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Dock",
        # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.sim.spawners.html#isaaclab.sim.spawners.from_files.UsdFileCfg
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(Path(__file__).parent / "Dock.usd"),
        ),
        # InitialStateCfg: https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectCfg.InitialStateCfg
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.0, 0.0, 0.0))
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(1.0, 1.0, 1.0),
                                     enable_color_temperature=True,
                                     color_temperature=6150,
                                     exposure=9.0,
                                     intensity=1.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.actions

    # ActionsCfg is a manager for processing and applying actions for a given world
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#action-manager

    thrusters = ThrusterActionCfg()


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.observations

    # ObservationsCfg is a manager for observation groups containing observation terms:
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#observation-manager

    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationGroupCfg
    @configclass
    class PolicyCfg(ObsGroup):
        # Each of these are a ObservationTermCfg:
        # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#isaaclab.managers.ObservationTermCfg

        # general observations
        robot_vel_b_xy = ObsTerm(func=mdp.robot_vel_b_xy,
                                 noise=GaussianNoiseCfg(mean=0.0, std=0.05))

        imu_heading_vel = ObsTerm(func=mdp.imu_heading_vel,
                                  params={"asset_cfg": SceneEntityCfg(name="imu")},
                                  noise=GaussianNoiseCfg(mean=0.0, std=1*math.pi/180))

        imu_acc_b_xy = ObsTerm(func=mdp.imu_acc_b_xy,
                               params={"asset_cfg": SceneEntityCfg(name="imu")},
                               noise=GaussianNoiseCfg(mean=0.0, std=0.2))

        # previous actions
        last_processed_action = ObsTerm(func=mdp.last_processed_actions,
                                        params={"history_length": 5, "stride": 2})
        # Required: (history_length - 1) * (stride) + 1 <= int(max_latency*60)

        # task specific observations
        lidar_ray_hits_depth = ObsTerm(func=mdp.lidar_ray_hits_depth,
                                       params={"asset_cfg": SceneEntityCfg(name="ray_caster")},
                                       noise=GaussianNoiseCfg(mean=0.0, std=0.01))

        target_pos_error_b_xy = ObsTerm(func=mdp.target_pos_error_b_xy,
                                        params={"target": (0, 0),
                                                "asset_cfg": SceneEntityCfg(name="imu")},
                                        noise=GaussianNoiseCfg(mean=0.0, std=0.025))

        target_heading_error_sin_cos = ObsTerm(func=mdp.target_heading_error_sin_cos,
                                               params={"target": 0,
                                                       "asset_cfg": SceneEntityCfg(name="imu"),
                                                       "noise_gaussian_std": 0.8 * math.pi/180})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True  # single tensor required by skrl/train.py


    @configclass
    class CriticCfg(PolicyCfg):  # inherit PolicyCfg's ObservationTermCfgs

        # total hydro wind observations (x, y, yaw)
        total_hydro_wind_force_b_2d = ObsTerm(func=mdp.total_hydro_wind_force_b_2d)

        # wind observations (x, y, yaw)
        wind_force_b_2d = ObsTerm(func=mdp.wind_force_b_2d)

        wind_opp_direction_b_sin_cos = ObsTerm(func=mdp.wind_opp_direction_b_sin_cos,
                                               params={"asset_cfg": SceneEntityCfg(name="imu")})

        def __post_init__(self) -> None:
            self.enable_corruption = False  # no noise
            # get rid of noise added without using ObsTerm.noise
            if hasattr(self, "target_heading_error_sin_cos"):
                self.target_heading_error_sin_cos.params["noise_gaussian_std"] = 0.0

            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.events

    # Events alter the simulation state. Ex: Changing physics, applying forces, resetting state:
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.events

    # reset robot
    reset_robot = EventTerm(
        func=mdp.reset_robot_on_water,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x":     (-3.5, -2.5),
                "y":     (-1.0, 1.0),
                "z":     (0.0, 0.0),
                "roll":  (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw":   (-math.pi/8, math.pi/8),
            },
            "velocity_range": {
                "x":     (0.0, 0.0),
                "y":     (0.0, 0.0),
                "z":     (0.0, 0.0),
                "roll":  (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw":   (0.0, 0.0),
            },
        },
    )

    set_wind_params = EventTerm(
        func=mdp.set_wind_params,
        mode="reset",
        params={
            "wind_speed_range": (0.0, 0.0),
            "wind_dir_range":   (0.0, 0.0),
            "wind_dir_range2":  None
        }
    )

    # apply wind
    wind = EventTerm(
        func=mdp.apply_wind_force,
        mode="interval",
        interval_range_s=(0.0167, 0.0167),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "wind_speed_std":  0.0,
            "wind_dir_std":    0.0 *math.pi/180,
            "OU_theta":        0.0025,
        }
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.rewards

    # (1) Primary tasks: reach the goal
    reached_goal_flag = RewTerm(func=mdp.reached_goal_flag, weight=50.0)
    
    # (2) Primary tasks: avoid being too close to objects

    # (3) Time elapsed penalty
    # alive = RewTerm(func=mdp.is_alive, weight=-0.1)

    # (4) Failure penalty
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-50)

    # (5) Secondary tasks: go toward dock
    target_pos_xy_euclidean = RewTerm(func=mdp.target_pos_xy_euclidean,
                                      weight=-1.0,
                                      params={"target": (0, 0)})

    # (6) Secondary tasks: go toward target heading when docking
    # target_heading_RBF = RewTerm(func=mdp.target_heading_RBF,
    #                              weight=1.0,
    #                              params={"target": 0, "sigma": 0.5})

    # (7) Shaping tasks: low ASV acceleration
    # target_vel_xy_l2 = RewTerm(func=mdp.target_acc_xy_l2,
    #                            weight=-0.01,
    #                            params={"target": (0, 0, 0)})

    # (8) Shaping tasks: lower thruster magnitude, first, and second derivative
    # thruster_l2       = RewTerm(func=mdp.thruster_l2,       weight=-0.0)
    thruster_rate_l2  = RewTerm(func=mdp.thruster_rate_l2,  weight=-1.0)
    thruster_rate2_l2 = RewTerm(func=mdp.thruster_rate2_l2, weight=-0.5)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.terminations

    # (1) Time out
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.terminations.time_out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Hit object
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#isaaclab.envs.mdp.terminations.illegal_contact
    hit_object = DoneTerm(func=mdp.illegal_contact,
                          params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg(name="contact_forces")})
    
    # (3) Reached goal and stopped
    reached_goal_termination = DoneTerm(func=mdp.reached_goal_termination,
                                        params={"pos_target": (0, 0), "heading_target": 0,
                                                "pos_epsilon": 0.50, "heading_epsilon": 30 *math.pi/180,
                                                "max_pos_vel": 0.10, "max_ang_vel": 5 *math.pi/180},
                                        time_out=True) # avoid termination penalty


@configclass
class CurriculumManager:
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.managers.html#curriculum-manager

    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.mdp.html#module-isaaclab.envs.mdp.curriculums

    # waypoint2_obs_schedule = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "observations.policy.target_pos_error_b_xy.params.target",
    #         "modify_fn": mdp.override_value,
    #         "modify_params": {"value": (-1, 0),
    #                           "num_steps": 1_000}
    #     }
    # )

    # waypoint2_rew_schedule = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "rewards.target_pos_xy_RBF.params.target",
    #         "modify_fn": mdp.override_value,
    #         "modify_params": {"value": (-1, 0),
    #                           "num_steps": 1_000}
    #     }
    # )

    # wind_dir_schedule_1 = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.set_wind_params.params.wind_dir_range",
    #         "modify_fn": mdp.override_value_interp,
    #         "modify_params": {"term": mdp.get_common_step_counter,
    #                           "start_term": 6_000,
    #                           "end_term": 32_000,
    #                           "start_value": (0.0, 0.0),
    #                           "end_value": (-math.pi/2, math.pi/2)}
    #     }
    # )

    # wind_dir_schedule_2 = CurrTerm(
    #     func=mdp.modify_term_cfg,
    #     params={
    #         "address": "events.set_wind_params.params.wind_dir_range2",
    #         "modify_fn": mdp.override_value_interp,
    #         "modify_params": {"term": mdp.get_common_step_counter,
    #                           "start_term": 6_000,
    #                           "end_term": 32_000,
    #                           "start_value": (math.pi, math.pi),
    #                           "end_value": (math.pi/2, 3*math.pi/2)}
    #     }
    # )

    enable_waves_schedule = CurrTerm(
        func=mdp.enable_waves_steps,
        params={
            "num_steps": 64_000,
            "wave_scale": 1.0,
            "directional_spreading": 0.75,
            "wind_direction_rad": 0.0,
            "wind_speed_ms": 5.0,
            "fetch_km": 100.0,
            "gamma": 3.3,
            "water_depth_m": 50.0
        }
    )


##
# Environment configuration
##


@configclass
class DockingEnvCfg(ManagerBasedRLEnvCfg):
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.envs.html#isaaclab.envs.ManagerBasedRLEnvCfg
    # Scene settings
    scene: DockingSceneCfg = DockingSceneCfg(num_envs=256,
                                             env_spacing=0.0,
                                             replicate_physics=False)
    # Setting replicate_physics=True results in this error:
    # [omni.physx.plugin] Replication of this type is not supported: 2228224, prim path: /World/envs/env_0/Robot/BlueBoat
    # [omni.physx.plugin] Replication of this type is not supported: 2228224, prim path: /World/envs/env_0/Robot/BlueBoat/stbd_prop
    # [omni.physx.plugin] Replication of this type is not supported: 2228224, prim path: /World/envs/env_0/Robot/BlueBoat/port_prop

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumManager = CurriculumManager()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 10
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
