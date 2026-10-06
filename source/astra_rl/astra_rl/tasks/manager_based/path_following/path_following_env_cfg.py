# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

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
from isaaclab.sensors import ImuCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg, GaussianNoiseCfg

from .... import mdp

##
# Pre-defined configs
##

from .robot_cfg import RIGID_OBJECT_CFG  # isort:skip
from ....mdp.thruster_action import ThrusterActionCfg
from ....mdp.commands import PathCommandCfg

##
# Scene definition
##


@configclass
class PathFollowingSceneCfg(InteractiveSceneCfg):
    # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.assets.html#isaaclab.assets.RigidObjectCfg
    robot: RigidObjectCfg = RIGID_OBJECT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/BlueBoat/imu/Imu_Sensor")

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
    thrusters = ThrusterActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # general observations
        robot_vel_b_xy = ObsTerm(func=mdp.robot_vel_b_xy,  # 2
                                 noise=GaussianNoiseCfg(mean=0.0, std=0.01))

        imu_heading_vel = ObsTerm(func=mdp.imu_heading_vel,  # 1
                                  params={"asset_cfg": SceneEntityCfg(name="imu")},
                                  noise=GaussianNoiseCfg(mean=0.0, std=1.0 * math.pi/180))

        imu_acc_b_xy = ObsTerm(func=mdp.imu_acc_b_xy,  # 2
                               params={"asset_cfg": SceneEntityCfg(name="imu")},
                               noise=GaussianNoiseCfg(mean=0.0, std=0.02))

        # previous actions
        last_processed_action = ObsTerm(func=mdp.last_processed_actions,  # 10
                                        params={"history_length": 5, "stride": 2})
        # Required: (history_length - 1) * (stride) + 1 <= int(max_latency*60)

        # task specific observations
        local_path_waypoints_dxdy = ObsTerm(func=mdp.local_path_waypoints_dxdy,  # 6
                                            params={"num_lookahead": 3,
                                                    "asset_cfg": SceneEntityCfg(name="imu")},
                                            noise=GaussianNoiseCfg(mean=0.0, std=0.025))

        cross_track_error = ObsTerm(func=mdp.cross_track_error,  # 1
                                    params={"asset_cfg": SceneEntityCfg(name="imu")},
                                    noise=GaussianNoiseCfg(mean=0.0, std=0.025))

        path_heading_error_sin_cos = ObsTerm(func=mdp.path_heading_error_sin_cos,  # 2
                                             params={"asset_cfg": SceneEntityCfg(name="imu"),
                                                     "noise_gaussian_std": 1.0 * math.pi/180})

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True


    @configclass
    class CriticCfg(PolicyCfg):
        # total hydro wind observations (x, y, yaw)
        total_hydro_wind_force_b_2d = ObsTerm(func=mdp.total_hydro_wind_force_b_2d)
        # wind observations (x, y, yaw)
        wind_force_b_2d = ObsTerm(func=mdp.wind_force_b_2d)
        # wind direction (sin, cos)
        wind_opp_direction_b_sin_cos = ObsTerm(func=mdp.wind_opp_direction_b_sin_cos,
                                               params={"asset_cfg": SceneEntityCfg(name="imu")})

        def __post_init__(self) -> None:
            self.enable_corruption = False
            # get rid of noise added without using ObsTerm.noise
            if hasattr(self, "path_heading_error_sin_cos"):
                self.path_heading_error_sin_cos.params["noise_gaussian_std"] = 0.0
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    # reset robot
    reset_robot = EventTerm(
        func=mdp.reset_robot_on_water,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x":     (0.0, 0.0),
                "y":     (0.0, 0.0),
                "z":     (0.0, 0.0),
                "roll":  (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw":   (-math.pi, math.pi),
            },
            "velocity_range": {
                "x":     (-0.2, 0.2),
                "y":     (-0.1, 0.1),
                "z":     (0.0, 0.0),
                "roll":  (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw":   (-3.0 *math.pi/180, 3.0 *math.pi/180),
            },
        },
    )
    # wind
    set_wind_params = EventTerm(
        func=mdp.set_wind_params,
        mode="reset",
        params={
            "wind_speed_range": (0.0, 0.0),
            "wind_dir_range":   (-math.pi, math.pi),
        }
    )
    # apply wind
    wind = EventTerm(
        func=mdp.apply_wind_force,
        mode="interval",
        interval_range_s=(0.0167, 0.0167),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "wind_speed_std":  0.03,
            "wind_dir_std":    0.5 *math.pi/180,
            "OU_theta":        0.0025,
        }
    )
    # hydrodynamics DR
    randomize_hydro_params = EventTerm(
        func=mdp.randomize_hydrodynamics,
        mode="reset",
        params={
            "param_ranges": {
                "Iz":   (6.306603*(1-(2.6378867045903984*0.10)), 6.306603*((2.6378867045903984*0.10)+1)),
                "X_u":  (0.593417*(1-(2.999536290786401*0.10)), 0.593417*((2.999536290786401*0.10)+1)),
                "Y_v":  (71.255440*(1-(2.887917379255792*0.10)), 71.255440*((2.887917379255792*0.10)+1)),
                "N_r":  (4.478570*(1-(2.99989960134562*0.10)), 4.478570*((2.99989960134562*0.10)+1)),  # abs (-2.99989960134562 -> 2.99989960134562)
                "X_uu": (6.975154*(1-(0.07146042617817246*0.10)), 6.975154*((0.07146042617817246*0.10)+1)),
                "Y_vv": (148.370300*(1-(2.694839784108198*0.10)), 148.370300*((2.694839784108198*0.10)+1)),
                "N_rr": (13.159036*(1-(2.999973067907203*0.10)), 13.159036*((2.999973067907203*0.10)+1)),  # abs (-2.999973067907203 -> 2.999973067907203)
            }
        }
    )


@configclass
class RewardsCfg:
    # (1) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-10)

    # (2) Primary tasks: minimize cross-track error
    cross_track_error_RBF = RewTerm(func=mdp.cross_track_error_RBF,
                                    weight=2.0,
                                    params={"sigma": 0.3})

    # (3) Primary tasks: minimize heading error
    path_heading_error_RBF = RewTerm(func=mdp.path_heading_error_RBF,
                                     weight=1.0,
                                     params={"sigma": 0.3})

    # (4) Primary tasks: progress along the path at X m/s
    path_progress_RBF = RewTerm(func=mdp.path_progress_RBF,
                                weight=1.5,
                                params={"speed": 0.5, "sigma": 0.25})

    # (5) Shaping tasks: lower thruster magnitude, first, and second derivative
    thruster_l2       = RewTerm(func=mdp.thruster_l2,       weight=-0.0)
    thruster_rate_l2  = RewTerm(func=mdp.thruster_rate_l2,  weight=-1.5)
    thruster_rate2_l2 = RewTerm(func=mdp.thruster_rate2_l2, weight=-0.5)


@configclass
class TerminationsCfg:
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Too far from the path
    cross_track_limit = DoneTerm(
        func=mdp.cross_track_limit,
        params={"distance": 2.0}
    )


@configclass
class CurriculumManager:
    wind_speed_schedule = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.set_wind_params.params.wind_speed_range",
            "modify_fn": mdp.override_value_interp,
            "modify_params": {"term": mdp.get_common_step_counter,
                              "start_term": 8_000,
                              "end_term": 16_000,
                              "start_value": (0.0, 0.0),
                              "end_value": (0.0, 5.0)}
        }
    )

    enable_waves_schedule = CurrTerm(
        func=mdp.enable_waves_steps,
        params={
            "num_steps": 64_000,
            "wave_scale": 1.0,
            "directional_spreading": 0.75,
            "wind_direction_rad": 0.0,
            "wind_speed_ms": 4.0,
            "fetch_km": 100.0,
            "gamma": 3.3,
            "water_depth_m": 50.0
        }
    )


@configclass
class CommandsCfg:
    path = PathCommandCfg()


##
# Environment configuration
##


@configclass
class PathFollowingEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PathFollowingSceneCfg = PathFollowingSceneCfg(num_envs=256,
                                                         env_spacing=1.0,
                                                         replicate_physics=False)

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumManager = CurriculumManager()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 20
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
