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
from ....mdp.sys_id_action import SysIdActionCfg
from ....mdp.commands import OfflineCommandCfg

##
# Scene definition
##


@configclass
class SystemIdSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""
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
    sys_id_action = SysIdActionCfg()


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_pos_w_xy = ObsTerm(func=mdp.robot_pos_w_xy)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    # reset robot
    reset_robot = EventTerm(
        func=mdp.reset_robot_sys_id,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot")
        },
    )


@configclass
class RewardsCfg:
    # (1) Primary tasks: track recorded (x, y)
    recorded_pos_xy_RBF = RewTerm(func=mdp.recorded_pos_xy_RBF,
                                  weight=1.5,
                                  params={"sigma": 1.0})

    # (2) Primary tasks: track recorded (heading)
    recorded_heading_RBF = RewTerm(func=mdp.recorded_heading_RBF,
                                   weight=0.5,
                                   params={"sigma": 0.75, "asset_cfg": SceneEntityCfg(name="imu")})

    # (3) Secondary tasks: track recorded (vel_x, vel_y)
    recorded_vel_b_xy_RBF = RewTerm(func=mdp.recorded_vel_b_xy_RBF,
                                    weight=1.0,
                                    params={"sigma": 1.0})

    # (4) Secondary tasks: track recorded (heading_rate)
    recorded_heading_rate_RBF = RewTerm(func=mdp.recorded_heading_rate_RBF,
                                        weight=0.5,
                                        params={"sigma": 0.75, "asset_cfg": SceneEntityCfg(name="imu")})


@configclass
class TerminationsCfg:
    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CommandsCfg:
    trajectory = OfflineCommandCfg(command_dim=6)
    thrust = OfflineCommandCfg(command_dim=2)


##
# Environment configuration
##


@configclass
class SystemIdEnvCfg(ManagerBasedRLEnvCfg):
    scene: SystemIdSceneCfg = SystemIdSceneCfg(num_envs=256,
                                               env_spacing=0.0,
                                               replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 20  # [SYSID]
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
