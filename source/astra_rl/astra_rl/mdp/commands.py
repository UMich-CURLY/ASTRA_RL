from collections.abc import Sequence
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

# ---------------------------------------------
# Offline Command Term
# ---------------------------------------------
class OfflineCommand(CommandTerm):
    def __init__(self, cfg: "OfflineCommandCfg", env):
        super().__init__(cfg, env)
        self.command_buf = torch.zeros((self.num_envs, cfg.command_dim), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.command_buf

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass


@configclass
class OfflineCommandCfg(CommandTermCfg):
    class_type: type = OfflineCommand
    command_dim: int = 2
    resampling_time_range: tuple[float, float] = (0.0, 0.0)


@configclass
class CommandsCfg:
    """Command terms for the environment."""
    trajectory = OfflineCommandCfg(command_dim=6)
    thrust = OfflineCommandCfg(command_dim=2)


# path commands


class PathCommand(CommandTerm):
    def __init__(self, cfg: "PathCommandCfg", env):
        super().__init__(cfg, env)

        self.robot = env.scene[self.cfg.asset_cfg.name]
        self.command_buf = torch.zeros(
            (self.num_envs, self.cfg.num_waypoints, 3), device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return self.command_buf

    def _resample_command(self, env_ids: torch.Tensor):
        num_resample = len(env_ids)

        root_pos = self.robot.data.root_pos_w[env_ids]
        root_quat = self.robot.data.root_quat_w[env_ids]
        _, _, current_yaw = math_utils.euler_xyz_from_quat(root_quat)

        waypoints = torch.zeros(
            (num_resample, self.cfg.num_waypoints, 3), device=self.device
        )
        waypoints[:, :, 2] = root_pos[:, 2].unsqueeze(1)

        x = root_pos[:, 0]
        y = root_pos[:, 1]
        heading = current_yaw

        curvature = torch.zeros(num_resample, device=self.device)

        for i in range(self.cfg.num_waypoints):
            waypoints[:, i, 0] = x
            waypoints[:, i, 1] = y

            x = x + self.cfg.waypoint_step_size * torch.cos(heading)
            y = y + self.cfg.waypoint_step_size * torch.sin(heading)

            noise = (
                torch.rand(num_resample, device=self.device) * 2 - 1.0
            ) * self.cfg.max_curvature

            curvature = (
                self.cfg.smoothing_factor * curvature
                + (1 - self.cfg.smoothing_factor) * noise
            )

            heading = heading + curvature

        self.command_buf[env_ids] = waypoints

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if hasattr(self, "_waypoint_markers") and self._waypoint_markers is not None:
            self._waypoint_markers.set_visibility(debug_vis)
            return

        if debug_vis:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Command/waypoints",
                markers={
                    "waypoint": sim_utils.SphereCfg(
                        radius=0.1,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                    )
                },
            )
            self._waypoint_markers = VisualizationMarkers(marker_cfg)
            self._waypoint_markers.set_visibility(True)

    def _debug_vis_callback(self, event):
        if hasattr(self, "_waypoint_markers") and self._waypoint_markers is not None:
            positions = self.command_buf.reshape(-1, 3)
            self._waypoint_markers.visualize(positions)


@configclass
class PathCommandCfg(CommandTermCfg):
    class_type: type = PathCommand

    resampling_time_range: tuple[float, float] = (20.0, 20.0)
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    num_waypoints: int = 21
    waypoint_step_size: float = 0.5  # meters between waypoints
    max_curvature: float = 0.30  # max curve per step (radians)
    smoothing_factor: float = 0.60  # curve smoothing (0.0, 1.0)
