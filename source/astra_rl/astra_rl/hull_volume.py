import torch
from pxr import UsdGeom
from isaaclab.utils.math import transform_points

from .water import Water


class HullVolume:
    """
    Class for calculating the submerged volume and centroid of a hull
    """
    def __init__(self, robot, stage, HULLS_CFG, device):
        self.device = device
        self.H = len(HULLS_CFG)
        env_namespace = robot.root_physx_view.prim_paths[0].rsplit("/Robot", 1)[0]

        # zero tensor for clamp (Tensor input, Tensor min = None, Tensor max = None, *, Tensor out = None)
        self.zero_tensor = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        # total rest volume
        self.total_rest_vol = sum(hull["REST_VOLUME"] for hull in HULLS_CFG)

        # expand hull paths with the actual environment namespace
        self.hull_paths = []
        hull_pts_raw = []
        for hull_cfg in HULLS_CFG:
            p = hull_cfg["PATH"].replace("{ENV_REGEX_NS}", env_namespace)
            # error checking for invalid prim paths
            prim = stage.GetPrimAtPath(p)
            if not prim or not prim.IsValid():
                raise ValueError(f"[HullVolume] Prim path '{p}' not found.")

            self.hull_paths.append(p)
            hull_pts_raw.append(UsdGeom.Mesh(prim).GetPointsAttr().Get())  # (H)

        # get the max number of hull pts
        self.M_max = max([len(pts) for pts in hull_pts_raw])

        # pre-allocate padded M_max dim tensors
        self.all_hull_pts_b = torch.zeros((self.H, self.M_max, 3), device=self.device)
        self.all_hull_draft_max = torch.zeros((1, self.H, self.M_max), device=self.device)
        self.all_hull_draft_rest = torch.zeros((1, self.H, self.M_max), device=self.device)

        # pre-allocate self.H dim tensors
        self.neg_scale = torch.zeros((1, self.H), device=self.device)
        self.pos_scale = torch.zeros((1, self.H), device=self.device)
        self.vol_rest = torch.zeros((1, self.H), device=self.device)
        self.vol_max = torch.zeros((1, self.H), device=self.device)
        self.centroid_offset = torch.zeros((1, self.H, 3), device=self.device)

        for hull_idx in range(self.H):
            # get the pts of each hull
            M_h = len(hull_pts_raw[hull_idx])

            # OpenUSD point list to a tensor
            offset_tensor = torch.tensor(HULLS_CFG[hull_idx]["XFORM_OFFSET"], device=self.device, dtype=torch.float32)
            hull_pts_b = torch.tensor(list(hull_pts_raw[hull_idx]), device=self.device, dtype=torch.float32) + offset_tensor

            # max and rest draft per point in the body frame
            hull_draft_max = hull_pts_b[:, 2].max() - hull_pts_b[:, 2]
            hull_draft_rest = torch.clamp(HULLS_CFG[hull_idx]["REST_WATERLINE_Z"] - hull_pts_b[:, 2],
                                          min=self.zero_tensor,
                                          max=hull_draft_max)

            # add hull pts and draft to padded tensors
            self.all_hull_pts_b[hull_idx, :M_h, :] = hull_pts_b
            self.all_hull_draft_max[0, hull_idx, :M_h] = hull_draft_max
            self.all_hull_draft_rest[0, hull_idx, :M_h] = hull_draft_rest

            # save neg_scale and pos_scale from the sum of hull_draft_rest/above hull_draft_rest
            rest_draft_sum = hull_draft_rest.sum()
            above_rest_sum = (hull_draft_max - hull_draft_rest).sum()
            self.neg_scale[0, hull_idx] = HULLS_CFG[hull_idx]["REST_VOLUME"] / rest_draft_sum
            self.pos_scale[0, hull_idx] = (HULLS_CFG[hull_idx]["FULL_VOLUME"] - HULLS_CFG[hull_idx]["REST_VOLUME"]) / above_rest_sum

            # save the rest and full volume of each hull
            self.vol_rest[0, hull_idx] = HULLS_CFG[hull_idx]["REST_VOLUME"]
            self.vol_max[0, hull_idx] = HULLS_CFG[hull_idx]["FULL_VOLUME"]

            # centroid rest from hull pts in sim
            weighted_sum_rest = torch.sum(hull_pts_b * hull_draft_rest.unsqueeze(-1), dim=0)
            weighted_sum_rest[2] += 0.5 * torch.sum(hull_draft_rest * hull_draft_rest)
            centroid_rest_sim = weighted_sum_rest / (torch.sum(hull_draft_rest) + 1e-16)

            # get offset between sim and centroid rest from CAD measurements
            centroid_rest_cad = torch.tensor(HULLS_CFG[hull_idx]["REST_CENTROID"], device=self.device, dtype=centroid_rest_sim.dtype)
            self.centroid_offset[0, hull_idx] = centroid_rest_cad - centroid_rest_sim

        # sum draft rest and pre-flatten hull pts
        self.all_hull_draft_rest_sum = self.all_hull_draft_rest.sum(dim=2)  # (1, H)
        self.flat_hull_pts_b = self.all_hull_pts_b.view(-1, 3)


    def submerged_vol_centroid_b(self, robot_data):
        N = robot_data.root_pos_w.shape[0]
        H = self.H
        M_max = self.M_max
        zero_tensor = self.zero_tensor

        # transform body frame pts to world frame pts
        # https://isaac-sim.github.io/IsaacLab/v2.3.2/source/api/lab/isaaclab.utils.html#isaaclab.utils.math.transform_points
        hull_pts_w = transform_points(points=self.flat_hull_pts_b, pos=robot_data.root_pos_w, quat=robot_data.root_quat_w)
        hull_pts_w = hull_pts_w.view(N, H, M_max, 3)

        # get draft per point (N, H, M) from current wave height
        if Water.is_waves_enabled:
            wave_z = Water.get_wave_z(hull_pts_w[..., 0], hull_pts_w[..., 1], self.device)
            draft = wave_z.view(N, H, M_max).sub_(hull_pts_w[..., 2])
        else:
            draft = torch.neg(hull_pts_w[..., 2])

        # clamp and sum draft
        draft.clamp_(min=zero_tensor, max=self.all_hull_draft_max)
        draft_sum = torch.sum(draft, dim=2)  # (N, H)

        # compute volume from summed draft displacement (N, H)
        delta_sum = draft_sum - self.all_hull_draft_rest_sum
        vol_scale = torch.where(delta_sum < 0, self.neg_scale, self.pos_scale)
        vol = delta_sum.mul_(vol_scale).add_(self.vol_rest).clamp_(min=zero_tensor, max=self.vol_max)

        # centroid of currently submerged volume in the body frame (N, H, 3)
        weighted_sum = torch.matmul(draft.unsqueeze(-2), self.all_hull_pts_b).squeeze(-2)
        weighted_sum[..., 2].add_(draft.square_().sum(dim=2).mul_(0.5))
        denom = draft_sum.unsqueeze(-1).add_(1e-16)
        centroid = weighted_sum.div_(denom).add_(self.centroid_offset)

        return vol, centroid  # [vol: (N, H), centroid: (N, H, 3)]


    def get_total_rest_vol(self):
        return self.total_rest_vol  # [M^3]
