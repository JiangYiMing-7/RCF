"""
radar_pointnet_backbone.py

支持 4D 毫米波雷达格式:
  [range, azimuth, elevation, doppler, power, x, y, z]

主要组件:
 - load_point_cloud / load_image / load_label / load_calibration / project_lidar_to_image
 - KITTIDataset (适配上述格式，返回 raw_points 与 编码 features + xyz)
 - PointNet2Backbone (Set Abstraction + Feature Propagation)
 - RadarPointNetBackbone: 对外封装，直接接收 (B,N,8) 雷达数据并产生点特征
"""

import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader import KITTIDatasetRadar4D

# -------------------------
# PointNet++ Utilities
# -------------------------
def square_distance(src, dst):
    """Batched squared distance: src (B,N,3), dst (B,M,3) -> (B,N,M)"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    src_sq = (src ** 2).sum(dim=-1, keepdim=True)  # (B,N,1)
    dst_sq = (dst ** 2).sum(dim=-1).unsqueeze(1)   # (B,1,M)
    inner = torch.matmul(src, dst.permute(0,2,1))  # (B,N,M)
    return src_sq - 2 * inner + dst_sq


def farthest_point_sample(xyz, npoint):
    """
    Farthest point sampling (iterative)
    xyz: (B,N,3) tensor
    return: (B, npoint) long tensor indices
    """
    device = xyz.device
    B, N, _ = xyz.shape
    if npoint is None or npoint <= 0:
        return torch.zeros(B, 0, dtype=torch.long, device=device)
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_idx = torch.arange(B, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_idx, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        farthest = farthest.clamp(0, N - 1)
    return centroids


def index_points(points, idx):
    """
    points: (B,N,C)
    idx: (B,S) or (B,S,nsample)
    returns: (B,S,C) or (B,S,nsample,C)
    """
    device = points.device
    B = points.shape[0]
    if idx.dim() == 2:
        S = idx.shape[1]
        idx_expand = idx.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        return torch.gather(points, 1, idx_expand)
    elif idx.dim() == 3:
        B, S, nsample = idx.shape
        idx_flat = idx.reshape(B, -1)
        idx_expand = idx_flat.unsqueeze(-1).expand(-1, -1, points.shape[-1])
        grouped = torch.gather(points, 1, idx_expand)
        return grouped.reshape(B, S, nsample, -1)
    else:
        raise ValueError("idx dim illegal")


def ball_query(radius, nsample, xyz, new_xyz):
    """
    For each center in new_xyz find up to nsample points in xyz within radius.
    Return indices (B,S,nsample). If fewer than nsample, fill with nearest valid index.
    """
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    # squared distances (B,S,N)
    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)
    within = sqrdists <= (radius ** 2)
    # large value for masking
    masked = sqrdists.clone()
    masked[~within] = 1e10
    # argsort and take first nsample
    group_idx = masked.argsort(dim=-1)[:, :, :nsample].long()  # (B,S,nsample)
    # if empty groups (no within), then group_idx currently are nearest points (due to masked large)
    # but to be safe: ensure indices in-range
    group_idx = group_idx.clamp(0, N - 1)
    return group_idx


def three_nn_interpolate(xyz1, xyz2, points2):
    """
    Inverse distance weighted 3-NN interpolation
    xyz1: (B,N,3) target
    xyz2: (B,M,3) source
    points2: (B,M,C)
    returns: (B,N,C)
    """
    B, N, _ = xyz1.shape
    _, M, _ = xyz2.shape
    C = points2.shape[-1]
    dist = square_distance(xyz1, xyz2)  # (B,N,M)
    k = min(3, M)
    dists_k, idx_k = torch.topk(dist, k=k, dim=-1, largest=False, sorted=False)  # (B,N,k)
    eps = 1e-8
    inv = 1.0 / (dists_k + eps)
    norm = inv.sum(dim=-1, keepdim=True)
    weights = inv / norm  # (B,N,k)
    points2_grouped = index_points(points2, idx_k)  # (B,N,k,C)
    weights = weights.unsqueeze(-1)
    interpolated = (points2_grouped * weights).sum(dim=2)
    return interpolated


# -------------------------
# PointNet++ Layers
# -------------------------
class SetAbstractionLayer(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        last_channel = in_channel + 3  # local coords + features
        self.mlp_convs = nn.ModuleList()
        self.mlp_gns = nn.ModuleList()
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias=False))
            self.mlp_gns.append(nn.GroupNorm(min(32, out_channel), out_channel))
            last_channel = out_channel

    def forward(self, xyz, points=None):
        """
        xyz: (B,N,3)
        points: (B,N,C) or None
        return new_xyz (B,S,3), new_points (B,S,C_out)
        """
        B, N, _ = xyz.shape
        if self.group_all or (self.npoint is None) or self.npoint >= N:
            new_xyz = xyz.mean(dim=1, keepdim=True)
            grouped_xyz = xyz.unsqueeze(1)  # (B,1,N,3)
            grouped_points = points.unsqueeze(1) if points is not None else None
        else:
            fps_idx = farthest_point_sample(xyz, self.npoint)  # (B,S)
            new_xyz = index_points(xyz, fps_idx)               # (B,S,3)
            group_idx = ball_query(self.radius, self.nsample, xyz, new_xyz)  # (B,S,nsample)
            # safety clamp
            if torch.any(group_idx >= xyz.shape[1]) or torch.any(group_idx < 0):
                group_idx = group_idx.clamp(0, xyz.shape[1] - 1)
            grouped_xyz = index_points(xyz, group_idx)  # (B,S,nsample,3)
            grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
            grouped_points = index_points(points, group_idx) if points is not None else None
            # handle degenerate groups (if any)
            repeated_mask = (group_idx == group_idx[:,:,0:1]).all(dim=-1)
            if repeated_mask.any():
                for b in range(B):
                    for s in range(self.npoint):
                        if repeated_mask[b,s]:
                            grouped_xyz[b,s,:,:] = 0
                            if grouped_points is not None:
                                grouped_points[b,s,:,:] = points[b, fps_idx[b,s]].unsqueeze(0).repeat(self.nsample,1)

        if grouped_points is not None:
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)  # (B,S,nsample,3+C)
        else:
            new_points = grouped_xyz  # (B,S,nsample,3)

        # to (B, C_in, S, nsample)
        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        for conv, gn in zip(self.mlp_convs, self.mlp_gns):
            new_points = F.relu(gn(conv(new_points)))
        # max pool
        new_points = torch.max(new_points, dim=-1)[0]  # (B, C_out, S)
        return new_xyz, new_points.permute(0, 2, 1).contiguous()  # (B,S,C_out)


class FeaturePropagationLayer(nn.Module):
    def __init__(self, points1_channels, points2_channels, mlp):
        super().__init__()
        in_channel = points1_channels + points2_channels
        self.mlp_convs = nn.ModuleList()
        self.mlp_gns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1, bias=False))
            self.mlp_gns.append(nn.GroupNorm(min(32, out_channel), out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        xyz1: (B,N,3) fine
        xyz2: (B,M,3) coarse
        points1: (B,N,C1) fine features (can be None)
        points2: (B,M,C2) coarse features (can be None)
        returns: (B,N,C_out)
        """
        B, N, _ = xyz1.shape
        if points2 is None:
            interpolated = torch.zeros(B, N, 0, device=xyz1.device, dtype=xyz1.dtype)
        else:
            interpolated = three_nn_interpolate(xyz1, xyz2, points2)
        if points1 is None:
            points1 = torch.zeros(B, N, 0, device=xyz1.device, dtype=xyz1.dtype)
        new_points = torch.cat([points1, interpolated], dim=-1)  # (B,N,C)
        new_points = new_points.permute(0, 2, 1).contiguous()  # (B,C,N)
        for conv, gn in zip(self.mlp_convs, self.mlp_gns):
            new_points = F.relu(gn(conv(new_points)))
        return new_points.permute(0, 2, 1).contiguous()  # (B,N,C_out)


# -------------------------
# Backbone (PointNet2 style)
# -------------------------
class PointNet2Backbone(nn.Module):
    def __init__(self, input_channels=0):
        """
        input_channels: 点特征维度（不含 xyz）
        """
        super().__init__()
        # SA layers
        self.sa1 = SetAbstractionLayer(256, 0.2, 32, input_channels, [64, 64, 128])
        self.sa2 = SetAbstractionLayer(64, 0.4, 64, 128, [128, 128, 256])
        self.sa3 = SetAbstractionLayer(None, None, None, 256, [256, 512, 1024], group_all=True)
        # FP layers (channels aligned with SA outputs)
        self.fp3 = FeaturePropagationLayer(points1_channels=256, points2_channels=1024, mlp=[256, 256])
        self.fp2 = FeaturePropagationLayer(points1_channels=128, points2_channels=256, mlp=[256, 256])
        self.fp1 = FeaturePropagationLayer(points1_channels=input_channels, points2_channels=256, mlp=[128, 128])
        self.out_channels = 128

    def forward(self, xyz, points=None):
        """
        xyz: (B,N,3), points: (B,N,C) or None
        returns:
           l0_points_fp: (B,N,out_channels)
           l3_points: (B, S3, C3) (coarsest)
        """
        l1_xyz, l1_points = self.sa1(xyz, points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        l2_points_fp = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points_fp = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points_fp)
        l0_points_fp = self.fp1(xyz, l1_xyz, points, l1_points_fp)
        return l0_points_fp, l3_points


# -------------------------
# Radar wrapper: 编码 + Backbone
# -------------------------
class RadarPointNetBackbone(nn.Module):
    """
    接受原始雷达输入 (B,N,8) 并进行特征编码后调用 PointNet2Backbone.

    特征编码 (默认):
      输入 8 维: [r, az, el, doppler, power, x, y, z]
      编码输出特征 dim:
         [r, sin(az), cos(az), sin(el), cos(el), doppler, power] -> 7 dim

    可根据需要修改 encode_features()。
    """
    def __init__(self, encoded_feat_dim=7):
        super().__init__()
        # encoded_feat_dim 由 encode_features 决定，默认 7
        self.encoded_feat_dim = encoded_feat_dim
        # backbone expects input_channels = encoded_feat_dim
        self.backbone = PointNet2Backbone(input_channels=encoded_feat_dim)

    @staticmethod
    def encode_features(radar_points):
        """
        radar_points: (B,N,8) tensor: [r, az, el, doppler, power, x, y, z]
        returns:
          xyz: (B,N,3)
          feats: (B,N,7)
        """
        # ensure float
        pts = radar_points.float()
        r = pts[..., 0:1]            # (B,N,1)
        az = pts[..., 1]             # (B,N)
        el = pts[..., 2]             # (B,N)
        doppler = pts[..., 3:4]      # (B,N,1)
        power = pts[..., 4:5]        # (B,N,1)
        xyz = pts[..., 5:8]          # (B,N,3)

        # angle periodic encoding
        sin_az = torch.sin(az).unsqueeze(-1)
        cos_az = torch.cos(az).unsqueeze(-1)
        sin_el = torch.sin(el).unsqueeze(-1)
        cos_el = torch.cos(el).unsqueeze(-1)

        feats = torch.cat([r, sin_az, cos_az, sin_el, cos_el, doppler, power], dim=-1)
        return xyz, feats

    def forward(self, radar_points):
        """
        radar_points: (B,N,8) torch tensor
        returns:
          per-point features: (B,N,out_channels)
          global_coarse: (B,S3,C3)
        """
        assert radar_points.dim() == 3 and radar_points.shape[-1] == 8
        xyz, feats = self.encode_features(radar_points)
        l0_points_fp, l3_points = self.backbone(xyz, feats)
        return l0_points_fp, l3_points


# -------------------------
# Sanity check if run as script
# -------------------------
if __name__ == '__main__':
    # quick forward sanity test with random radar data
    B = 2
    N = 2048
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # create random radar points: [r, az, el, doppler, power, x, y, z]
    # angles in radians
    r = torch.rand(B, N, 1) * 100.0
    az = (torch.rand(B, N) - 0.5) * 2 * np.pi
    el = (torch.rand(B, N) - 0.5) * 0.5  # small elevation
    doppler = (torch.rand(B, N, 1) - 0.5) * 10.0
    power = (torch.rand(B, N, 1) * 30.0) - 90.0  # dB-like negative values
    # compute xyz from spherical (approx): x = r * cos(el) * cos(az) ...
    az_t = az.unsqueeze(-1)
    el_t = el.unsqueeze(-1)
    x = (r * torch.cos(el_t) * torch.cos(az_t))
    y = (r * torch.cos(el_t) * torch.sin(az_t))
    z = (r * torch.sin(el_t))
    radar = torch.cat([r, az_t, el_t, doppler, power, x, y, z], dim=-1).to(device)

    model = RadarPointNetBackbone().to(device)
    model.eval()
    with torch.no_grad():
        per_point_feat, coarse = model(radar)
    print("Input radar:", radar.shape)
    print("Per-point features:", per_point_feat.shape)  # (B,N,128)
    print("Coarse global:", coarse.shape)              # (B,1,1024) expected

    # test dataset encoding path (numpy)
    rp_np = np.concatenate([
        (r.cpu().numpy()),
        (az_t.cpu().numpy()),
        (el_t.cpu().numpy()),
        (doppler.cpu().numpy()),
        (power.cpu().numpy()),
        (x.cpu().numpy()),
        (y.cpu().numpy()),
        (z.cpu().numpy())
    ], axis=-1)  # careful: shapes must match, building here for a check

    # rp_np currently (B,N,8) but concatenation above might produce shape mismatch due to dims; skip deep test
    print("Sanity check complete.")
