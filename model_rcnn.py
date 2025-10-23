# ============================================
# 功能: 三维检测头（Proposal生成、RoI特征聚合、边界框与类别预测）
# 说明: 该模块基于融合后的点级特征输出最终3D检测结果
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# Proposal Layer
class ProposalLayer(nn.Module):
    """
    Proposal Layer:
    基于点云特征（或融合特征）生成候选框（RoI）。
    输出初步3D框的中心、尺寸和方向估计。
    """
    def __init__(self, in_channels, num_proposals=256):
        super(ProposalLayer, self).__init__()
        self.num_proposals = num_proposals
        self.conv1 = nn.Conv1d(in_channels, 256, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 7, kernel_size=1)  # 3D box: x, y, z, l, w, h, ry

    def forward(self, point_features, xyz):
        """
        输入:
            point_features: (B, N, C)
            xyz: (B, N, 3)
        输出:
            proposals: (B, num_proposals, 7)
        """
        B, N, C = point_features.shape
        x = point_features.permute(0, 2, 1)  # (B,C,N)
        x = F.relu(self.bn1(self.conv1(x)))
        preds = self.conv2(x).permute(0, 2, 1)  # (B,N,7)

        # 简单策略：取 topk 特征响应点作为 proposal
        scores = torch.norm(point_features, dim=-1)  # (B,N)
        topk = min(self.num_proposals, N)
        _, idx = torch.topk(scores, topk, dim=-1)
        idx_expand = idx.unsqueeze(-1).repeat(1, 1, 7)
        proposals = torch.gather(preds, 1, idx_expand)  # (B,K,7)
        centers = torch.gather(xyz, 1, idx.unsqueeze(-1).repeat(1, 1, 3))
        proposals[:, :, :3] += centers  # 偏移回原点坐标

        return proposals  # (B,K,7)



# RoI Feature Extractor

class RoIFeatureExtractor(nn.Module):
    """
    RoI特征聚合模块：
    根据候选框收集框内点的特征并进行池化（max/mean），输出固定长度特征向量。
    """
    def __init__(self, pool_method='max', feature_dim=256):
        super(RoIFeatureExtractor, self).__init__()
        self.pool_method = pool_method
        self.feature_dim = feature_dim
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

    def forward(self, points, point_features, proposals):
        """
        输入:
            points: (B, N, 3)
            point_features: (B, N, C)
            proposals: (B, K, 7)
        输出:
            roi_features: (B, K, 256)
        """
        B, N, C = point_features.shape
        _, K, _ = proposals.shape
        roi_features = []

        for b in range(B):
            cur_points = points[b]  # (N,3)
            cur_feats = point_features[b]  # (N,C)
            cur_props = proposals[b]  # (K,7)
            single_feats = []

            for i in range(K):
                cx, cy, cz, l, w, h, ry = cur_props[i]
                # 简单选取框内点
                mask_x = (cur_points[:, 0] > cx - l / 2) & (cur_points[:, 0] < cx + l / 2)
                mask_y = (cur_points[:, 1] > cy - w / 2) & (cur_points[:, 1] < cy + w / 2)
                mask_z = (cur_points[:, 2] > cz - h / 2) & (cur_points[:, 2] < cz + h / 2)
                mask = mask_x & mask_y & mask_z
                pts_in_roi = cur_feats[mask]
                if pts_in_roi.shape[0] == 0:
                    pooled = torch.zeros((self.feature_dim,), device=cur_feats.device)
                else:
                    if self.pool_method == 'max':
                        pooled = pts_in_roi.max(dim=0)[0]
                    else:
                        pooled = pts_in_roi.mean(dim=0)
                single_feats.append(pooled)
            single_feats = torch.stack(single_feats, dim=0)  # (K,C)
            roi_features.append(single_feats)
        roi_features = torch.stack(roi_features, dim=0)  # (B,K,C)
        roi_features = self.fc(roi_features)
        return roi_features  # (B,K,256)


# RCNN Head
class RCNNHead(nn.Module):
    """
    RCNN 检测头：对每个 RoI 的特征进行分类与3D边界框精修
    """
    def __init__(self, in_channels=256, num_classes=3):
        super(RCNNHead, self).__init__()
        self.shared_fc = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        # 分类分支
        self.cls_layer = nn.Linear(256, num_classes)
        # 回归分支（7个参数）
        self.reg_layer = nn.Linear(256, 7)

    def forward(self, roi_features):
        """
        输入:
            roi_features: (B, K, in_channels)
        输出:
            cls_logits: (B, K, num_classes)
            reg_preds: (B, K, 7)
        """
        x = self.shared_fc(roi_features)
        cls_logits = self.cls_layer(x)
        reg_preds = self.reg_layer(x)
        return cls_logits, reg_preds



# PointFusionRCNN

class PointFusionRCNN(nn.Module):
    """
    整体检测网络封装：
    输入为融合特征与坐标，依次生成 proposal、RoI 聚合、分类与回归输出。
    """
    def __init__(self, feature_dim=256, num_classes=3, num_proposals=256):
        super(PointFusionRCNN, self).__init__()
        self.proposal_layer = ProposalLayer(feature_dim, num_proposals)
        self.roi_extractor = RoIFeatureExtractor(pool_method='max', feature_dim=feature_dim)
        self.rcnn_head = RCNNHead(in_channels=256, num_classes=num_classes)

    def forward(self, points, fused_features):
        """
        输入:
            points: (B, N, 3)
            fused_features: (B, N, C)
        输出:
            proposals: (B, K, 7)
            cls_logits: (B, K, num_classes)
            reg_preds: (B, K, 7)
        """
        proposals = self.proposal_layer(fused_features, points)
        roi_features = self.roi_extractor(points, fused_features, proposals)
        cls_logits, reg_preds = self.rcnn_head(roi_features)
        return proposals, cls_logits, reg_preds


# 测试函数
def _shape_check():
    """
    基本形状验证测试：随机输入检查网络流程是否通畅
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, C = 2, 2048, 256
    points = torch.rand(B, N, 3, device=device)
    feats = torch.rand(B, N, C, device=device)
    model = PointFusionRCNN(feature_dim=C, num_classes=3, num_proposals=128).to(device)
    proposals, cls_logits, reg_preds = model(points, feats)
    print("proposals:", proposals.shape)     # (B,K,7)
    print("cls_logits:", cls_logits.shape)   # (B,K,3)
    print("reg_preds:", reg_preds.shape)     # (B,K,7)


if __name__ == "__main__":
    _shape_check()
