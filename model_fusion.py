# ============================================
# 功能: 图像特征编码、点云到图像投影采样、点级特征融合模块
# 说明: 提供 build_image_encoder、sample_image_features、fuse_features 等接口
# ============================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np


def _to_tensor_matrix(mat, device):
    """
    辅助函数：将 numpy 矩阵或 torch 矩阵转换为 torch.Tensor（float32）并移动到 device 上。
    支持输入为 numpy.ndarray、torch.Tensor。
    """
    if isinstance(mat, np.ndarray):
        return torch.from_numpy(mat.astype(np.float32)).to(device)
    elif isinstance(mat, torch.Tensor):
        return mat.float().to(device)
    else:
        raise TypeError("calibration matrix must be numpy.ndarray or torch.Tensor")


class ImageEncoder(nn.Module):
    """
    图像编码器：使用ResNet（可选）提取语义特征图
    - 默认使用 ResNet18（截断到 conv5 之前），输出通道为512。
    - 若需更轻量或不同backbone可以替换 build_image_encoder 函数中的选项。
    """
    def __init__(self, backbone='resnet18', pretrained=True):
        super(ImageEncoder, self).__init__()
        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            # 截断全连接层与avgpool，保留到 layer4（输出 feature map）
            self.encoder = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            )
            self.out_channels = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.encoder = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            )
            self.out_channels = 512
        else:
            raise ValueError("unsupported backbone: " + str(backbone))

    def forward(self, x):
        """
        输入:
            x: (B, 3, H, W) 原始图像（已归一化）
        输出:
            feat: (B, C, Hf, Wf) 特征图
        """
        return self.encoder(x)


def build_image_encoder(backbone='resnet18', pretrained=True):
    """
    工厂函数：构建并返回 ImageEncoder 实例
    """
    return ImageEncoder(backbone=backbone, pretrained=pretrained)


def _build_projection_matrix_batch(calib_list, device):
    """
    将一批标定信息（list of dict）转换为 batched 投影矩阵（B, 3, 4）
    输入:
        calib_list: 长度为 B 的 list，每个元素为 dict, 包含 'P2','R0_rect','Tr_velo_to_cam'（numpy 或 torch）
    返回:
        proj_batch: (B, 3, 4) torch.Tensor
    说明:
        投影矩阵计算方式：proj = P2 * R0_rect[:3,:] * Tr_velo_to_cam
        其中 R0_rect 是 4x4，Tr_velo_to_cam 是 4x4（扩展形式）
    """
    proj_mats = []
    for calib in calib_list:
        if not ('P2' in calib and 'R0_rect' in calib and 'Tr_velo_to_cam' in calib):
            raise KeyError("calib dict must contain keys 'P2', 'R0_rect', 'Tr_velo_to_cam'")
        P2 = calib['P2']
        R0 = calib['R0_rect']
        Tr = calib['Tr_velo_to_cam']
        # 保证为 numpy 或 torch，然后转为 torch
        P2_t = _to_tensor_matrix(P2, device)         # (3,4)
        R0_t = _to_tensor_matrix(R0, device)         # (4,4)
        Tr_t = _to_tensor_matrix(Tr, device)         # (4,4)
        # 计算 3x4 投影矩阵
        proj = P2_t @ R0_t[:3, :] @ Tr_t            # (3,4)
        proj_mats.append(proj.unsqueeze(0))         # (1,3,4)
    proj_batch = torch.cat(proj_mats, dim=0)       # (B,3,4)
    return proj_batch


def sample_image_features(points_xyz, image_features, calib_list, image_size):
    """
    将点云的 (x,y,z) 投影到图像，并在特征图上采样对应的特征（双线性插值）
    支持 batch 操作
    输入:
        points_xyz: (B, N, 3) 点坐标（Velodyne/雷达坐标系），类型 torch.FloatTensor
        image_features: (B, C, Hf, Wf) 图像特征图（ResNet输出）
        calib_list: 长度为 B 的标定字典列表（每个包含 'P2','R0_rect','Tr_velo_to_cam'）
                    这些矩阵可以是 numpy 或 torch
        image_size: (H_img, W_img) 原始图像像素尺寸（高, 宽），用于归一化到 [-1,1]
    输出:
        sampled_feats: (B, N, C) 每个点对应的图像特征；若点投影到图像外或深度<=0，则对应特征为0
    说明:
        - 使用 grid_sample 在 feature map 上采样，先将像素坐标归一化到 [-1,1]
        - 处理 z<=0 的点（位于相机后方）将采样特征置零
    """
    assert points_xyz.ndim == 3 and image_features.ndim == 4
    device = image_features.device
    B, N, _ = points_xyz.shape
    _, C, Hf, Wf = image_features.shape
    H_img, W_img = image_size

    # 计算 batched 投影矩阵 (B,3,4)
    proj_batch = _build_projection_matrix_batch(calib_list, device)  # (B,3,4)

    # 将点扩展为齐次坐标 (B,4,N)
    ones = torch.ones((B, N, 1), dtype=points_xyz.dtype, device=device)
    pts_hom = torch.cat([points_xyz, ones], dim=-1)  # (B,N,4)
    pts_hom = pts_hom.permute(0, 2, 1)               # (B,4,N)

    # Project: cam_pts = proj_mat (3x4) @ pts_hom (4xN) -> (B,3,N)
    proj = proj_batch.bmm(pts_hom)                   # (B,3,N)
    xs = proj[:, 0, :]  # (B,N)
    ys = proj[:, 1, :]
    zs = proj[:, 2, :]

    # 处理深度：避免除零
    eps = 1e-6
    valid_mask = zs > eps   # (B,N) 在相机前方的点
    # 计算像素坐标（浮点）
    xs = xs / (zs + (~valid_mask).float() * 1.0)  # 若无效点，先避免 nan
    ys = ys / (zs + (~valid_mask).float() * 1.0)

    # 归一化至 [-1,1]（基于原始图像尺寸）
    # 注意 grid_sample 的坐标系：x 对应宽方向（列），y 对应高方向（行）
    x_norm = 2.0 * (xs / (W_img - 1.0)) - 1.0
    y_norm = 2.0 * (ys / (H_img - 1.0)) - 1.0

    # 组合为 grid，shape -> (B, N, 1, 2) where last dim = (x, y)
    grid = torch.stack([x_norm, y_norm], dim=2).unsqueeze(2)  # (B,N,1,2)

    # 使用 grid_sample 在 image_features 上采样：需要 reshape grid 为 (B, N, 1, 2)
    # grid_sample expects grid in shape (B, H_out, W_out, 2). 我们想要 N 个采样点 -> treat H_out=N, W_out=1
    # 因此输入 image_features: (B,C,Hf,Wf), grid: (B, N, 1, 2) -> output: (B, C, N, 1)
    sampled = F.grid_sample(image_features, grid, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, C, N, 1)
    sampled = sampled.squeeze(3).permute(0, 2, 1).contiguous()  # (B, N, C)

    # 对于投影到图像外或位于相机后方的点，将对应特征置0
    valid_mask = valid_mask.unsqueeze(-1)  # (B, N, 1)
    sampled = sampled * valid_mask.float()

    return sampled  # (B, N, C)


class SimplePointImageAttention(nn.Module):
    """
    简单的点-图像特征注意力融合模块
    - 输入：point_feats (B,N,Cp)， image_feats_sampled (B,N,Ci)
    - 输出：fused_feats (B,N,Cout)
    融合策略：计算点特征作为 Query，图像特征作为 Key/Value，生成 attention 权重后进行加权求和并拼接
    """
    def __init__(self, point_dim, img_dim, hidden_dim=128, out_dim=None):
        super(SimplePointImageAttention, self).__init__()
        self.point_proj = nn.Linear(point_dim, hidden_dim, bias=False)
        self.img_proj = nn.Linear(img_dim, hidden_dim, bias=False)
        self.att_fc = nn.Linear(hidden_dim, 1, bias=False)
        if out_dim is None:
            out_dim = point_dim + img_dim
        self.out_fc = nn.Linear(point_dim + img_dim, out_dim)

    def forward(self, point_feats, img_feats):
        """
        point_feats: (B, N, Cp)
        img_feats: (B, N, Ci)
        返回:
            fused: (B, N, Cout)
        """
        # 计算 attention logits（逐点）
        q = self.point_proj(point_feats)  # (B,N,H)
        k = self.img_proj(img_feats)      # (B,N,H)
        # element-wise interaction then activation
        x = F.relu(q * k)                 # (B,N,H)
        alpha = torch.sigmoid(self.att_fc(x))  # (B,N,1)，在 (0,1)
        # 加权图像特征
        attended_img = alpha * img_feats   # (B,N,Ci)
        # 拼接并映射
        fused = torch.cat([point_feats, attended_img], dim=-1)  # (B,N,Cp+Ci)
        fused = self.out_fc(fused)  # (B,N,Cout)
        return fused


def fuse_features(point_feats, sampled_img_feats, method='concat', attention_module=None):
    """
    将点云特征与采样到的图像特征进行融合
    输入:
        point_feats: (B, N, Cp) 点特征
        sampled_img_feats: (B, N, Ci) 对应的图像特征（由 sample_image_features 得到）
        method: 'concat' 或 'attention'
        attention_module: 若 method='attention'，传入一个注意力模块实例（如 SimplePointImageAttention）
    输出:
        fused_feats: (B, N, Cout)
    说明:
        - concat: 直接沿通道拼接 -> (Cp + Ci)
        - attention: 使用 attention_module 来进行融合（需返回 (B,N,Cout)）
    """
    if method == 'concat':
        fused = torch.cat([point_feats, sampled_img_feats], dim=-1)
        return fused
    elif method == 'attention':
        if attention_module is None:
            raise ValueError("attention_module must be provided when method='attention'")
        return attention_module(point_feats, sampled_img_feats)
    else:
        raise ValueError("unsupported fusion method: " + str(method))


# -------------------------
# 单元测试：检查形状与投影正确性（在无GPU环境也能运行）
# -------------------------
def _shape_check():
    """
    基本单元测试，产生随机点与图像特征，测试采样与融合输出形状
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B = 2
    N = 4096
    C_img = 512
    H_img, W_img = 375, 1242  # KITTI 原始图像大小示例
    # 随机生成点坐标（假定在前方）
    pts = torch.rand(B, N, 3, device=device) * torch.tensor([50.0, 10.0, 2.0], device=device)  # x,y,z 范围示例
    # 随机生成图像特征图（ResNet 输出尺寸示例）
    Hf, Wf = 12, 39  # 约等于 375/32, 1242/32
    img_feats = torch.rand(B, C_img, Hf, Wf, device=device)
    # 构建简单 calib（这里用单位变换作为示例，实际应使用真实标定）
    P2 = np.array([[700.0, 0.0, W_img/2.0, 0.0],
                   [0.0, 700.0, H_img/2.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    R0_rect = np.eye(4, dtype=np.float32)
    Tr = np.eye(4, dtype=np.float32)
    calib_list = [{'P2': P2, 'R0_rect': R0_rect, 'Tr_velo_to_cam': Tr} for _ in range(B)]
    # 采样图像特征
    sampled = sample_image_features(pts, img_feats, calib_list, (H_img, W_img))
    print("sampled_feats shape:", sampled.shape)  # 期望 (B, N, C_img)
    # 随机点特征
    point_feats = torch.rand(B, N, 64, device=device)
    # 融合（拼接）
    fused = fuse_features(point_feats, sampled, method='concat')
    print("fused_feats shape (concat):", fused.shape)  # (B, N, 64 + C_img)
    # 融合（注意力）
    att_mod = SimplePointImageAttention(point_dim=64, img_dim=C_img, hidden_dim=128, out_dim=256).to(device)
    fused_att = fuse_features(point_feats, sampled, method='attention', attention_module=att_mod)
    print("fused_feats shape (attention):", fused_att.shape)  # (B, N, 256)


if __name__ == "__main__":
    _shape_check()
