"""
文件名: sanity_check.py
功能: 验证工程模块接口一致性与功能通路正确性
运行要求:
    已存在以下文件:
        - data_loader.py
        - model_pointnet.py
        - model_fusion.py
        - model_rcnn.py
        - train_eval.py
"""

import torch
import numpy as np
import os

from model_pointnet import PointNet2Backbone
from model_fusion import sample_image_features
from model_rcnn import PointFusionRCNN
from train_eval import compute_3d_iou, evaluate_map, match_proposals_to_gts, load_checkpoint_full

# =======================================================
# 1) 初始化随机种子与设备
# =======================================================
torch.manual_seed(42)
np.random.seed(42)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

print(f"[设备] 使用: {device}")

# =======================================================
# 2) 构建模型骨干、图像分支与检测头
# =======================================================
print("[构建模型] PointNet2Backbone + ImageEncoder + PointFusionRCNN")

# 假设原始点特征通道为3 (可以自定义)
backbone = PointNet2Backbone(input_channels=3).to(device)
from model_fusion import ImageEncoder
image_encoder = ImageEncoder().to(device)
feat_dim = backbone.out_channels + image_encoder.out_channels
rcnn_head = PointFusionRCNN(feature_dim=feat_dim, num_classes=3, num_proposals=256).to(device)

# =======================================================
# 3) 构造伪造数据
# =======================================================
def check_input_sanity(xyz, points=None):
    if not torch.all(torch.isfinite(xyz)):
        raise ValueError("输入 xyz 中含 NaN 或 inf")
    if torch.isnan(xyz).any():
        print("[Warning] xyz 含 NaN，将以 0 替换")
        xyz = torch.nan_to_num(xyz)
    if points is not None and not torch.all(torch.isfinite(points)):
        print("[Warning] points 含 NaN，将以 0 替换")
        points = torch.nan_to_num(points)
    return xyz, points

print("[构造测试数据]")

num_points = 2048
# 点坐标 + 3 通道特征 (可以是RGB、强度等随机特征)
points = np.random.rand(num_points, 6).astype(np.float32)  # 前3列是xyz, 后3列是点特征
image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
calib = np.eye(3, 4).astype(np.float32)

# 单帧 batch 封装
batch_points = torch.from_numpy(points).unsqueeze(0).float().to(device)
batch_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
calibs = [calib]

# 拆分 xyz 与 point feature
xyz = batch_points[:, :, :3]      # (B, N, 3)
feat = batch_points[:, :, 3:]     # (B, N, 3)

# 调用安全检查
xyz, feat = check_input_sanity(xyz, feat)

# =======================================================
# 4) 前向推理流程 (提供点特征)
# =======================================================
print("[前向推理]")

img_feats = image_encoder(batch_image)

pts_xyz = batch_points[:, :, :3]           # B, N, 3
pts_feats = batch_points[:, :, 3:]         # B, N, C  (PointNet2Backbone 期望通道在 dim=1)
print(pts_xyz.shape, pts_feats.shape)

point_feats, _ = backbone(pts_xyz, pts_feats)  # 输入点特征不为 None
img_sampled_feats = sample_image_features(pts_xyz, img_feats, calibs, (256, 256))
fused_feats = torch.cat([point_feats, img_sampled_feats], dim=-1)

proposals, cls_logits, reg_preds = rcnn_head(pts_xyz, fused_feats)

print(f"输出维度: proposals={proposals.shape}, cls_logits={cls_logits.shape}, reg_preds={reg_preds.shape}")

# =======================================================
# 5) IoU 与匹配函数验证
# =======================================================
print("[验证 IoU 与匹配函数]")

box_a = np.array([0, 0, 0, 4, 2, 2, 0], dtype=np.float32)
box_b = np.array([0.5, 0.2, 0, 4, 2, 2, 0], dtype=np.float32)
iou_val = compute_3d_iou(box_a, box_b)
print(f"IoU(box_a, box_b) = {iou_val:.4f}")

props = np.random.rand(10, 7).astype(np.float32)
gts = np.random.rand(3, 7).astype(np.float32)
matches, ious = match_proposals_to_gts(props, gts, 0.5, 0.3)
print(f"匹配结果样例: {matches}, IoU最大值范围=({ious.min():.3f}, {ious.max():.3f})")

# =======================================================
# 6) mAP 评估函数验证
# =======================================================
print("[验证 evaluate_map 函数]")

predictions = [[(box_a, 0.9, "Car"), (box_b, 0.7, "Car")]]
ground_truths = [[(box_a, "Car")]]
result = evaluate_map(predictions, ground_truths, iou_threshold=0.5, classes=["Car"])
print(f"mAP 结果: {result}")

# =======================================================
# 7) Checkpoint 加载函数验证（空路径容错）
# =======================================================
print("[验证 load_checkpoint_full 容错]")

ckpt_path = "temp_ckpt.pth"
state_dicts = {
    "backbone": backbone.state_dict(),
    "image": image_encoder.state_dict(),
    "rcnn": rcnn_head.state_dict(),
}
torch.save({"model_state_dicts": state_dicts, "epoch": 1}, ckpt_path)
try:
    epoch_loaded = load_checkpoint_full(backbone, image_encoder, rcnn_head, None, ckpt_path, device)
    print(f"Checkpoint 加载成功，恢复 epoch = {epoch_loaded}")
finally:
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

# =======================================================
# 8) 全流程结束
# =======================================================
print("✅ Sanity Check 全部通过！模型与接口一致性验证完成。")
