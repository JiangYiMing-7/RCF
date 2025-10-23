# ============================================
# 文件名: inference_main.py
# 功能: 模型推理与结果导出（符合4D毫米波雷达+单目融合赛题规范）
# 说明:
#   - 所有功能以函数形式实现（无命令行）
#   - 支持批量目录遍历、KITTI格式输出、mAP评估
# ============================================

import os
import json
import time
import glob
import numpy as np
import torch
from typing import Dict, Any, List, Optional

from data_loader import KITTIDataset
from model_pointnet import PointNet2Backbone
from model_fusion import build_image_encoder
from model_rcnn import PointFusionRCNN
from train_eval import load_checkpoint_full, compute_3d_iou, evaluate_map


# -------------------------
# 工具函数
# -------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def load_json(path: str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(obj: Dict[str, Any], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -------------------------
# 模型加载函数
# -------------------------
def build_and_load_model(config_path: str,
                         checkpoint_path: str,
                         device: torch.device):
    """
    根据 config.json 构建模型并加载 checkpoint 权重
    返回 (backbone, image_encoder, rcnn)
    """
    cfg = load_json(config_path)
    # 构建模型
    input_channels = cfg.get('point_input_channels', 0)
    out_channels = cfg.get('point_out_channels', 128)
    img_backbone = cfg.get('image_backbone', 'resnet18')
    pretrained = cfg.get('image_pretrained', False)
    num_classes = cfg.get('num_classes', 1)
    num_proposals = cfg.get('num_proposals', 256)

    backbone = PointNet2Backbone(input_channels=input_channels).to(device)
    image_encoder = build_image_encoder(backbone=img_backbone, pretrained=pretrained).to(device)
    feature_dim = out_channels + image_encoder.out_channels
    rcnn = PointFusionRCNN(feature_dim=feature_dim, num_classes=num_classes, num_proposals=num_proposals).to(device)

    # 加载权重
    optimizer = None
    load_checkpoint_full(backbone, image_encoder, rcnn, optimizer, checkpoint_path, device)
    return backbone, image_encoder, rcnn, cfg


# -------------------------
# 推理核心函数
# -------------------------
@torch.no_grad()
def inference_on_dataset(dataset: KITTIDataset,
                         backbone: torch.nn.Module,
                         image_encoder: torch.nn.Module,
                         rcnn: torch.nn.Module,
                         device: torch.device,
                         save_dir: str,
                         score_threshold: float = 0.3):
    """
    对整个数据集执行推理并保存 KITTI 格式结果
    dataset: KITTIDataset 实例（split='testing'）
    save_dir: 输出目录，生成每帧的 .txt 文件
    """
    ensure_dir(save_dir)
    backbone.eval()
    image_encoder.eval()
    rcnn.eval()

    all_predictions = []
    all_ground_truths = []
    total_time = 0.0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        frame_id = sample['frame_id']
        img = torch.from_numpy(sample['image']).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        points = torch.from_numpy(sample['points']).float().unsqueeze(0).to(device)
        calibs = [sample['calib']]
        gt_labels = sample.get('labels', [])

        start_time = time.time()
        # 前向计算
        img_feat = image_encoder(img)
        pts_xyz = points[:, :, :3]
        point_feat, _ = backbone(pts_xyz, None)

        from model_fusion import sample_image_features
        sampled_img_feat = sample_image_features(pts_xyz, img_feat, calibs, (img.shape[2], img.shape[3]))
        fused_feat = torch.cat([point_feat, sampled_img_feat], dim=-1)
        proposals, cls_logits, reg_preds = rcnn(pts_xyz, fused_feat)
        infer_time = time.time() - start_time
        total_time += infer_time

        # 转换输出格式
        cls_scores = torch.softmax(cls_logits, dim=-1)
        num_cls = cls_scores.shape[-1]
        preds = []
        for i in range(proposals.shape[1]):
            score = float(cls_scores[0, i].max().cpu().numpy())
            label = int(cls_scores[0, i].argmax().cpu().numpy())
            if score < score_threshold:
                continue
            box = reg_preds[0, i].cpu().numpy()
            preds.append((label, box, score))

        # 保存为 KITTI 格式文件
        save_path = os.path.join(save_dir, f"{frame_id}.txt")
        save_kitti_txt(save_path, preds)

        # 记录预测与GT用于评估
        gt_boxes = []
        for obj in gt_labels:
            dims = obj['dimensions']
            loc = obj['location']
            ry = obj['rotation_y']
            box = np.array([loc[0], loc[1], loc[2], dims[2], dims[1], dims[0], ry], dtype=np.float32)
            gt_boxes.append(box)
        all_predictions.append([(p[1], p[2]) for p in preds])
        all_ground_truths.append(gt_boxes)

    avg_time = total_time / len(dataset)
    print(f"推理完成: {len(dataset)} 帧, 平均每帧耗时 {avg_time:.3f}s")

    return all_predictions, all_ground_truths


# -------------------------
# KITTI格式结果保存
# -------------------------
def save_kitti_txt(save_path: str, preds: List[tuple]):
    """
    将预测结果保存为 KITTI 格式 .txt 文件
    格式: [type truncated occluded alpha bbox h w l x y z ry score]
    由于本项目无2D检测环节，bbox用0填充。
    """
    with open(save_path, 'w') as f:
        for (cls_id, box, score) in preds:
            # box = [x, y, z, l, w, h, ry]
            x, y, z, l, w, h, ry = box
            obj_type = {0: 'Car', 1: 'Cyclist', 2: 'Pedestrian'}.get(cls_id, 'Unknown')
            line = f"{obj_type} 0 0 0 0 0 0 0 {h:.2f} {w:.2f} {l:.2f} {x:.2f} {y:.2f} {z:.2f} {ry:.2f} {score:.3f}\n"
            f.write(line)


# -------------------------
# mAP评估与结果保存
# -------------------------
def evaluate_and_save(predictions, ground_truths, save_path: str):
    """
    使用 train_eval.evaluate_map() 计算 mAP 并保存结果JSON
    """
    res = evaluate_map(predictions, ground_truths, iou_threshold=0.5)
    save_json(res, save_path)
    print(f"评估结果已保存至 {save_path}")
    return res


# -------------------------
# 主推理管线函数
# -------------------------
def run_inference_pipeline(experiment_dir: str,
                           checkpoint_name: str = 'best_model.pth',
                           output_dir: str = None,
                           split: str = 'testing',
                           score_threshold: float = 0.3,
                           eval_mode: bool = True):
    """
    主推理函数
    参数:
        experiment_dir: 训练输出目录（包含 config.json 与 checkpoints/）
        checkpoint_name: 要加载的 checkpoint 文件名
        output_dir: 推理结果输出目录（若为空则为 experiment_dir/results）
        split: 测试数据集 split 名称
        score_threshold: 筛选预测的置信度阈值
        eval_mode: 若为 True 则执行 mAP 评估
    """
    config_path = os.path.join(experiment_dir, 'config.json')
    checkpoint_path = os.path.join(experiment_dir, 'checkpoints', checkpoint_name)
    if output_dir is None:
        output_dir = os.path.join(experiment_dir, 'results')
    ensure_dir(output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone, image_encoder, rcnn, cfg = build_and_load_model(config_path, checkpoint_path, device)

    # 构建测试集
    data_root = cfg['data_root']
    dataset = KITTIDataset(root_dir=data_root, split=split, transform=None)

    print(f"开始推理，共 {len(dataset)} 帧。保存目录: {output_dir}")
    preds, gts = inference_on_dataset(dataset, backbone, image_encoder, rcnn, device, output_dir, score_threshold)

    if eval_mode and len(gts[0]) > 0:
        eval_save = os.path.join(output_dir, 'evaluation_results.json')
        res = evaluate_and_save(preds, gts, eval_save)
        print("mAP评估结果:", res)
        return res
    else:
        print("推理完成，无评估（可能为测试集无GT标签）。")
        return None


# -------------------------
# 示例运行
# -------------------------
def example_inference_run():
    """
    示例函数：展示如何调用推理主流程
    """
    exp_dir = './experiment_demo'  # 修改为你的实验目录
    run_inference_pipeline(
        experiment_dir=exp_dir,
        checkpoint_name='best_model.pth',
        output_dir=os.path.join(exp_dir, 'results'),
        split='validation',
        score_threshold=0.3,
        eval_mode=True
    )


if __name__ == "__main__":
    print("inference_main.py 已加载。请调用 run_inference_pipeline(experiment_dir=...) 运行推理。")
