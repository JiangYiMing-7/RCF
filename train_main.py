# ============================================
# 功能: 训练主文件（函数式接口）
# 说明:
#   - 所有流程以函数形式提供，无命令行接口
#   - 支持随机种子、GPU自动选择、断点续训、训练日志保存
#   - 调用 train_eval.train_model_full 执行完整训练流程
# ============================================

import os
import json
import time
import random
from typing import Dict, Any, Optional

import numpy as np
import torch

# 确保工程模块在同一目录或已加入 sys.path
from data_loader import KITTIDatasetRadar4D
from model_pointnet import PointNet2Backbone
from model_fusion import build_image_encoder
from model_rcnn import PointFusionRCNN
from train_eval import train_model_full, validate_and_save_predictions

# -------------------------
# 配置与工具函数
# -------------------------
def set_random_seed(seed: int = 42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 以下项用于确定性（代价是可能降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(obj: Dict[str, Any], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -------------------------
# 构建模型函数
# -------------------------
def build_models(cfg: Dict[str, Any], device: torch.device):
    """
    根据配置构造并返回 backbone, image_encoder, rcnn 三个模块（已移动到 device）
    cfg: 包含模型配置的字典（backbone/out_channels, image_backbone, rcnn params 等）
    """
    # PointNet2 Backbone
    input_channels = cfg.get('point_input_channels', 0) # 按需修改
    out_channels = cfg.get('point_out_channels', 128)
    backbone = PointNet2Backbone(input_channels=input_channels)
    # Image encoder
    img_backbone = cfg.get('image_backbone', 'resnet18')
    pretrained = cfg.get('image_pretrained', True)
    image_encoder = build_image_encoder(backbone=img_backbone, pretrained=pretrained)
    # RCNN head
    feature_dim = out_channels + image_encoder.out_channels  # point feat + image feat (拼接)
    num_classes = cfg.get('num_classes', 3)  # 按需修改
    num_proposals = cfg.get('num_proposals', 256)
    rcnn = PointFusionRCNN(feature_dim=feature_dim, num_classes=num_classes, num_proposals=num_proposals)

    # move to device
    backbone.to(device)
    image_encoder.to(device)
    rcnn.to(device)
    return backbone, image_encoder, rcnn


# -------------------------
# 构建数据集函数
# -------------------------
def build_datasets(cfg: Dict[str, Any]):
    """
    构造训练与验证数据集实例（KITTIDataset）
    cfg 字段使用:
      - data_root
      - train_split_name (e.g., 'training')
      - val_split_name (e.g., 'validation')
    返回 train_dataset, val_dataset
    """
    root = cfg['data_root']
    train_split = cfg.get('train_split', 'training')
    val_split = cfg.get('val_split', 'validation')
    train_ds = KITTIDatasetRadar4D(root_dir=root, split=train_split, transform=None)
    val_ds = KITTIDatasetRadar4D(root_dir=root, split=val_split, transform=None)
    return train_ds, val_ds


# -------------------------
# 保存训练配置与日志
# -------------------------
def init_experiment(save_root: str, cfg: Dict[str, Any]):
    """
    初始化实验目录，保存配置文件并返回路径
    结构:
      save_root/
        checkpoints/
        logs/
        config.json
    """
    ensure_dir(save_root)
    ckpt_dir = os.path.join(save_root, 'checkpoints')
    log_dir = os.path.join(save_root, 'logs')
    ensure_dir(ckpt_dir)
    ensure_dir(log_dir)
    # 保存配置
    cfg_path = os.path.join(save_root, 'config.json')
    save_json(cfg, cfg_path)
    return {'save_root': save_root, 'ckpt_dir': ckpt_dir, 'log_dir': log_dir, 'cfg_path': cfg_path}


# -------------------------
# 断点加载函数
# -------------------------
def load_checkpoint_if_available(backbone, image_model, rcnn_model, optimizer, ckpt_path: Optional[str], device):
    """
    若提供 ckpt_path 且文件存在，则加载 checkpoint 并恢复模型与优化器状态
    返回: 恢复的 epoch(int) 或 0
    NOTE: 我们期望 ckpt 格式与 train_eval.train_model_full 保存的格式一致
    """
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return 0
    ckpt = torch.load(ckpt_path, map_location=device)
    # 支持两种保存格式：若包含 'backbone_state' 等字段，使用该格式；否则尝试通用键
    if 'backbone_state' in ckpt and 'image_state' in ckpt and 'rcnn_state' in ckpt:
        backbone.load_state_dict(ckpt['backbone_state'])
        image_model.load_state_dict(ckpt['image_state'])
        rcnn_model.load_state_dict(ckpt['rcnn_state'])
        if optimizer is not None and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        return ckpt.get('epoch', 0)
    elif 'model_state_dicts' in ckpt:
        st = ckpt['model_state_dicts']
        if 'backbone' in st:
            backbone.load_state_dict(st['backbone'])
        if 'image' in st:
            image_model.load_state_dict(st['image'])
        if 'rcnn' in st:
            rcnn_model.load_state_dict(st['rcnn'])
        if optimizer is not None and 'optimizer_state' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        return ckpt.get('epoch', 0)
    else:
        # 不识别的格式，抛出错误以提醒用户检查
        raise RuntimeError(f"无法识别的 checkpoint 文件结构: {ckpt_path}")


# -------------------------
# 日志记录
# -------------------------
class SimpleLogger:
    def __init__(self, log_file: str):
        self.log_file = log_file
        ensure_dir(os.path.dirname(self.log_file))
        # 写入头部
        with open(self.log_file, 'a') as f:
            f.write(f"\n---- New run at {time.asctime()} ----\n")

    def info(self, msg: str):
        s = f"[{time.asctime()}] {msg}"
        print(s)
        with open(self.log_file, 'a') as f:
            f.write(s + "\n")


# -------------------------
# 主训练入口函数
# -------------------------
def run_training_pipeline(config: Dict[str, Any]):
    """
      - config: 一个字典，包含必要的字段（见下）
    必需字段（config）:
      - data_root: 数据根路径（KITTI-like 目录结构）
      - save_root: 保存实验的根目录
      - seed: 随机种子
      - device: 'cuda' or 'cpu' 或 None(自动选择)
      - model 配置: point_input_channels, point_out_channels, image_backbone, image_pretrained, num_classes, num_proposals
      - train 超参: num_epochs, batch_size, lr, weight_decay, iou_pos_thresh, iou_neg_thresh, score_threshold
      - resume_checkpoint: 可选 checkpoint path
    返回:
      history 字典（由 train_model_full 返回）
    """
    # 随机种子与设备
    seed = config.get('seed', 42)
    set_random_seed(seed)
    device_str = config.get('device', None)
    if device_str is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    # 初始化实验目录
    save_root = config['save_root']
    meta = init_experiment(save_root, config)
    logger = SimpleLogger(os.path.join(meta['log_dir'], 'train.log'))
    logger.info("实验目录初始化完成: " + str(meta['save_root']))
    # 构建数据集
    logger.info("加载数据集...")
    train_ds, val_ds = build_datasets(config)
    logger.info(f"训练样本数: {len(train_ds)}, 验证样本数: {len(val_ds)}")
    # 构建模型
    logger.info("构建模型...")
    backbone, image_encoder, rcnn = build_models(config, device)
    logger.info("模型构建完成，参数初始化。")
    # 如果指定 resume checkpoint，则加载（这里只做检查，不创建 optimizer）
    resume_ckpt = config.get('resume_checkpoint', None)
    if resume_ckpt:
        # 为了加载模型权重，需要临时 optimizer（或传 None 到加载函数）
        try:
            load_checkpoint_if_available(backbone, image_encoder, rcnn, None, resume_ckpt, device)
            logger.info(f"已从 {resume_ckpt} 恢复模型权重（optimizer 未恢复）")
        except Exception as e:
            logger.info(f"加载 checkpoint 失败: {e}")
            raise

    # 调用 train_model_full
    logger.info("开始训练（调用 train_eval.train_model_full）...")
    # 传入 train_ds, val_ds, 模型以及训练参数
    history = train_model_full(
        train_dataset=train_ds,
        val_dataset=val_ds,
        model_backbone=backbone,
        model_image=image_encoder,
        model_rcnn=rcnn,
        device=device,
        classes=config.get('classes', ['Car', 'Cyclist', 'Truck']),
        num_epochs=config.get('num_epochs', 30),
        batch_size=config.get('batch_size', 4),
        lr=config.get('lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-4),
        save_dir=meta['ckpt_dir'],
        iou_pos_thresh=config.get('iou_pos_thresh', 0.6),
        iou_neg_thresh=config.get('iou_neg_thresh', 0.45),
        score_threshold=config.get('score_threshold', 0.3)
    )
    logger.info("训练完成。历史指标摘要:")
    logger.info(str(history))
    # 训练结束后保存 history 到 save_root
    hist_path = os.path.join(save_root, 'history.json')
    save_json(history, hist_path)
    logger.info(f"训练历史已保存到 {hist_path}")
    return history


# -------------------------
# 单独的评估/推理调用
# -------------------------
def run_inference_on_dataset(checkpoint_path: str,
                             data_root: str,
                             split: str,
                             out_dir: str,
                             classes: list,
                             device: Optional[torch.device] = None,
                             score_threshold: float = 0.3):
    """
    从 checkpoint 加载模型并在指定 split（例如 'testing' 或 'validation'）上运行推理，保存预测 .txt 并返回评估结果
    返回: eval_res dict（由 train_eval.validate_and_save_predictions 返回的内容）
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 构建数据集（单进程，batch_size=1）
    dataset = KITTIDatasetRadar4D(root_dir=data_root, split=split, transform=None)
    # val_loader 与 validate_and_save_predictions 预期结构：可传入 iterator yielding sample dicts
    # 构造 list loader
    data_list = [dataset[i] for i in range(len(dataset))]
    # 构建模型（参数从配置文件尝试读取 save_root/config.json；否则使用默认）
    # 尝试从 checkpoint 文件夹加载配置
    # 由于 checkpoint 格式可能不同，在此采用 conservative 方法：构造模型时使用默认超参数
    # 使用默认设置
    backbone = PointNet2Backbone(input_channels=0).to(device)
    image_encoder = build_image_encoder(backbone='resnet18', pretrained=False).to(device)
    rcnn = PointFusionRCNN(feature_dim=128 + image_encoder.out_channels, num_classes=len(classes)).to(device)
    # 加载 checkpoint（恢复权重）
    load_checkpoint_if_available(backbone, image_encoder, rcnn, None, checkpoint_path, device)
    # 构造 val_loader（generator）
    def simple_gen():
        for s in data_list:
            yield s
    # 调用 validate_and_save_predictions
    ensure_dir(out_dir)
    eval_res = validate_and_save_predictions(simple_gen(), backbone, image_encoder, rcnn, device, out_dir, classes, score_threshold=score_threshold)
    return eval_res



def example_run():
    """
    这是一个示例调用函数，请在外部脚本中调用 run_training_pipeline(config)
    """
    config = {
        'data_root': '/path/to/KITTI',     # 修改为你的数据路径
        'save_root': './experiment_demo',
        'seed': 42,
        'device': None,
        'point_input_channels': 0,
        'point_out_channels': 128,
        'image_backbone': 'resnet18',
        'image_pretrained': True,
        'num_classes': 1,
        'num_proposals': 256,
        'train_split': 'training',
        'val_split': 'validation',
        'classes': ['Car', 'Cyclist', 'Truck'],
        'num_epochs': 20,
        'batch_size': 4,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'iou_pos_thresh': 0.6,
        'iou_neg_thresh': 0.45,
        'score_threshold': 0.3,
        'resume_checkpoint': None
    }
    history = run_training_pipeline(config)
    print("示例训练完成，history keys:", history.keys())


# 仅在交互式导入时不执行。需手动调用 run_training_pipeline / run_inference_on_dataset。
if __name__ == "__main__":
    # 不自动运行训练，避免误操作
    print("train_main.py 已加载。请在交互式环境中调用 run_training_pipeline(config) 来开始训练。")
