# ============================================
# 功能: 训练、验证、评估与 checkpoint 管理
# 接口保证:
#   - compute_3d_iou(box_a, box_b)
#   - evaluate_map(predictions, ground_truths, iou_threshold=0.5, classes=None)
#   - load_checkpoint_full(backbone, image_model, rcnn_model, optimizer, load_path, device)
#   - save_checkpoint(state_dicts, optimizer, epoch, save_path)
#   - train_one_epoch, validate_one_epoch, train_model_full, validate_and_save_predictions
# ============================================

import os
import math
import time
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# -------------------------
# 几何工具与 3D IoU
# -------------------------
def _normalize_angle(angle: float) -> float:
    """归一化角度到 [-pi, pi)"""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def box3d_to_corners_bev(box: np.ndarray) -> np.ndarray:
    """
    将 3D 框 (x,y,z,l,w,h,ry) 转为 BEV 的 4 个角点（逆时针）
    返回 shape (4,2)
    """
    x, y, z, l, w, h, ry = box
    dx = l / 2.0; dy = w / 2.0
    corners = np.array([[ dx,  dy],
                        [ dx, -dy],
                        [-dx, -dy],
                        [-dx,  dy]], dtype=np.float32)
    c = math.cos(ry); s = math.sin(ry)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    corners_rot = corners @ R.T
    corners_rot[:, 0] += x
    corners_rot[:, 1] += y
    return corners_rot

def polygon_area(poly: List[Tuple[float, float]]) -> float:
    """多边形面积（shoelace）"""
    if len(poly) < 3:
        return 0.0
    x = [p[0] for p in poly]; y = [p[1] for p in poly]
    area = 0.0
    n = len(poly)
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j] - x[j] * y[i]
    return abs(area) * 0.5

def is_point_inside_edge(p, a, b):
    """判断点 p 是否在有向边 a->b 的左侧（包含边）"""
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0]) >= -1e-9

def line_intersection(p1, p2, q1, q2):
    """计算线段 p1-p2 与 q1-q2 的交点（若退化返回中点）"""
    x1,y1 = p1; x2,y2 = p2; x3,y3 = q1; x4,y4 = q2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-8:
        # 平行或近似平行，返回中点作为退化解
        return ((x1+x2)/2.0, (y1+y2)/2.0)
    px = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4)) / denom
    py = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4)) / denom
    return (px, py)

def polygon_clip(subjectPolygon: List[Tuple[float, float]], clipPolygon: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Sutherland–Hodgman 多边形裁剪：计算 subjectPolygon 与 clipPolygon 的交多边形
    顶点以列表形式（顺时针或逆时针）给出
    """
    outputList = subjectPolygon
    if len(outputList) == 0:
        return []
    cp1 = clipPolygon[-1]
    for cp2 in clipPolygon:
        inputList = outputList
        outputList = []
        if not inputList:
            break
        s = inputList[-1]
        for e in inputList:
            if is_point_inside_edge(e, cp1, cp2):
                if not is_point_inside_edge(s, cp1, cp2):
                    outputList.append(line_intersection(s, e, cp1, cp2))
                outputList.append(e)
            elif is_point_inside_edge(s, cp1, cp2):
                outputList.append(line_intersection(s, e, cp1, cp2))
            s = e
        cp1 = cp2
    return outputList

def bev_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """计算 BEV 上的旋转矩形 IoU"""
    poly1 = box3d_to_corners_bev(box_a).tolist()
    poly2 = box3d_to_corners_bev(box_b).tolist()
    inter_poly = polygon_clip(poly1, poly2)
    inter_area = polygon_area(inter_poly)
    a1 = polygon_area(poly1); a2 = polygon_area(poly2)
    union = a1 + a2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def compute_3d_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    精确计算两个 3D 有向框的 IoU（考虑旋转）：
      - BEV 多边形交并
      - 高度重叠 (z 中心 + h)
      - 交体积 = inter_area * inter_h
      - IoU = inter_vol / union_vol
    输入:
      box_*: numpy array shape (7,) => [x,y,z,l,w,h,ry]
    返回:
      IoU 浮点数（0~1）
    """
    # BEV 交面积
    poly1 = box3d_to_corners_bev(box_a).tolist()
    poly2 = box3d_to_corners_bev(box_b).tolist()
    inter_poly = polygon_clip(poly1, poly2)
    inter_area = polygon_area(inter_poly)
    # 高度重叠
    za, zb = box_a[2], box_b[2]
    ha, hb = box_a[5], box_b[5]
    a_min, a_max = za - ha/2.0, za + ha/2.0
    b_min, b_max = zb - hb/2.0, zb + hb/2.0
    inter_h = max(0.0, min(a_max, b_max) - max(a_min, b_min))
    inter_vol = inter_area * inter_h
    vol_a = box_a[3] * box_a[4] * box_a[5]
    vol_b = box_b[3] * box_b[4] * box_b[5]
    union = vol_a + vol_b - inter_vol
    if union <= 0:
        return 0.0
    iou = inter_vol / union
    return float(max(0.0, min(1.0, iou)))

# -------------------------
# 评估函数：evaluate_map
# -------------------------
def evaluate_map(predictions: List[List[Tuple[np.ndarray, float, Optional[str]]]],
                 ground_truths: List[List[Tuple[np.ndarray, Optional[str]]]],
                 iou_threshold: float = 0.5,
                 classes: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    通用 mAP / precision / recall 计算函数，兼容单类与多类输入。
    inputs:
      predictions: list of length N_images, each item is a list of tuples:
                   - (box(np.array[7]), score(float))  或
                   - (box(np.array[7]), score(float), class_str)
      ground_truths: list of length N_images, each item is list of:
                   - box(np.array[7]) 或 (box(np.array[7]), class_str)
      iou_threshold: IoU 阈值（默认 0.5）
      classes: 可选类名列表（若提供则按类别计算 AP）
    returns:
      dict {
        'mAP': float,
        'per_class_ap': {class_name: AP},
        'precision': overall_precision,
        'recall': overall_recall
      }
    说明:
      - 如果 classes 未提供，函数会把所有预测视为单类（计算单类 AP）
      - predictions 中允许包含或不包含 class 信息；若包含，会按类聚合
    """
    # 标准化输入：将 predictions 转为每帧 (box, score, class_str) 的形式（class_str 可为 None）
    N = len(predictions)
    preds_per_img = []
    for i in range(N):
        preds_img = []
        for item in predictions[i]:
            if len(item) == 3:
                box, score, cls = item
            elif len(item) == 2:
                box, score = item
                cls = None
            else:
                raise ValueError("predictions element must be (box, score) or (box, score, class)")
            preds_img.append((np.array(box, dtype=np.float32), float(score), cls))
        preds_per_img.append(preds_img)
    gts_per_img = []
    for i in range(N):
        gts_img = []
        for item in ground_truths[i]:
            if len(item) == 2:
                box, cls = item
            else:
                box = item
                cls = None
            gts_img.append((np.array(box, dtype=np.float32), cls))
        gts_per_img.append(gts_img)
    # Determine classes
    if classes is None:
        # collect classes from GTs and preds; if none, treat as single class '__all__'
        class_set = set()
        for img in gts_per_img:
            for _, cls in img:
                if cls is not None:
                    class_set.add(cls)
        for img in preds_per_img:
            for _, _, cls in img:
                if cls is not None:
                    class_set.add(cls)
        if len(class_set) == 0:
            classes = ['__all__']
        else:
            classes = sorted(list(class_set))
    # per-class predictions/gts aggregation
    per_class_preds = {c: [] for c in classes}
    per_class_gts = {c: [] for c in classes}
    for img_idx in range(N):
        for c in classes:
            # preds for this class in this image
            p_list = [(box, score) for (box, score, cls) in preds_per_img[img_idx] if (cls == c) or (cls is None and c=='__all__') or (cls is None and len(classes)==1)]
            g_list = [box for (box, cls_gt) in gts_per_img[img_idx] if (cls_gt == c) or (cls_gt is None and c=='__all__') or (cls_gt is None and len(classes)==1)]
            per_class_preds[c].append(p_list)
            per_class_gts[c].append(g_list)
    per_class_ap = {}
    per_class_prec = {}
    per_class_rec = {}
    # compute per-class AP
    for c in classes:
        all_preds = []  # list of (img_idx, box, score)
        gt_count = 0
        for img_idx in range(N):
            for (box, score) in per_class_preds[c][img_idx]:
                all_preds.append((img_idx, box, score))
            gt_count += len(per_class_gts[c][img_idx])
        if gt_count == 0:
            per_class_ap[c] = 0.0
            per_class_prec[c] = 0.0
            per_class_rec[c] = 0.0
            continue
        # sort by score desc
        all_preds.sort(key=lambda x: x[2], reverse=True)
        n_preds = len(all_preds)
        tp = np.zeros((n_preds,), dtype=np.int32)
        fp = np.zeros((n_preds,), dtype=np.int32)
        # per-image matched flags
        matched = {i: np.zeros((len(per_class_gts[c][i]),), dtype=np.int32) for i in range(N)}
        for idx_pred, (img_idx, box, score) in enumerate(all_preds):
            best_iou = 0.0
            best_j = -1
            for j, gt_box in enumerate(per_class_gts[c][img_idx]):
                if matched[img_idx][j] == 1:
                    continue
                iou = compute_3d_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou >= iou_threshold:
                tp[idx_pred] = 1
                matched[img_idx][best_j] = 1
            else:
                fp[idx_pred] = 1
        tp_cum = np.cumsum(tp).astype(np.float32)
        fp_cum = np.cumsum(fp).astype(np.float32)
        rec = tp_cum / (gt_count + 1e-8)
        prec = tp_cum / (tp_cum + fp_cum + 1e-8)
        # AP via interpolation
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i-1] = max(mpre[i-1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        ap = 0.0
        for i in idx:
            ap += (mrec[i+1] - mrec[i]) * mpre[i+1]
        per_class_ap[c] = float(ap)
        per_class_prec[c] = float(prec[-1]) if prec.size > 0 else 0.0
        per_class_rec[c] = float(rec[-1]) if rec.size > 0 else 0.0
    # mAP mean over classes
    mAP = float(np.mean(list(per_class_ap.values()))) if len(per_class_ap) > 0 else 0.0
    # overall precision/recall average
    overall_prec = float(np.mean(list(per_class_prec.values()))) if len(per_class_prec) > 0 else 0.0
    overall_rec = float(np.mean(list(per_class_rec.values()))) if len(per_class_rec) > 0 else 0.0
    return {'mAP': mAP, 'per_class_ap': per_class_ap, 'precision': overall_prec, 'recall': overall_rec}

# -------------------------
# Checkpoint 保存 / 加载
# -------------------------
def save_checkpoint(state_dicts: Dict[str, Any], optimizer: Optional[optim.Optimizer], epoch: int, save_path: str):
    """
    保存 checkpoint：
      state_dicts: {'backbone':..., 'image':..., 'rcnn':...}
      optimizer: optimizer 对象（可为 None）
    """
    state = {'epoch': epoch, 'model_state_dicts': state_dicts}
    if optimizer is not None:
        state['optimizer_state'] = optimizer.state_dict()
    torch.save(state, save_path)

def load_checkpoint_full(backbone: nn.Module, image_model: nn.Module, rcnn_model: nn.Module,
                         optimizer: Optional[optim.Optimizer], load_path: str, device: torch.device) -> int:
    """
    加载 checkpoint，并返回恢复的 epoch
    支持先前多种保存格式（兼容性处理）
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"checkpoint not found: {load_path}")
    ckpt = torch.load(load_path, map_location=device)
    # 支持旧版字段或新版 'model_state_dicts'
    if 'model_state_dicts' in ckpt:
        st = ckpt['model_state_dicts']
        if 'backbone' in st:
            backbone.load_state_dict(st['backbone'])
        if 'image' in st:
            image_model.load_state_dict(st['image'])
        if 'rcnn' in st:
            rcnn_model.load_state_dict(st['rcnn'])
    else:
        # 支持字段名 'backbone_state' 等（train_main 保存的格式）
        if 'backbone_state' in ckpt:
            backbone.load_state_dict(ckpt['backbone_state'])
        if 'image_state' in ckpt:
            image_model.load_state_dict(ckpt['image_state'])
        if 'rcnn_state' in ckpt:
            rcnn_model.load_state_dict(ckpt['rcnn_state'])
        # 兼容早期直接保存 state_dicts
        if 'backbone' in ckpt:
            try:
                backbone.load_state_dict(ckpt['backbone'])
            except Exception:
                pass
    # 恢复 optimizer
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    epoch = ckpt.get('epoch', 0)
    return int(epoch)

def match_proposals_to_gts(proposals: np.ndarray,
                           gts: np.ndarray,
                           iou_pos_thresh: float = 0.6,
                           iou_neg_thresh: float = 0.45):
    """
    将 proposals 与 ground truth boxes 匹配，生成索引与 IoU 信息
    输入:
      proposals: (K,7) numpy 数组
      gts: (M,7) numpy 数组
      iou_pos_thresh: 正样本阈值
      iou_neg_thresh: 负样本阈值
    输出:
      matches: (K,) numpy int 数组
               - 若为 >=0，则为匹配的 gt 索引
               - 若为 -1，则为负样本（IoU < neg_thresh）
               - 若为 -2，则为忽略样本（neg_thresh <= IoU < pos_thresh）
      ious_max: (K,) 每个 proposal 对应的最大 IoU
    """
    K = proposals.shape[0]
    M = gts.shape[0]
    matches = -2 * np.ones((K,), dtype=np.int32)
    ious_max = np.zeros((K,), dtype=np.float32)
    if M == 0:
        # 无 GT，全部视为负样本
        matches[:] = -1
        return matches, ious_max

    # 计算 IoU 矩阵
    ious = np.zeros((K, M), dtype=np.float32)
    for i in range(K):
        for j in range(M):
            ious[i, j] = compute_3d_iou(proposals[i], gts[j])
    ious_max = np.max(ious, axis=1)
    gt_indices = np.argmax(ious, axis=1)

    for i in range(K):
        iou = ious_max[i]
        if iou >= iou_pos_thresh:
            matches[i] = int(gt_indices[i])  # 正样本
        elif iou < iou_neg_thresh:
            matches[i] = -1  # 负样本
        else:
            matches[i] = -2  # 忽略样本
    return matches, ious_max


# -------------------------
# 损失组合：Classification + Regression + IoU
# -------------------------
class DetectionLoss:
    """
    综合损失，包含:
      - 分类损失（二分类 BCE 或 多类 CrossEntropy，支持 ignore label=-1）
      - 回归损失（SmoothL1，仅正样本）
      - IoU loss (1 - IoU)，仅正样本
    前端需传入:
      - cls_logits: (B,K) 或 (B,K,num_cls)
      - cls_targets: (B,K) int (0/1 或类别 idx)，ignore 使用 -1
      - reg_preds: (B,K,7) encoded deltas
      - reg_targets: (B,K,7)
      - pos_mask: (B,K) bool
      - proposals: (B,K,7)
      - gt_boxes: (B,K,7)
    返回 dict: {'cls_loss', 'reg_loss','iou_loss','loss'}
    """
    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 2.0, iou_weight: float = 1.0, focal_gamma: float = 2.0, focal_alpha: float = 0.25):
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.iou_weight = iou_weight
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor, alpha: float, gamma: float):
        prob = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * ((1 - p_t) ** gamma) * ce_loss
        return loss

    def compute(self, cls_logits: torch.Tensor, cls_targets: torch.Tensor,
                reg_preds: torch.Tensor, reg_targets: torch.Tensor,
                pos_mask: torch.Tensor, proposals: torch.Tensor, gt_boxes: torch.Tensor,
                use_focal: bool = True):
        device = cls_logits.device
        B, K = cls_targets.shape[0], cls_targets.shape[1]
        # 分类损失（支持多分类或二分类）
        if cls_logits.dim() == 3 and cls_logits.shape[-1] > 1:
            logits_flat = cls_logits.view(-1, cls_logits.shape[-1])
            targets_flat = cls_targets.view(-1).long()
            loss_all = F.cross_entropy(logits_flat, targets_flat, reduction='none').view(B, K)
            mask_valid = (cls_targets != -1)
            if mask_valid.sum() > 0:
                cls_loss = (loss_all * mask_valid.float()).sum() / mask_valid.sum()
            else:
                cls_loss = torch.tensor(0.0, device=device)
        else:
            # 二分类
            if cls_logits.dim() == 3:
                logits_bin = cls_logits.squeeze(-1)
            else:
                logits_bin = cls_logits
            labels = cls_targets.float()
            mask_valid = (cls_targets != -1)
            if use_focal and self.focal_gamma > 0:
                per_elem = self.focal_loss(logits_bin, labels, self.focal_alpha, self.focal_gamma)
            else:
                per_elem = F.binary_cross_entropy_with_logits(logits_bin, labels, reduction='none')
            if mask_valid.sum() > 0:
                cls_loss = (per_elem * mask_valid.float()).sum() / mask_valid.sum()
            else:
                cls_loss = torch.tensor(0.0, device=device)
        # 回归 & IoU（仅 pos）
        pos_mask_bool = pos_mask.bool()
        reg_loss = torch.tensor(0.0, device=device)
        iou_loss = torch.tensor(0.0, device=device)
        if pos_mask_bool.sum() > 0:
            pred_deltas = reg_preds[pos_mask_bool]  # (P,7)
            tgt_deltas = reg_targets[pos_mask_bool]  # (P,7)
            reg_loss = self.smoothl1(pred_deltas, tgt_deltas).mean()
            # IoU loss: decode pred boxes and compute IoU with gt_boxes
            proposals_pos = proposals[pos_mask_bool].detach().cpu().numpy()
            gt_pos = gt_boxes[pos_mask_bool].detach().cpu().numpy()
            pred_deltas_np = pred_deltas.detach().cpu().numpy()
            decoded_preds = np.stack([decode_box(proposals_pos[i], pred_deltas_np[i]) for i in range(pred_deltas_np.shape[0])], axis=0)
            ious = np.array([compute_3d_iou(decoded_preds[i], gt_pos[i]) for i in range(decoded_preds.shape[0])], dtype=np.float32)
            iou_loss = torch.tensor((1.0 - ious).mean(), device=device)
        total_loss = self.cls_weight * cls_loss + self.reg_weight * reg_loss + self.iou_weight * iou_loss
        return {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'iou_loss': iou_loss, 'loss': total_loss}

# -------------------------
# encode/decode box（proposal <-> gt）
# -------------------------
def encode_box(proposal: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """编码回归目标（同 train_main 中 encode_box）"""
    px,py,pz,pl,pw,ph,pry = proposal
    gx,gy,gz,gl,gw,gh,gry = gt
    tx = (gx - px) / (pl + 1e-6)
    ty = (gy - py) / (pw + 1e-6)
    tz = (gz - pz) / (ph + 1e-6)
    tl = math.log((gl + 1e-6)/(pl + 1e-6))
    tw = math.log((gw + 1e-6)/(pw + 1e-6))
    th = math.log((gh + 1e-6)/(ph + 1e-6))
    dtheta = _normalize_angle(gry - pry)
    return np.array([tx,ty,tz,tl,tw,th,dtheta], dtype=np.float32)

def decode_box(proposal: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """把编码 delta 解码为 box（同 train_main 中 decode_box）"""
    px,py,pz,pl,pw,ph,pry = proposal
    tx,ty,tz,tl,tw,th,dtheta = delta
    gx = tx * pl + px
    gy = ty * pw + py
    gz = tz * ph + pz
    gl = math.exp(tl) * pl
    gw = math.exp(tw) * pw
    gh = math.exp(th) * ph
    gry = _normalize_angle(pry + dtheta)
    return np.array([gx,gy,gz,gl,gw,gh,gry], dtype=np.float32)

# -------------------------
# 训练与验证循环
# -------------------------
def train_one_epoch(model_backbone: nn.Module,
                    model_image: nn.Module,
                    model_rcnn: nn.Module,
                    dataloader,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    loss_fn: DetectionLoss,
                    iou_pos_thresh: float = 0.6,
                    iou_neg_thresh: float = 0.45,
                    log_interval: int = 10):
    """
    训练一个 epoch（dataloader yields list of samples per batch）
    每个 sample dict 需包含: 'image','points','calib','labels','frame_id'
    """
    model_backbone.train(); model_image.train(); model_rcnn.train()
    running_loss = 0.0
    num_batches = 0
    t0 = time.time()
    for batch_idx, batch_samples in enumerate(dataloader):
        B = len(batch_samples)
        # collate as in train_main: images -> tensor, points padded, calibs list, labels list
        imgs = np.stack([s['image'] for s in batch_samples], axis=0)
        imgs_t = torch.from_numpy(imgs).permute(0,3,1,2).float().to(device) / 255.0
        pts_list = [s['points'] for s in batch_samples]
        maxN = max(p.shape[0] for p in pts_list)
        pts_pad = np.zeros((B, maxN, 4), dtype=np.float32)
        mask = np.zeros((B, maxN), dtype=np.bool_)
        for i in range(B):
            n = pts_list[i].shape[0]
            pts_pad[i,:n,:] = pts_list[i]
            mask[i,:n] = True
        pts_t = torch.from_numpy(pts_pad).float().to(device)
        calibs = [s['calib'] for s in batch_samples]
        labels_batch = [s.get('labels', []) for s in batch_samples]
        # forward
        img_feats = model_image(imgs_t)
        pts_xyz = pts_t[:, :, :3]
        point_feats, _ = model_backbone(pts_xyz, None)
        from model_fusion import sample_image_features
        sampled_img_feats = sample_image_features(pts_xyz, img_feats, calibs, (imgs_t.shape[2], imgs_t.shape[3]))
        fused = torch.cat([point_feats, sampled_img_feats], dim=-1)
        proposals, cls_logits, reg_preds = model_rcnn(pts_xyz, fused)
        # build targets per image (as in train_main)
        Bk, K, _ = proposals.shape
        cls_targets = -1 * torch.ones((B, K), dtype=torch.int64, device=device)
        reg_targets = torch.zeros((B, K, 7), dtype=torch.float32, device=device)
        pos_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        gt_boxes_tensor = torch.zeros((B, K, 7), dtype=torch.float32, device=device)
        props_np = proposals.detach().cpu().numpy()
        for b in range(B):
            gt_objs = labels_batch[b]
            gts = []
            for obj in gt_objs:
                dims = obj['dimensions']; loc = obj['location']; ry = obj['rotation_y']
                gts.append(np.array([loc[0], loc[1], loc[2], dims[2], dims[1], dims[0], ry], dtype=np.float32))
            if len(gts) == 0:
                gts_np = np.zeros((0,7), dtype=np.float32)
            else:
                gts_np = np.stack(gts, axis=0)
            matches, _ = match_proposals_to_gts(props_np[b], gts_np, iou_pos_thresh, iou_neg_thresh)
            for k in range(K):
                m = matches[k]
                if m == -1:
                    cls_targets[b,k] = 0
                elif m == -2:
                    cls_targets[b,k] = -1
                else:
                    cls_targets[b,k] = 1
                    pos_mask[b,k] = True
                    reg_targets[b,k,:] = torch.from_numpy(encode_box(props_np[b][k], gts_np[m]))
                    gt_boxes_tensor[b,k,:] = torch.from_numpy(gts_np[m])
        # compute loss
        loss_dict = loss_fn.compute(cls_logits, cls_targets, reg_preds, reg_targets, pos_mask, proposals, gt_boxes_tensor)
        loss = loss_dict['loss']
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model_backbone.parameters()) + list(model_image.parameters()) + list(model_rcnn.parameters()), max_norm=10.0)
        optimizer.step()
        running_loss += float(loss.item())
        num_batches += 1
        if (batch_idx + 1) % log_interval == 0:
            avg = running_loss / num_batches
            elapsed = time.time() - t0
            print(f"Epoch {epoch} Batch {batch_idx+1} avg_loss {avg:.4f} elapsed {elapsed:.1f}s")
    epoch_loss = running_loss / max(1, num_batches)
    return epoch_loss

def validate_one_epoch(model_backbone: nn.Module,
                       model_image: nn.Module,
                       model_rcnn: nn.Module,
                       dataloader,
                       device: torch.device,
                       score_threshold: float = 0.3):
    """
    在验证集上推理并返回预测与GT（用于 evaluate_map）
    dataloader yields single-sample batches (or list of 1 sample)
    返回:
      predictions_list: list per image of [(box, score, class_str)...]
      gts_list: list per image of [(box, class_str)...]
    """
    model_backbone.eval(); model_image.eval(); model_rcnn.eval()
    preds_all = []
    gts_all = []
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, list) or isinstance(batch, tuple):
                sample = batch[0]
            else:
                sample = batch
            img_np = sample['image']; pts_np = sample['points']; calib = sample['calib']
            labels = sample.get('labels', [])
            # prepare tensors
            img_t = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).float().to(device) / 255.0
            pts_t = torch.from_numpy(pts_np).unsqueeze(0).float().to(device)
            img_feats = model_image(img_t)
            pts_xyz = pts_t[:, :, :3]
            point_feats, _ = model_backbone(pts_xyz, None)
            from model_fusion import sample_image_features
            sampled_img_feats = sample_image_features(pts_xyz, img_feats, [calib], (img_t.shape[2], img_t.shape[3]))
            fused = torch.cat([point_feats, sampled_img_feats], dim=-1)
            proposals, cls_logits, reg_preds = model_rcnn(pts_xyz, fused)
            # parse results
            pred_list = []
            cls_scores = None
            if cls_logits.dim() == 3 and cls_logits.shape[-1] > 1:
                probs = F.softmax(cls_logits[0], dim=-1).cpu().numpy()  # (K, num_cls)
                for k in range(probs.shape[0]):
                    score = float(np.max(probs[k]))
                    cls_id = int(np.argmax(probs[k]))
                    if score < score_threshold:
                        continue
                    delta = reg_preds[0, k].cpu().numpy()
                    prop = proposals[0, k].cpu().numpy()
                    box = decode_box(prop, delta)
                    cls_name = None
                    pred_list.append((box, score, cls_name))
            else:
                if cls_logits.dim() == 3:
                    logits = cls_logits[0,:,0].cpu().numpy()
                else:
                    logits = cls_logits[0].cpu().numpy()
                scores = 1.0/(1.0 + np.exp(-logits))
                for k in range(len(scores)):
                    score = float(scores[k])
                    if score < score_threshold:
                        continue
                    delta = reg_preds[0, k].cpu().numpy()
                    prop = proposals[0, k].cpu().numpy()
                    box = decode_box(prop, delta)
                    pred_list.append((box, score, None))
            preds_all.append(pred_list)
            # GTs
            gt_list = []
            for obj in labels:
                dims = obj['dimensions']; loc = obj['location']; ry = obj['rotation_y']; cls_name = obj.get('type', None)
                box = np.array([loc[0], loc[1], loc[2], dims[2], dims[1], dims[0], ry], dtype=np.float32)
                gt_list.append((box, cls_name))
            gts_all.append(gt_list)
    return preds_all, gts_all

# -------------------------
# validate_and_save_predictions（供 inference_main.py 调用）
# -------------------------
def validate_and_save_predictions(val_loader, model_backbone, model_image, model_rcnn, device, save_dir, classes: List[str], score_threshold: float = 0.3):
    """
    在验证/测试集上推理，保存每帧预测为 KITTI-like .txt（save_dir/predictions）
    并返回 evaluate_map() 的结果字典
    """
    preds_all = []
    gts_all = []
    preds_out_dir = os.path.join(save_dir, 'predictions')
    os.makedirs(preds_out_dir, exist_ok=True)
    model_backbone.eval(); model_image.eval(); model_rcnn.eval()
    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, list) or isinstance(batch, tuple):
                sample = batch[0]
            else:
                sample = batch
            frame_id = sample.get('frame_id', '000000')
            calib = sample['calib']
            img_shape = sample['image'].shape[:2]
            # inference for this sample
            img_t = torch.from_numpy(sample['image']).permute(2,0,1).unsqueeze(0).float().to(device)/255.0
            pts_t = torch.from_numpy(sample['points']).unsqueeze(0).float().to(device)
            img_feats = model_image(img_t)
            pts_xyz = pts_t[:,:, :3]
            pf, _ = model_backbone(pts_xyz, None)
            from model_fusion import sample_image_features
            sampled = sample_image_features(pts_xyz, img_feats, [calib], (img_t.shape[2], img_t.shape[3]))
            fused = torch.cat([pf, sampled], dim=-1)
            proposals, cls_logits, reg_preds = model_rcnn(pts_xyz, fused)
            # build preds list and save txt
            out_preds = []
            if cls_logits.dim() == 3 and cls_logits.shape[-1] > 1:
                probs = F.softmax(cls_logits[0], dim=-1).cpu().numpy()
                for k in range(probs.shape[0]):
                    score = float(np.max(probs[k]))
                    if score < score_threshold:
                        continue
                    cls_id = int(np.argmax(probs[k]))
                    delta = reg_preds[0,k].cpu().numpy()
                    prop = proposals[0,k].cpu().numpy()
                    box = decode_box(prop, delta)
                    out_preds.append((box, score, classes[cls_id] if cls_id < len(classes) else classes[0]))
            else:
                if cls_logits.dim() == 3:
                    logits_vec = cls_logits[0,:,0].cpu().numpy()
                else:
                    logits_vec = cls_logits[0].cpu().numpy()
                scores = 1.0/(1.0 + np.exp(-logits_vec))
                for k, score in enumerate(scores):
                    if score < score_threshold:
                        continue
                    delta = reg_preds[0,k].cpu().numpy()
                    prop = proposals[0,k].cpu().numpy()
                    box = decode_box(prop, delta)
                    out_preds.append((box, float(score), classes[0] if len(classes)>0 else '__all__'))
            # save per-frame txt in KITTI-like format
            txt_path = os.path.join(preds_out_dir, f"{frame_id}.txt")
            with open(txt_path, 'w') as f:
                for (box, score, cls_name) in out_preds:
                    # simple KITTI-like format: class truncated occluded alpha bbox h w l x y z ry score
                    # 这里不计算 2D bbox，写 0 占位；alpha 也写 0 占位（若需可以用 project_box_to_image 实现）
                    x,y,z,l,w,h,ry = box
                    f.write(f"{cls_name} 0 0 0 0 0 0 0 {h:.4f} {w:.4f} {l:.4f} {x:.4f} {y:.4f} {z:.4f} {ry:.6f} {score:.6f}\n")
            # prepare for evaluation
            preds_all.append([(box, score, cls_name) for (box, score, cls_name) in out_preds])
            gts = []
            for obj in sample.get('labels', []):
                dims = obj['dimensions']; loc = obj['location']; ry = obj['rotation_y']; cls_name = obj.get('type', None)
                box = np.array([loc[0], loc[1], loc[2], dims[2], dims[1], dims[0], ry], dtype=np.float32)
                gts.append((box, cls_name))
            gts_all.append(gts)
    # evaluate
    eval_res = evaluate_map(preds_all, gts_all, iou_threshold=0.5, classes=classes)
    eval_res['predictions_dir'] = preds_out_dir
    return eval_res

# -------------------------
# 完整训练流程包装（与 train_main.py 配合）
# -------------------------
def train_model_full(train_dataset,
                     val_dataset,
                     model_backbone: nn.Module,
                     model_image: nn.Module,
                     model_rcnn: nn.Module,
                     device: torch.device,
                     classes: List[str],
                     num_epochs: int = 30,
                     batch_size: int = 4,
                     lr: float = 1e-3,
                     weight_decay: float = 1e-4,
                     save_dir: str = './checkpoints',
                     iou_pos_thresh: float = 0.6,
                     iou_neg_thresh: float = 0.45,
                     score_threshold: float = 0.3):
    """
    完整训练流程（兼容 train_main.py 的调用）：
      - 构建 DataLoader（使用 collate_fn=lambda x:x）
      - 每 epoch 调用 train_one_epoch，validate_and_save_predictions
      - 保存 epoch ckpt 与 best_model
    返回 history
    """
    os.makedirs(save_dir, exist_ok=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=lambda x: x)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=lambda x: x)
    params = list(model_backbone.parameters()) + list(model_image.parameters()) + list(model_rcnn.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    loss_obj = DetectionLoss()
    best_map = -1.0
    history = {'train_loss': [], 'val_map': []}
    for epoch in range(1, num_epochs+1):
        t0 = time.time()
        train_loss = train_one_epoch(model_backbone, model_image, model_rcnn, train_loader, optimizer, device, epoch, loss_obj, iou_pos_thresh, iou_neg_thresh)
        t1 = time.time()
        eval_res = validate_and_save_predictions(val_loader, model_backbone, model_image, model_rcnn, device, save_dir, classes, score_threshold)
        t2 = time.time()
        print(f"Epoch {epoch} train_loss {train_loss:.4f} val_mAP {eval_res['mAP']:.4f} time_train {t1-t0:.1f}s time_val {t2-t1:.1f}s")
        history['train_loss'].append(train_loss)
        history['val_map'].append(eval_res['mAP'])
        # save checkpoint
        state_dicts = {'backbone': model_backbone.state_dict(), 'image': model_image.state_dict(), 'rcnn': model_rcnn.state_dict()}
        ckpt_path = os.path.join(save_dir, f'ckpt_epoch_{epoch}.pth')
        save_checkpoint(state_dicts, optimizer, epoch, ckpt_path)
        if eval_res['mAP'] > best_map:
            best_map = eval_res['mAP']
            best_path = os.path.join(save_dir, 'best_model.pth')
            save_checkpoint(state_dicts, optimizer, epoch, best_path)
    return history

