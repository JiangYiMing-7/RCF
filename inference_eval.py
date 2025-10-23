import torch
import numpy as np

def inference(model, points, image, calib, score_threshold=0.5):
    """
    Perform inference to get 3D bounding box predictions.
    points: (1, N, 4) tensor (x,y,z,reflectance)
    image: (1, 3, H, W) tensor
    calib: calibration dict
    Returns:
      boxes: list of (box, score), where box=[x,y,z,l,w,h,yaw]
    """
    model.eval()
    with torch.no_grad():
        cls_logits, bbox_preds = model(points, image, calib)  # (1,N,1), (1,N,7)
        scores = torch.sigmoid(cls_logits).squeeze(0).squeeze(-1)  # (N,)
        mask = scores > score_threshold
        selected_points = points[0, mask, :3]    # (M,3)
        selected_scores = scores[mask]            # (M,)
        selected_preds = bbox_preds[0, mask, :]   # (M,7)
        boxes = []
        # Decode boxes
        for i in range(selected_points.shape[0]):
            px, py, pz = selected_points[i].cpu().numpy()
            dx, dy, dz, h, w, l, ry = selected_preds[i].cpu().numpy()
            cx = px + dx
            cy = py + dy
            cz = pz + dz
            boxes.append(([cx, cy, cz, l, w, h, ry], selected_scores[i].item()))
        return boxes

def compute_iou_3d(box1, box2):
    """
    计算两个轴对齐box的 3D IoU（忽略朝向）
    box格式为：[x, y, z, l, w, h, ry]
    """
    x1, y1, z1, l1, w1, h1, _ = box1
    x2, y2, z2, l2, w2, h2, _ = box2

    x1_min = x1 - l1/2; x1_max = x1 + l1/2
    y1_min = y1 - w1/2; y1_max = y1 + w1/2
    z1_min = z1 - h1/2; z1_max = z1 + h1/2
    x2_min = x2 - l2/2; x2_max = x2 + l2/2
    y2_min = y2 - w2/2; y2_max = y2 + w2/2
    z2_min = z2 - h2/2; z2_max = z2 + h2/2

    dx = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    dy = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    dz = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    inter_vol = dx * dy * dz
    vol1 = l1 * w1 * h1
    vol2 = l2 * w2 * h2
    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-6)
    return iou

def evaluate_map(predictions, ground_truths, iou_threshold=0.5):
    """
    计算一组预测结果与真实值之间的平均准确率（mAP）
    predictions：由（[x, y, z, l, w, h, ry]，score）组成的列表
    ground_truths：由 [x, y, z, l, w, h, ry] 组成的列表
    """
    if len(ground_truths) == 0:
        return 0.0
    # Sort by score desc
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)
    tp = 0
    fp = 0
    matched = set()
    for box, score in predictions:
        found_match = False
        for i, gt in enumerate(ground_truths):
            if i in matched:
                continue
            if compute_iou_3d(box, gt) >= iou_threshold:
                tp += 1
                matched.add(i)
                found_match = True
                break
        if not found_match:
            fp += 1
    fn = len(ground_truths) - tp
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    ap = precision if tp > 0 else 0.0
    return ap
