# ============================================
# 功能: 加载4D毫米波雷达点云、单目图像及KITTI标签，构建可用于训练的Dataset
# ============================================

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


# -----------------------
# 点云加载函数
# -----------------------
def load_point_cloud(bin_path):
    """
    读取4D毫米波雷达点云文件 (.bin)
    每个点包含:
        [range, azimuth, elevation, doppler, power, x, y, z]
    返回:
        points_full: (N, 8)
        points_xyzp: (N, 5) -> [x, y, z, doppler, power]
    """
    raw = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 8)
    if raw.shape[1] != 8:
        raise ValueError(f"[Error] 点云格式错误: {bin_path}, 期望形状 (N,8)，实际 {raw.shape}")

    # 提取主要通道
    range_, azimuth, elevation, doppler, power, x, y, z = raw.T
    points_xyzp = np.stack([x, y, z, doppler, power], axis=1)
    return points_xyzp, raw


# -----------------------
# 图像与标签加载
# -----------------------
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return np.array(image)


def load_label(label_path):
    """
    读取KITTI风格标签
    """
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            if len(parts) < 15:
                continue
            labels.append({
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': [float(x) for x in parts[4:8]],
                'dimensions': [float(x) for x in parts[8:11]],  # [h, w, l]
                'location': [float(x) for x in parts[11:14]],   # [x, y, z]
                'rotation_y': float(parts[14])
            })
    return labels


# -----------------------
# 标定文件解析
# -----------------------
def load_calibration(calib_path):
    calib = {}
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"标定文件不存在: {calib_path}")

    with open(calib_path, 'r') as f:
        lines = f.readlines()

    raw = {}
    for line in lines:
        if ':' not in line:
            continue
        key, value = line.split(':', 1)
        raw[key] = np.array([float(x) for x in value.strip().split(' ') if x != ''])

    for key in ['P0', 'P1', 'P2', 'P3']:
        if key in raw:
            calib[key] = raw[key].reshape(3, 4)

    if 'R0_rect' in raw:
        R = raw['R0_rect'].reshape(3, 3)
        R_ext = np.eye(4)
        R_ext[:3, :3] = R
        calib['R0_rect'] = R_ext

    if 'Tr_velo_to_cam' in raw:
        T = raw['Tr_velo_to_cam'].reshape(3, 4)
        T_ext = np.eye(4)
        T_ext[:3, :4] = T
        calib['Tr_velo_to_cam'] = T_ext

    return calib


# -----------------------
# 雷达点投影到图像
# -----------------------
def project_lidar_to_image(points_xyz, calib):
    N = points_xyz.shape[0]
    pts_hom = np.hstack((points_xyz, np.ones((N, 1))))
    P2 = calib['P2']
    R0 = calib['R0_rect']
    Tr = calib['Tr_velo_to_cam']
    proj_mat = P2 @ R0[:3, :] @ Tr
    cam_pts = (proj_mat @ pts_hom.T).T
    u = cam_pts[:, 0] / cam_pts[:, 2]
    v = cam_pts[:, 1] / cam_pts[:, 2]
    return np.stack([u, v], axis=1)


# -----------------------
# Dataset定义
# -----------------------
class KITTIDatasetRadar4D(Dataset):
    """
    适配4D毫米波雷达 + 图像 + KITTI标签的PyTorch Dataset
    返回:
        {
            'image': (H,W,3),
            'points': (N,5) -> [x, y, z, doppler, power],
            'raw_points': (N,8) -> [range, azimuth, elevation, doppler, power, x, y, z],
            'calib': dict,
            'labels': list,
            'frame_id': str
        }
    """
    def __init__(self, root_dir, split='training', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        img_dir = os.path.join(root_dir, split, 'image_2')
        self.frame_ids = sorted([
            os.path.splitext(f)[0] for f in os.listdir(img_dir)
            if f.endswith('.png') or f.endswith('.jpg')
        ])

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        frame_id = self.frame_ids[idx]
        img_path = os.path.join(self.root_dir, self.split, 'image_2', f"{frame_id}.png")
        pc_path = os.path.join(self.root_dir, self.split, 'velodyne', f"{frame_id}.bin")
        calib_path = os.path.join(self.root_dir, self.split, 'calib', f"{frame_id}.txt")
        label_path = os.path.join(self.root_dir, self.split, 'label_2', f"{frame_id}.txt")

        image = load_image(img_path)
        points, raw_points = load_point_cloud(pc_path)
        calib = load_calibration(calib_path)
        labels = load_label(label_path)

        sample = {
            'image': image,
            'points': points,
            'raw_points': raw_points,
            'calib': calib,
            'labels': labels,
            'frame_id': frame_id
        }

        if self.transform:
            sample = self.transform(sample)
        return sample
