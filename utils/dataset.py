import os
import random
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import transforms
from configs.config import Config


class CULaneDataset(Dataset):
    def __init__(self, root_dir, phase='train'):
        self.root = root_dir
        self.phase = phase
        self.img_dir = os.path.join(root_dir, "image")
        # 兼容两种标签文件夹名
        if os.path.exists(os.path.join(root_dir, "labels")):
            self.lbl_dir = os.path.join(root_dir, "labels")
        else:
            self.lbl_dir = os.path.join(root_dir, "culane_smooth_annotations")

        self.files = [f for f in os.listdir(self.img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        print(f"[{phase}] Loaded {len(self.files)} images")

        # 颜色增强
        if self.phase == 'train':
            self.color_transform = transforms.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05
            )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. 读取图像
        img_name = self.files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: raise ValueError(f"Bad Image: {img_path}")

        orig_h, orig_w = img.shape[:2]

        # 2. 调整尺寸 (Resize)
        img_resized = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))

        # 3. 读取并处理标签
        base_name = os.path.splitext(img_name)[0]
        lbl_name = base_name + '.lines.txt'
        if not os.path.exists(os.path.join(self.lbl_dir, lbl_name)):
            lbl_name = base_name + '.txt'

        lbl_path = os.path.join(self.lbl_dir, lbl_name)
        lines_params = []

        if os.path.exists(lbl_path):
            with open(lbl_path, 'r') as f:
                for line in f:
                    coords = list(map(float, line.strip().split()))
                    pts = np.array(coords).reshape(-1, 2)
                    if len(pts) < 4: continue

                    xs = pts[:, 0] / orig_w
                    ys = pts[:, 1] / orig_h

                    try:
                        # ==========================================
                        # 【核心】坐标平移与拟合
                        # 坐标系: y_shift = y - 1.0 (范围 -1 ~ 0)
                        # ==========================================
                        ys_shifted = ys - 1.0

                        # 标准3次拟合
                        # 得到的 coeffs 对应: x = p*y'^3 + k*y'^2 + m*y' + b
                        coeffs = np.polyfit(ys_shifted, xs, 3)

                        # 保存 [p, k, m, b]
                        lines_params.append([coeffs[0], coeffs[1], coeffs[2], coeffs[3]])

                    except np.linalg.LinAlgError:
                        continue

        lines_params = np.array(lines_params, dtype=np.float32) if len(lines_params) > 0 else np.zeros((0, 4),
                                                                                                       dtype=np.float32)

        # 4. 几何增强 (Flip)
        if self.phase == 'train' and random.random() > 0.5:
            # 图片翻转
            img_resized = cv2.flip(img_resized, 1)

            # 参数翻转
            if len(lines_params) > 0:
                # 左右翻转逻辑: x' = 1 - x
                # 对于 y_shift 坐标系 (-1 ~ 0)
                # x = p*y^3 + k*y^2 + m*y + b
                # 1 - x = 1 - (p*y^3 + ...) = -p*y^3 - k*y^2 - m*y + (1-b)
                lines_params[:, 0] *= -1  # p -> -p
                lines_params[:, 1] *= -1  # k -> -k
                lines_params[:, 2] *= -1  # m -> -m
                lines_params[:, 3] = 1.0 - lines_params[:, 3]  # b -> 1-b

        # 5. 转 Tensor (核心：确保这里生成的是 img_tensor)
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        if self.phase == 'train':
            img_tensor = self.color_transform(img_tensor)

        # 6. 封装 Target
        target = {}
        if len(lines_params) > 0:
            target["lines"] = torch.from_numpy(lines_params)
            target["labels"] = torch.ones(len(lines_params), dtype=torch.int64)
        else:
            target["lines"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)

        # 【关键检查】必须返回 img_tensor (3, H, W) 和 target (dict)
        return img_tensor, target


# ============================================================
# Collate Fn (注意：必须在类外面，没有缩进)
# ============================================================
def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets