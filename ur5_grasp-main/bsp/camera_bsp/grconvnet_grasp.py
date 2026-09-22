"""Self-contained GR-ConvNet (RGB-D) grasp candidate generator.

The network definition is copied from the LGD repository so this module has no
runtime dependency on it.  Weights are loaded from a plain ``state_dict``
(``weights/grconvnet/cornell_rgbd_ch32.pt``).

The network was trained on 224x224 crops centred on the object, with the
channel order depth(1) + RGB(3); depth is ``clip(d - mean, -1, 1)`` and RGB is
``/255 - mean``.  Width is normalised by ``input_size / 2`` (112), which the
original inference helper multiplied by 150 by mistake.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x_in):
        x = self.bn1(self.conv1(x_in))
        x = F.relu(x)
        x = self.bn2(self.conv2(x))
        return x + x_in


class GenerativeResnet(nn.Module):
    """GR-ConvNet3 with an optional dropout head, identical to the training code."""

    def __init__(self, input_channels=4, output_channels=1, channel_size=32,
                 dropout=False, prob=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, channel_size, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(channel_size)

        self.conv2 = nn.Conv2d(channel_size, channel_size * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(channel_size * 2)

        self.conv3 = nn.Conv2d(channel_size * 2, channel_size * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(channel_size * 4)

        self.res1 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res2 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res3 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res4 = ResidualBlock(channel_size * 4, channel_size * 4)
        self.res5 = ResidualBlock(channel_size * 4, channel_size * 4)

        self.conv4 = nn.ConvTranspose2d(channel_size * 4, channel_size * 2, kernel_size=4,
                                        stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(channel_size * 2)

        self.conv5 = nn.ConvTranspose2d(channel_size * 2, channel_size, kernel_size=4,
                                        stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(channel_size)

        self.conv6 = nn.ConvTranspose2d(channel_size, channel_size, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(in_channels=channel_size, out_channels=output_channels, kernel_size=2)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(self, x_in):
        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)

        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(x))
            cos_output = self.cos_output(self.dropout_cos(x))
            sin_output = self.sin_output(self.dropout_sin(x))
            width_output = self.width_output(self.dropout_wid(x))
        else:
            pos_output = self.pos_output(x)
            cos_output = self.cos_output(x)
            sin_output = self.sin_output(x)
            width_output = self.width_output(x)

        return pos_output, cos_output, sin_output, width_output


DEFAULT_WEIGHTS = Path(__file__).resolve().parents[3] / "weights" / "grconvnet" / "cornell_rgbd_ch32.pt"


class GrConvNetGrasp:
    """Resident GR-ConvNet grasp proposal on a mask/box crop."""

    def __init__(self, weights=DEFAULT_WEIGHTS, input_size=224, padding=1.3,
                 channel_size=32, dropout=True, prob=0.1, device="cuda",
                 min_distance=20, threshold=0.2, mask_weight=0.7, thickness_gate=0.4):
        self.weights = str(weights)
        self.input_size = int(input_size)
        self.padding = float(padding)
        self.min_distance = int(min_distance)
        self.threshold = float(threshold)
        self.mask_weight = float(mask_weight)
        self.thickness_gate = float(thickness_gate)
        self.width_scale = self.input_size / 2.0

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        self.model = GenerativeResnet(
            input_channels=4, output_channels=1, channel_size=channel_size,
            dropout=dropout, prob=prob,
        )
        state_dict = torch.load(self.weights, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _build_input(self, rgb_bgr, depth_raw, depth_scale, box):
        height, width = rgb_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        side = max(x2 - x1, y2 - y1) * self.padding
        side = max(side, 16.0)
        side = min(side, float(min(height, width)))
        left = min(max(center_x - side / 2.0, 0.0), width - side)
        top = min(max(center_y - side / 2.0, 0.0), height - side)
        left_i, top_i, side_i = int(round(left)), int(round(top)), int(round(side))

        rgb_crop = rgb_bgr[top_i:top_i + side_i, left_i:left_i + side_i]
        depth_crop = depth_raw[top_i:top_i + side_i, left_i:left_i + side_i]
        if rgb_crop.size == 0 or depth_crop.size == 0:
            return None, "empty crop"

        rgb_resized = cv2.resize(rgb_crop, (self.input_size, self.input_size),
                                 interpolation=cv2.INTER_AREA)
        depth_resized = cv2.resize(depth_crop.astype(np.float32),
                                   (self.input_size, self.input_size),
                                   interpolation=cv2.INTER_NEAREST)

        depth_m = depth_resized * float(depth_scale)
        depth_norm = np.clip(depth_m - depth_m.mean(), -1.0, 1.0)

        rgb = cv2.cvtColor(rgb_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = rgb - rgb.mean()
        rgb = rgb.transpose(2, 0, 1)

        stacked = np.concatenate([depth_norm[None, :, :], rgb], axis=0)
        tensor = torch.from_numpy(stacked).unsqueeze(0).to(self.device).float()
        return tensor, (left_i, top_i, side_i)

    def _detect(self, q_img, ang_img, width_img, no_grasps, search=None, valid_mask=None):
        search = q_img if search is None else search
        if valid_mask is not None:
            search = np.where(valid_mask, search, -1e9)
        k = self.min_distance * 2 + 1
        dilated = cv2.dilate(search, np.ones((k, k), np.uint8))
        peaks = (search >= dilated) & (search > self.threshold)
        ys, xs = np.nonzero(peaks)
        if len(ys) == 0:
            iy, ix = np.unravel_index(int(np.argmax(search)), search.shape)
            ys, xs = np.array([iy]), np.array([ix])
        scores = q_img[ys, xs]
        order = np.argsort(scores)[::-1][:max(1, int(no_grasps))]
        results = []
        for idx in order:
            iy, ix = int(ys[idx]), int(xs[idx])
            results.append({
                "center_224": (ix, iy),
                "angle_rad": float(ang_img[iy, ix]),
                "width_px_224": float(width_img[iy, ix]),
                "q": float(q_img[iy, ix]),
            })
        return results

    @torch.no_grad()
    def propose_all(self, rgb_bgr, depth_raw, depth_scale, box, no_grasps=1, mask=None):
        built = self._build_input(rgb_bgr, depth_raw, depth_scale, box)
        if isinstance(built[0], str):
            return [], built[0]
        tensor, (left, top, side) = built

        valid_mask = None
        thickness_norm = None
        if mask is not None:
            mask_u8 = np.asarray(mask, dtype=np.uint8)
            mask_crop = mask_u8[top:top + side, left:left + side]
            if mask_crop.size:
                valid_mask = cv2.resize(
                    mask_crop, (self.input_size, self.input_size),
                    interpolation=cv2.INTER_NEAREST) > 0
                distance = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
                distance_crop = distance[top:top + side, left:left + side]
                distance_resized = cv2.resize(
                    distance_crop, (self.input_size, self.input_size),
                    interpolation=cv2.INTER_LINEAR)
                peak = float(distance_resized.max())
                if peak > 0:
                    thickness_norm = distance_resized / peak

        pos, cos, sin, width = self.model(tensor)
        q_img = pos.cpu().numpy().squeeze()
        ang_img = (torch.atan2(sin, cos) / 2.0).cpu().numpy().squeeze()
        width_img = width.cpu().numpy().squeeze() * self.width_scale

        q_img = cv2.GaussianBlur(q_img, (0, 0), 2.0)
        ang_img = cv2.GaussianBlur(ang_img, (0, 0), 2.0)
        width_img = cv2.GaussianBlur(width_img, (0, 0), 1.0)

        search = q_img
        if thickness_norm is not None:
            if self.thickness_gate > 0:
                gate = thickness_norm >= self.thickness_gate
                valid_mask = gate if valid_mask is None else (valid_mask & gate)
            if self.mask_weight > 0:
                weight = (1.0 - self.mask_weight) + self.mask_weight * thickness_norm
                search = q_img * weight

        scale = side / float(self.input_size)
        proposals = []
        for item in self._detect(q_img, ang_img, width_img, no_grasps, search, valid_mask):
            cx224, cy224 = item["center_224"]
            u = int(round(cx224 * scale + left))
            v = int(round(cy224 * scale + top))
            proposals.append({
                "center": (u, v),
                "angle_rad": item["angle_rad"],
                "angle_deg": float(np.degrees(item["angle_rad"])),
                "width_px": float(item["width_px_224"] * scale),
                "q": item["q"],
                "crop": (left, top, side),
            })
        return proposals, None

    def propose(self, rgb_bgr, depth_raw, depth_scale, box, no_grasps=1, mask=None):
        proposals, reason = self.propose_all(
            rgb_bgr, depth_raw, depth_scale, box, no_grasps, mask=mask)
        if not proposals:
            return None, reason or "no grasp above threshold"
        return proposals[0], None
