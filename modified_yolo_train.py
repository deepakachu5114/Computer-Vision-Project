import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import os
from ultralytics import YOLO
from ultralytics.nn.modules import Conv as UltralyticsConv
from ultralytics.nn.modules import C2f as UltralyticsC2f
from ultralytics.nn.modules import Concat, Bottleneck, SPPF
from ultralytics.nn.modules.conv import autopad

print("Defining YOLOv11-inspired custom modules...")

class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        reduced_channels = max(channels // reduction_ratio, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.SiLU(),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class CBAMC3(UltralyticsC2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        if isinstance(n, bool):
            n_val = 1 if n else 0
        else:
            n_val = int(n)
        shortcut_val = bool(shortcut)
        super().__init__(c1, c2, n_val, shortcut_val, g, e)
        self.se = SEBlock(c2)
        self.channel_attn = ChannelAttention(c2)
        self.spatial_attn = SpatialAttention(kernel_size=7)

    def forward(self, x):
        y = super().forward(x)
        y = y * self.channel_attn(y)
        y = y * self.spatial_attn(y)
        y = self.se(y)
        return y

class ELAN(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = UltralyticsConv(c1, c_, 1, 1)
        n_int = max(1, int(n))
        self.m1 = nn.Sequential(*[UltralyticsConv(c_, c_, 3, 1) for _ in range(n_int)])
        self.m2 = nn.Sequential(*[UltralyticsConv(c_, c_, 3, 1) for _ in range(n_int)])
        self.cv2 = UltralyticsConv(c_ * (2 + n_int), c2, 1, 1)
        self.se = SEBlock(c2)

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m1(x1)
        y2 = self.m2(y1)
        return self.se(self.cv2(torch.cat([x1, y1, y2], dim=1)))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn_input = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn_input)
        return self.sigmoid(attn)

class RepConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, p=None, g=1):
        super().__init__()
        p = p if p is not None else autopad(k, p)
        self.conv = UltralyticsConv(c1, c2, k, s, p, g)
        self.conv1x1 = UltralyticsConv(c1, c2, 1, s, autopad(1, None), g)
        self.bn_identity = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None

    def forward(self, x):
        result = self.conv(x) + self.conv1x1(x)
        if self.bn_identity is not None:
            result += self.bn_identity(x)
        return result

    def fuse(self):
        if self.bn_identity is not None:
            pass
        return self

class SPD(nn.Module):
    def __init__(self, c1, c2, n_kernels=3):
        super().__init__()
        c_ = c2 // n_kernels
        self.convs = nn.ModuleList([
            UltralyticsConv(c1, c_, 3, 1, p=1*i, d=1*i) 
            for i in range(1, n_kernels+1)
        ])
        self.fusion = UltralyticsConv(c_ * n_kernels, c2, 1, 1)
        self.se = SEBlock(c2)

    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        y = self.fusion(torch.cat(outputs, dim=1))
        return self.se(y)

class BiFormerBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = SEBlock(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.ffn = nn.Sequential(
            UltralyticsConv(channels, channels * reduction, 1, 1),
            nn.SiLU(),
            UltralyticsConv(channels * reduction, channels, 1, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class SimFusion(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.cv1 = UltralyticsConv(c1, c2, 1, 1)
        self.cv2 = UltralyticsConv(c2, c2, 3, 1)
        self.se = SEBlock(c2)

    def forward(self, x):
        x = self.cv1(x)
        x = self.cv2(x)
        x = self.se(x)
        return x

print("Custom modules defined.")

import sys
from ultralytics.nn import tasks

modules_to_register = {
    'SEBlock': SEBlock,
    'CBAMC3': CBAMC3,
    'ELAN': ELAN,
    'ChannelAttention': ChannelAttention,
    'SpatialAttention': SpatialAttention,
    'RepConv': RepConv,
    'SPD': SPD,
    'BiFormerBlock': BiFormerBlock,
    'SimFusion': SimFusion
}

for name, module in modules_to_register.items():
    sys.modules[f'ultralytics.nn.modules.{name}'] = module
    setattr(tasks, name, module)

print("Custom modules injected into ultralytics namespace.")

print("Creating data.yaml...")
data_yaml_content = {
    'path': '/kaggle/working/D Fire Dataset',
    'train': 'data/train/images',
    'val': 'data/val/images',
    'test': 'data/test/images',
    'names': ['smoke', 'fire'],
    'nc': 2
}
os.makedirs('/kaggle/working/', exist_ok=True)
data_yaml_path = '/kaggle/working/data.yaml'
with open(data_yaml_path, 'w') as f:
    yaml.dump(data_yaml_content, f, default_flow_style=False)

print(f"data.yaml created at {data_yaml_path}")
print(yaml.dump(data_yaml_content))
