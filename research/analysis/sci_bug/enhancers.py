"""Own re-implementations of SCI (CVPR'22) and Zero-DCE++ (TPAMI'21) inference networks.

Architectures transcribed from the official repos (vis-opt-group/SCI, Li-Chongyi/Zero-DCE_extension);
only the released weight tensors are loaded (torch.load(weights_only=True)), no third-party code runs.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SCIEnhance(nn.Module):
    def __init__(self, layers=1, channels=3):
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(3, channels, 3, 1, 1), nn.ReLU())
        self.conv = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.ModuleList([self.conv for _ in range(layers)])  # shared module, as in the official code
        self.out_conv = nn.Sequential(nn.Conv2d(channels, 3, 3, 1, 1), nn.Sigmoid())

    def forward(self, x):
        fea = self.in_conv(x)
        for conv in self.blocks:
            fea = fea + conv(fea)
        fea = self.out_conv(fea)
        return torch.clamp(fea + x, 0.0001, 1)


class SCI(nn.Module):
    """Returns (illumination i, enhanced r = clamp(x / i, 0, 1)). Since x <= i <= 1, r >= x pixel-wise."""

    def __init__(self, weight_path):
        super().__init__()
        self.enhance = _SCIEnhance()
        sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        own = self.state_dict()
        matched = {k: v for k, v in sd.items() if k in own}
        assert len(matched) == len(own), f"SCI weight mismatch: {set(own) - set(matched)}"
        self.load_state_dict(matched)

    def forward(self, x):
        i = self.enhance(x)
        return i, torch.clamp(x / i, 0, 1)


class _CSDN(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.depth_conv = nn.Conv2d(cin, cin, 3, 1, 1, groups=cin)
        self.point_conv = nn.Conv2d(cin, cout, 1, 1, 0)

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class ZeroDCEpp(nn.Module):
    def __init__(self, weight_path, scale_factor=12):
        super().__init__()
        f = 32
        self.scale_factor = scale_factor
        self.e_conv1, self.e_conv2, self.e_conv3, self.e_conv4 = _CSDN(3, f), _CSDN(f, f), _CSDN(f, f), _CSDN(f, f)
        self.e_conv5, self.e_conv6, self.e_conv7 = _CSDN(2 * f, f), _CSDN(2 * f, f), _CSDN(2 * f, 3)
        sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        self.load_state_dict(sd, strict=True)

    def forward(self, x):
        H, W = x.shape[-2:]
        s = self.scale_factor
        ph, pw = (-H) % s, (-W) % s  # pad (not crop) so geometry and boxes are preserved exactly
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate")
        xd = F.interpolate(xp, scale_factor=1 / s, mode="bilinear") if s != 1 else xp
        r = F.relu
        x1 = r(self.e_conv1(xd)); x2 = r(self.e_conv2(x1)); x3 = r(self.e_conv3(x2)); x4 = r(self.e_conv4(x3))
        x5 = r(self.e_conv5(torch.cat([x3, x4], 1))); x6 = r(self.e_conv6(torch.cat([x2, x5], 1)))
        xr = torch.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        if s != 1:
            xr = F.interpolate(xr, size=xp.shape[-2:], mode="bilinear", align_corners=True)  # == UpsamplingBilinear2d
        y = xp
        for _ in range(8):
            y = y + xr * (y * y - y)
        return y[..., :H, :W].clamp(0, 1)


def to_tensor(rgb_uint8):
    return torch.from_numpy(np.ascontiguousarray(rgb_uint8)).float().div(255).permute(2, 0, 1)[None]


def to_uint8(t):
    return (t[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255 + 0.5).astype(np.uint8)


def gamma_match(rgb_uint8, target_mean):
    """Brightness-only control: global gamma chosen so the mean luminance matches target_mean (0-255)."""
    x = rgb_uint8.astype(np.float32) / 255
    lo, hi = 0.05, 1.0
    for _ in range(30):  # bisection on gamma (smaller gamma = brighter)
        g = (lo + hi) / 2
        if (x ** g).mean() * 255 < target_mean:
            hi = g
        else:
            lo = g
    return (np.clip(x ** ((lo + hi) / 2), 0, 1) * 255 + 0.5).astype(np.uint8)
