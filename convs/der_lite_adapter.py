"""
Multi-Scale Task Adapters for DER-Lite.

V5 (MultiScaleTaskAdapter): lightweight, ~64K per adapter.
V7 (DeeperTaskAdapter): deeper per-scale processing + residual fusion,
                         ~200K per adapter (3x params, richer features).
"""

import torch
import torch.nn as nn


class MultiScaleTaskAdapter(nn.Module):
    def __init__(self, in_channels_list=(16, 32, 64), common_dim=128,
                 out_dim=128):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.common_dim = common_dim
        self.out_dim = out_dim

        self.projectors = nn.ModuleList([
            self._make_proj(in_c) for in_c in in_channels_list
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)

        concat_dim = common_dim * len(in_channels_list)
        self.final = nn.Sequential(
            nn.Linear(concat_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _make_proj(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, self.common_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.common_dim),
            nn.ReLU(inplace=True),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, fmaps):
        features = []
        for fmap, proj in zip(fmaps, self.projectors):
            feat = self.pool(proj(fmap)).view(fmap.size(0), -1)
            features.append(feat)
        fused = torch.cat(features, dim=1)
        return self.final(fused)


class DeeperTaskAdapter(nn.Module):
    """
    V7 adapter: deeper per-scale conv processing + residual fusion MLP.

    Per-scale (for each fmap):
      1x1 conv: in_c -> hidden
      BN + ReLU
      1x1 conv: hidden -> out_per_scale
      Pool + Flatten

    Fusion:
      Concat all scales
      Linear -> hidden  (with residual from concat projection)
      BN + ReLU
      Linear -> out_dim

    Target ~200K params per adapter (3x v5), still ~11x less than pDER.
    """

    def __init__(self, in_channels_list=(16, 32, 64), common_dim=128,
                 out_dim=128, hidden_scale=64):
        super().__init__()

        self.in_channels_list = in_channels_list
        self.common_dim = common_dim
        self.out_dim = out_dim
        self.hidden_scale = hidden_scale

        self.projectors = nn.ModuleList([
            self._make_deep_proj(in_c) for in_c in in_channels_list
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)

        concat_dim = hidden_scale * len(in_channels_list)
        self.fusion_layer1 = nn.Linear(concat_dim, common_dim, bias=False)
        self.fusion_bn = nn.BatchNorm1d(common_dim)
        self.fusion_layer2 = nn.Linear(common_dim, out_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self._initialize_weights()

    def _make_deep_proj(self, in_c):
        return nn.Sequential(
            nn.Conv2d(in_c, self.common_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.common_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.common_dim, self.hidden_scale, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(self.hidden_scale),
            nn.ReLU(inplace=True),
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, fmaps):
        features = []
        for fmap, proj in zip(fmaps, self.projectors):
            feat = self.pool(proj(fmap)).view(fmap.size(0), -1)
            features.append(feat)

        fused = torch.cat(features, dim=1)

        h = self.fusion_layer1(fused)
        h = self.relu(self.fusion_bn(h))
        out = self.fusion_layer2(h)
        return out
