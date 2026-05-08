"""
Multi-Scale Task Adapter for DER-Lite.

Processes all backbone feature map scales to produce richer task-specific
features without training the backbone. Parameter-efficient via 1x1 conv
projections.

Automatically adapts to any number of fmaps (3 for ResNet32, 4 for ResNet18).
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

        # Dynamically create one projector per scale
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
