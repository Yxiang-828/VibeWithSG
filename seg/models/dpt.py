"""DINOv2 + lightweight DPT-style segmentation model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class DinoSpec:
    hub_name: str
    embed_dim: int
    layers: tuple[int, ...]


DINO_SPECS = {
    "dinov2_vits14": DinoSpec("dinov2_vits14", 384, (2, 5, 8, 11)),
    "dinov2_vitb14": DinoSpec("dinov2_vitb14", 768, (2, 5, 8, 11)),
    "dinov2_vitl14": DinoSpec("dinov2_vitl14", 1024, (5, 11, 17, 23)),
    "dinov2_vitl14_reg": DinoSpec("dinov2_vitl14_reg", 1024, (5, 11, 17, 23)),
}


class ResidualConvUnit(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class FeatureFusionBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual = ResidualConvUnit(channels)
        self.out = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        if skip is not None:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = x + skip
        return self.out(self.residual(x))


class DPTDecoder(nn.Module):
    """Fuse four ViT block outputs into full-resolution class logits.

    DINOv2 exposes same-resolution token grids for each selected block. This
    keeps the DPT top-down fusion pattern while using layer depth as the feature
    hierarchy.
    """

    def __init__(self, in_channels: int, num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.ModuleList(
            [
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                for _ in range(4)
            ]
        )
        self.fuse4 = FeatureFusionBlock(hidden_dim)
        self.fuse3 = FeatureFusionBlock(hidden_dim)
        self.fuse2 = FeatureFusionBlock(hidden_dim)
        self.fuse1 = FeatureFusionBlock(hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=1),
        )

    def forward(self, features: tuple[torch.Tensor, ...], output_size: tuple[int, int]) -> torch.Tensor:
        p1, p2, p3, p4 = [proj(feat) for proj, feat in zip(self.proj, features)]
        x = self.fuse4(p4)
        x = self.fuse3(x, p3)
        x = self.fuse2(x, p2)
        x = self.fuse1(x, p1)
        logits = self.head(x)
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)


class DinoDPTSegmentation(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        freeze_backbone: bool = True,
        decoder_dim: int = 256,
    ) -> None:
        super().__init__()
        if backbone_name not in DINO_SPECS:
            raise ValueError(f"Unsupported DINOv2 backbone: {backbone_name}")
        self.spec = DINO_SPECS[backbone_name]
        self.backbone = torch.hub.load("facebookresearch/dinov2", self.spec.hub_name)
        self.freeze_backbone = freeze_backbone
        self.decoder = DPTDecoder(self.spec.embed_dim, num_classes, hidden_dim=decoder_dim)

        if freeze_backbone:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output_size = x.shape[-2:]
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.backbone.get_intermediate_layers(
                    x,
                    n=self.spec.layers,
                    reshape=True,
                    return_class_token=False,
                    norm=True,
                )
        else:
            features = self.backbone.get_intermediate_layers(
                x,
                n=self.spec.layers,
                reshape=True,
                return_class_token=False,
                norm=True,
            )
        return self.decoder(features, output_size)

