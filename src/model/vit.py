import torch.nn as nn
from torchvision.models import (
    vit_b_16 as _vit_b_16, ViT_B_16_Weights,
    vit_b_32 as _vit_b_32, ViT_B_32_Weights,
    vit_l_16 as _vit_l_16, ViT_L_16_Weights,
    vit_l_32 as _vit_l_32, ViT_L_32_Weights,
    vit_h_14 as _vit_h_14, ViT_H_14_Weights,
)

# (constructor, pretrained_weights, hidden_dim)
_VIT_REGISTRY = {
    "vit_b_16": (_vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1,                   768),
    "vit_b_32": (_vit_b_32, ViT_B_32_Weights.IMAGENET1K_V1,                   768),
    "vit_l_16": (_vit_l_16, ViT_L_16_Weights.IMAGENET1K_V1,                  1024),
    "vit_l_32": (_vit_l_32, ViT_L_32_Weights.IMAGENET1K_V1,                  1024),
    # ViT-H/14 has no IMAGENET1K_V1; SWAG linear weights are the only option
    "vit_h_14": (_vit_h_14, ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1,      1280),
}


def _make_vit(arch: str, num_classes: int, pretrained: bool = False):
    constructor, weights_enum, hidden_dim = _VIT_REGISTRY[arch]
    weights = weights_enum if pretrained else None
    model = constructor(weights=weights)
    # Always replace the head so num_classes is respected
    model.heads.head = nn.Linear(hidden_dim, num_classes)
    return model


def ViT_B_16(num_classes: int = 1000, pretrained: bool = False):
    return _make_vit("vit_b_16", num_classes, pretrained)


def ViT_B_32(num_classes: int = 1000, pretrained: bool = False):
    return _make_vit("vit_b_32", num_classes, pretrained)


def ViT_L_16(num_classes: int = 1000, pretrained: bool = False):
    return _make_vit("vit_l_16", num_classes, pretrained)


def ViT_L_32(num_classes: int = 1000, pretrained: bool = False):
    return _make_vit("vit_l_32", num_classes, pretrained)


def ViT_H_14(num_classes: int = 1000, pretrained: bool = False):
    return _make_vit("vit_h_14", num_classes, pretrained)
