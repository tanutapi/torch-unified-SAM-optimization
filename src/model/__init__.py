from .resnet import *
from .pyramidnet import *
from .wideresnet import *
from .vit import ViT_B_16, ViT_B_32, ViT_L_16, ViT_L_32, ViT_H_14


MODEL_MAP = {
    "resnet18":    ResNet18,
    "resnet34":    ResNet34,
    "resnet50":    ResNet50,
    "resnet101":   ResNet101,
    "resnet152":   ResNet152,
    "wideresnet28": WideResNet28,
    "wideresnet34": WideResNet34,
    "pyramidnet":  PyramidNet,
    # Vision Transformers (torchvision)
    "vit_b_16":   ViT_B_16,
    "vit_b_32":   ViT_B_32,
    "vit_l_16":   ViT_L_16,
    "vit_l_32":   ViT_L_32,
    "vit_h_14":   ViT_H_14,
}