#!/usr/bin/env python3
"""Build and preprocess inputs for ImageNet-pretrained SqueezeNet 1.1."""

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

ROOT = Path(__file__).resolve().parents[1]
IMAGE_SIZE = 64
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(name):
    if name != "SqueezeNet 1.1":
        raise ValueError(f"Unknown architecture: {name}")
    model = squeezenet1_1(weights=SqueezeNet1_1_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=1)
    return model


def preprocess(images):
    images = F.interpolate(images, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    return (images.expand(-1, 3, -1, -1) - MEAN) / STD
