import os
import random
import math
from collections import Counter
from typing import Callable, Optional, Union, Any
import xml.etree.ElementTree as ET

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
from albumentations.pytorch import ToTensorV2

!pip install torchinfo
from torchinfo import summary

from tqdm import tqdm
!pip install segment_anything
from segment_anything import sam_model_registry, SamPredictor

### U-Net Calass

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_op(x)


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        down = self.conv(x)
        pooling = self.pool(down)
        return down, pooling



class UpSample(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout_p=0.3):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels=out_channels + skip_channels, out_channels=out_channels)
        #first extract features,then randomly zero out some of the channels,
        #and then pass the result further through the network.
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Если размеры не совпадают, можно подрезать или расширить skip-соединение
        if x1.size() != x2.size():
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Encoder: extract features and store them for skip connections
        # down_i are feature maps BEFORE pooling (used later for skip connections)
        # p_i are pooled feature maps passed deeper into the network

        self.down_conv_1 = DownSample(in_channels, 64)
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)
        self.down_conv_4 = DownSample(256, 512)
        
        #Bottleneck: process deepest features
        # The deepest part of the network capturing high-level features

        self.bottleneck = DoubleConv(512, 1024)

        # Decoder: apply upsampling and concatenate corresponding encoder features
        # Skip connections are implemented here:
        # encoder feature maps (down_i) are concatenated with
        # upsampled decoder feature maps inside the UpSample module

        self.up_conv_1 = UpSample(1024, 512, 512)
        self.up_conv_2 = UpSample(512, 256, 256)
        self.up_conv_3 = UpSample(256, 128, 128)
        self.up_conv_4 = UpSample(128, 64, 64)

        #Final layer
        # Maps features to the required number of output channels (classes)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):

        down_1, p1 = self.down_conv_1(x)
        down_2, p2 = self.down_conv_2(p1)
        down_3, p3 = self.down_conv_3(p2)
        down_4, p4 = self.down_conv_4(p3)

        bottleneck = self.bottleneck(p4)

        up_1 = self.up_conv_1(bottleneck, down_4)
        up_2 = self.up_conv_2(up_1, down_3)
        up_3 = self.up_conv_3(up_2, down_2)
        up_4 = self.up_conv_4(up_3, down_1)

        out = self.out(up_4)
        return out



                           