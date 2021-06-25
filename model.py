# coding: utf-8

import torch
from torch import nn
import torch.nn.functional as F
from unet_parts import *


class Generator(nn.Module):
    def __init__(self, bilinear=True):
        super(Generator, self).__init__()
        self.bilinear = bilinear

        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 1)
        self.emb = nn.Linear(7, 512)

    def forward(self, x, angle):        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        angle = self.emb(angle)
        angle = angle.view(angle.shape[0], angle.shape[1], 1, 1)
        x5 = x5 * angle
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.vgg = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, padding=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=2),
        )

    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
