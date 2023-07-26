# U-net for tumour segmentation

import torch
import torch.nn as nn

class tumour_unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(tumour_unet, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.down1 = tumour_unet(in_channels, 64)
        self.down2 = tumour_unet(64, 128)
        self.down3 = tumour_unet(128, 256)
        self.down4 = tumour_unet(256, 512)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.up3 = tumour_unet(512 + 256, 256)
        self.up2 = tumour_unet(256 + 128, 128)
        self.up1 = tumour_unet(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.maxpool(x1)
        x3 = self.down2(x2)
        x4 = self.maxpool(x3)
        x5 = self.down3(x4)
        x6 = self.maxpool(x5)
        x7 = self.down4(x6)
        x = self.upsample(x7)

        x = torch.cat([x, x5], dim=1)
        x = self.up3(x)
        x = self.upsample(x)

        x = torch.cat([x, x3], dim=1)
        x = self.up2(x)
        x = self.upsample(x)

        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        
        return self.final_conv(x)