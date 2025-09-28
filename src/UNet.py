import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = (3, 3), padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = (3, 3), padding = 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.rel = nn.ReLU()

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        return self.rel(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.up(x))
        return x

class UNet(nn.Module):
    def __init__(self, num_classes, in_channels = 3, encoder_blocks = 4, multiplicity = 64, pool_stride = 2):
        super(UNet, self).__init__()
        self.initial_conv = DoubleConv(in_channels, multiplicity)
        self.encoder_blocks = encoder_blocks
        self.multiplicity = multiplicity
        self.pool_stride = pool_stride
        self.num_classes = num_classes

        self.maxpool = nn.MaxPool2d(2, stride = pool_stride)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.res = nn.ModuleList()
        self.final_conv = nn.Conv2d(multiplicity, num_classes, kernel_size = 1)

        skip_channels = [multiplicity]
        kernels = multiplicity
        for _ in range(self.encoder_blocks):
            self.downs.append(DoubleConv(kernels, kernels * 2))
            kernels *= 2
            skip_channels.append(kernels)

        for i in range(self.encoder_blocks):
            self.ups.append(UpConv(kernels, kernels // 2))
            self.res.append(DoubleConv(kernels // 2 + skip_channels[-(i + 2)], kernels // 2))
            kernels //= 2

    def forward(self, x):
        residuals = []
        x = self.initial_conv(x)
        residuals.append(x)

        for i, conv in enumerate(self.downs):
            x = self.maxpool(x)
            x = conv(x)
            residuals.append(x)

        for up, res_block, skip in zip(self.ups, self.res, reversed(residuals[:-1])):
            x = up(x)
            x = torch.cat([skip, x], dim = 1)
            x = res_block(x)

        x = self.final_conv(x)
        return x
