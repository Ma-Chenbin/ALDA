# Chenbin Ma July 2021
# This is the implementation of CycleGAN in PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.5):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.dropout(x)
        return x


class UnetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_downs):
        super(UnetGenerator, self).__init__()
        self.num_downs = num_downs
        prev_channels = in_channels

        # Down part
        self.down_blocks = nn.ModuleList()
        for i in range(num_downs):
            self.down_blocks.append(ConvBlock(prev_channels, 2 ** i))
            prev_channels = 2 ** i

        # Up part
        self.up_blocks = nn.ModuleList()
        for i in range(num_downs - 1, -1, -1):
            self.up_blocks.append(UpBlock(prev_channels, prev_channels // 2))
            prev_channels = prev_channels // 2

        self.last_conv = nn.Conv2d(prev_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        for down_block in self.down_blocks:
            x = down_block(x)

        for up_block in self.up_blocks:
            x = up_block(x)

        x = self.last_conv(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels, dropout_rate=0)
        self.conv_block2 = ConvBlock(out_channels, out_channels, dropout_rate=0)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return x


class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CycleGAN, self).__init__()
        self.G = UnetGenerator(in_channels, out_channels, num_downs=9)
        self.D_X = Discriminator(in_channels=out_channels)
        self.D_Y = Discriminator(in_channels=in_channels)

    def forward(self, x):
        fake_y = self.G(x)
        return fake_y

    def adversarial_loss(self, output, target, device):
        if device == 'cpu':
            loss = F.binary_cross_entropy(output, target)
        else:
            loss = F.binary_cross_entropy(output, target, reduction='sum')
            loss = loss / output.size(0)
        return loss

    def gradient_penalty(self, y_real, y_fake):
        batch_size = y_real.size(0)
        alpha = torch.empty(batch_size, 1, 1, 1).uniform_(0, 1).to(y_real.device)
        interpolates = (alpha * y_real + ((1 - alpha) * y_fake)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(d_interpolates.size()).to(y_real.device),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        return gradient_penalty

    def cycle_loss(self, y_real, y_fake):
        loss = F.l1_loss(y_real, y_fake)
        return loss

    def identity_loss(self, y_real, y_iden):
        loss = F.l1_loss(y_real, y_iden)
        return loss

    def forward(self, x, y, device):
        fake_y = self.G(x)
        D_X = self.D_X(fake_y)
        D_Y = self.D_Y(y)

        g_loss = self.adversarial_loss(D_X, torch.ones(D_X.size()).to(device), device) + \
                 self.adversarial_loss(D_Y, torch.ones(D_Y.size()).to(device), device) + \
                 self.cycle_loss(fake_y, x) + self.identity_loss(y, fake_y)

        return g_loss


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        output = self.main(x)
        return output

