import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size=(3, 3), dilation=(2, 2), padding=(2, 2)
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels, channels, kernel_size=(3, 3), dilation=(4, 4), padding=(4, 4)
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels, channels, kernel_size=(3, 3), dilation=(8, 8), padding=(8, 8)
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        return x + self.block(x)  # Residual connection


class SpectrogramEnhancer(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
        )

        # ---- Bottleneck with temporal context ----
        self.temporal_block = TemporalContextBlock(128)

        # ---- Decoder with skips ----
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, 32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(32 + 32, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        # ---- Encoder ----
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)

        # ---- Bottleneck ----
        bottleneck = self.temporal_block(e3)

        # ---- Decoder + Skip Connections ----
        d3 = self.dec3(bottleneck)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        out = self.dec1(d2)

        return F.relu(out)
