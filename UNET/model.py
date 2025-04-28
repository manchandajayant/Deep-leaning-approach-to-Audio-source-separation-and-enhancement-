import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(
                in_c, out_c, kernel_size=5, stride=2, padding=2, output_padding=1
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class UNetVocalSeparator(nn.Module):
    def __init__(self, base_channels=16):
        super().__init__()
        # ---- Encoder ----
        self.down1 = DownBlock(1, base_channels)
        self.down2 = DownBlock(base_channels, base_channels * 2)
        self.down3 = DownBlock(base_channels * 2, base_channels * 4)
        self.down4 = DownBlock(base_channels * 4, base_channels * 8)
        self.down5 = DownBlock(base_channels * 8, base_channels * 16)
        self.down6 = DownBlock(base_channels * 16, base_channels * 32)

        # ---- Decoder ----
        self.dec6 = UpBlock(base_channels * 32, base_channels * 16, dropout=True)
        self.dec5 = UpBlock(base_channels * 32, base_channels * 8, dropout=True)
        self.dec4 = UpBlock(base_channels * 16, base_channels * 4, dropout=True)

        self.dec3 = UpBlock(base_channels * 8, base_channels * 2, dropout=False)
        self.dec2 = UpBlock(base_channels * 4, base_channels, dropout=False)

        # final upconv → mask
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                base_channels * 2,
                1,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.down1(x)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        e4 = self.down4(e3)
        e5 = self.down5(e4)
        e6 = self.down6(e5)

        d6 = self.dec6(e6)
        d5 = self.dec5(torch.cat([d6, e5], dim=1))
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        mask = self.dec1(torch.cat([d2, e1], dim=1))

        return mask
