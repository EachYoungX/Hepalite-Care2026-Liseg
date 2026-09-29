import torch
import torch.nn as nn
import torch.nn.functional as F


class AddCoords(nn.Module):
    """Append normalized spatial coordinate channels to a feature tensor."""

    def __init__(self, signal_threshold=1e-4):
        super().__init__()
        self.signal_threshold = signal_threshold

    def forward(self, x):
        b, _, h, w = x.size()

        x_range = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype)
        y_range = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype)

        y_grid, x_grid = torch.meshgrid(y_range, x_range, indexing="ij")

        x_grid = x_grid.expand(b, 1, h, w)
        y_grid = y_grid.expand(b, 1, h, w)

        signal_gate = (
            x.abs().amax(dim=(1, 2, 3), keepdim=True) > self.signal_threshold
        ).to(dtype=x.dtype)
        x_grid = x_grid * signal_gate
        y_grid = y_grid * signal_gate

        return torch.cat([x, x_grid, y_grid], dim=1)


class SEBlock2D(nn.Module):
    """Recalibrate feature channels with squeeze-and-excitation attention."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        mid_ch = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_ch, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_ch, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class AxialFactorizedBlock2D(nn.Module):
    """Factorize a square convolution into orthogonal axial branches."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.h_conv = nn.Conv2d(
            in_channels, out_channels, (kernel_size, 1), padding=(padding, 0)
        )
        self.w_conv = nn.Conv2d(
            in_channels, out_channels, (1, kernel_size), padding=(0, padding)
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        self.se = SEBlock2D(out_channels, reduction=4)

        self.conv_final = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.h_conv(x) + self.w_conv(x)
        out = self.bn(out)
        out = self.gelu(out)

        out = self.se(out)

        out = self.conv_final(out)
        return out + identity


class HepaLite25D(nn.Module):
    """Multi-task teacher network for 2.5D segmentation and staging."""

    def __init__(self, in_channels=5, out_classes=1, init_features=16):
        super().__init__()
        f = init_features

        self.add_coords = AddCoords()

        self.stem = AxialFactorizedBlock2D(in_channels + 2, f)

        self.down1 = nn.Conv2d(f, f * 2, 3, stride=2, padding=1)
        self.enc1 = AxialFactorizedBlock2D(f * 2, f * 2)

        self.down2 = nn.Conv2d(f * 2, f * 4, 3, stride=2, padding=1)
        self.enc2 = AxialFactorizedBlock2D(f * 4, f * 4)

        self.down3 = nn.Conv2d(f * 4, f * 8, 3, stride=2, padding=1)
        self.bottleneck = AxialFactorizedBlock2D(f * 8, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec2 = AxialFactorizedBlock2D(f * 8, f * 4)

        self.up1 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec1 = AxialFactorizedBlock2D(f * 4, f * 2)

        self.up0 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec0 = AxialFactorizedBlock2D(f * 2, f)

        self.seg_out = nn.Conv2d(f, out_classes, kernel_size=1)

        self.staging_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(f * 8, 64),
            nn.GELU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        x = self.add_coords(x)

        s = self.stem(x)
        e1 = self.enc1(self.down1(s))
        e2 = self.enc2(self.down2(e1))
        b = self.bottleneck(self.down3(e2))

        staging_logits = self.staging_head(b)

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), s], dim=1))
        seg_logits = self.seg_out(d0)

        return seg_logits, staging_logits


class HepaLiteStudent25D(nn.Module):
    """Student network with optional squeeze-excitation and deep supervision."""

    def __init__(
        self,
        in_channels=5,
        out_classes=1,
        init_features=16,
        deep_supervision=True,
        use_se=True,
    ):
        super().__init__()
        f = init_features
        self.deep_supervision = deep_supervision
        self.use_se = use_se

        self.add_coords = AddCoords()

        self.stem = AxialFactorizedBlock2D(in_channels + 2, f)

        self.down1 = nn.Conv2d(f, f * 2, 3, stride=2, padding=1)
        self.enc1 = AxialFactorizedBlock2D(f * 2, f * 2)

        self.down2 = nn.Conv2d(f * 2, f * 4, 3, stride=2, padding=1)
        self.enc2 = AxialFactorizedBlock2D(f * 4, f * 4)

        self.down3 = nn.Conv2d(f * 4, f * 8, 3, stride=2, padding=1)
        self.bottleneck = AxialFactorizedBlock2D(f * 8, f * 8)

        self.up2 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec2 = AxialFactorizedBlock2D(f * 8, f * 4)

        self.up1 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec1 = AxialFactorizedBlock2D(f * 4, f * 2)

        self.up0 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec0 = AxialFactorizedBlock2D(f * 2, f)

        self.seg_out = nn.Conv2d(f, out_classes, kernel_size=1)

        if self.deep_supervision:
            self.ds_conv2 = nn.Conv2d(f * 2, out_classes, kernel_size=1)
            self.ds_conv3 = nn.Conv2d(f * 4, out_classes, kernel_size=1)
            self.ds_conv4 = nn.Conv2d(f * 8, out_classes, kernel_size=1)

        if not self.use_se:
            for module in self.modules():
                if isinstance(module, AxialFactorizedBlock2D):
                    module.se = nn.Identity()

    def forward(self, x):
        x = self.add_coords(x)

        s = self.stem(x)
        e1 = self.enc1(self.down1(s))
        e2 = self.enc2(self.down2(e1))
        b = self.bottleneck(self.down3(e2))

        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), s], dim=1))

        seg_logits = self.seg_out(d0)

        if self.training and self.deep_supervision:
            ds2_logits = F.interpolate(
                self.ds_conv2(d1),
                size=seg_logits.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            ds3_logits = F.interpolate(
                self.ds_conv3(d2),
                size=seg_logits.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            ds4_logits = F.interpolate(
                self.ds_conv4(b),
                size=seg_logits.shape[2:],
                mode="bilinear",
                align_corners=True,
            )
            return [seg_logits, ds2_logits, ds3_logits, ds4_logits]
        else:
            return seg_logits
