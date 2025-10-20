import torch
import torch.nn as nn

class SimpleUNet(nn.Module):
    """Simplified U-Net for demo purposes with batch normalization, LeakyReLU, dropout and sigmoid activation."""

    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.2):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self._conv_block(in_channels, 32, dropout_p)
        self.enc2 = self._conv_block(32, 64, dropout_p)
        self.enc3 = self._conv_block(64, 128, dropout_p)
        self.enc4 = self._conv_block(128, 128, dropout_p)

        # Decoder (upsampling)
        self.dec4 = self._conv_block(128 + 128, 128, dropout_p)
        self.dec3 = self._conv_block(128 + 64, 64, dropout_p)
        self.dec2 = self._conv_block(64 + 32, 32, dropout_p)
        self.dec1 = nn.Conv2d(32, out_channels, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for final output

    def _conv_block(self, in_ch, out_ch, dropout_p=0.2):
        """Conv block with batch normalization and LeakyReLU: Conv -> BN -> LeakyReLU -> Dropout -> Conv -> BN -> LeakyReLU -> Dropout"""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_p),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(dropout_p)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # 256x256 -> 256x256
        e2 = self.enc2(self.pool(e1))  # 256x256 -> 128x128
        e3 = self.enc3(self.pool(e2))  # 128x128 -> 64x64
        e4 = self.enc4(self.pool(e3))  # 64x64 -> 32x32

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(e4), e3], 1)) # 32x32 -> 64x64
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], 1)) # 64x64 -> 128x128
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], 1)) # 128x128 -> 256x256
        out = self.dec1(d2)

        # Apply sigmoid activation to final output
        out = self.sigmoid(out)

        return out



class ContextBlock(nn.Module):
    def __init__(self, channels, dropout_p):
        super().__init__()

        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout1 = nn.Dropout2d(dropout_p)
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.dropout2 = nn.Dropout2d(dropout_p)

        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        c1 = self.activation(self.conv1(self.norm1(x)))
        d1 = self.dropout1(c1)
        c2 = self.activation(self.conv2(self.norm2(d1)))
        d2 = self.dropout1(c2)

        return x + d2


class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=4, dropout_p=0.2):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self.encoder_block(in_channels, 32, dropout_p)
        self.enc2 = self.encoder_block(32, 64, dropout_p)
        self.enc3 = self.encoder_block(64, 128, dropout_p)
        self.enc4 = self.encoder_block(128, 128, dropout_p)

        # Decoder (upsampling)
        self.dec4 = self.decoder_block(128 + 128, 128)
        self.dec3 = self.decoder_block(128 + 64, 64)
        self.dec2 = self.decoder_block(64 + 32, 32)
        self.dec1 = nn.Conv2d(32, out_channels, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for final output

    def encoder_block(self, in_channels, out_channels, dropout_p):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            ContextBlock(out_channels, dropout_p),
        )

    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.Conv2d(in_channels, out_channels, 1),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)              # 256x256 -> 256x256
        e2 = self.enc2(self.pool(e1))  # 256x256 -> 128x128
        e3 = self.enc3(self.pool(e2))  # 128x128 -> 64x64
        e4 = self.enc4(self.pool(e3))  # 64x64 -> 32x32

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upsample(e4), e3], 1)) # 32x32 -> 64x64
        d3 = self.dec3(torch.cat([self.upsample(d4), e2], 1)) # 64x64 -> 128x128
        d2 = self.dec2(torch.cat([self.upsample(d3), e1], 1)) # 128x128 -> 256x256
        out = self.dec1(d2)

        # Apply sigmoid activation to final output
        out = self.sigmoid(out)

        return out


class DiceLoss(nn.Module):
    """Dice Loss for binary segmentation.

    Dice Loss = 1 - Dice Coefficient
    Dice Coefficient = (2 * |X ∩ Y|) / (|X| + |Y|)

    Args:
        smooth (float): Smoothing factor to avoid division by zero (default: 1e-6)
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Sigmoid output from model [B, H, W] (values between 0-1)
            targets: Binary ground truth [B, H, W] (values 0 or 1)
        """
        # Flatten tensors using reshape to handle non-contiguous memory layout
        predictions = predictions.reshape(-1)
        targets = targets.reshape(-1).float()

        # Calculate intersection and union
        intersection = (predictions * targets).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum() + targets.sum() + self.smooth)

        # Return Dice Loss (1 - Dice Coefficient)
        return 1 - dice_coeff
