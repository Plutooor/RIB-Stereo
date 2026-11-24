import timm
import math
import torch
import torch.nn as nn
from typing import List, Tuple, Union
from einops import rearrange



class AdaptiveGlobalRefiner(nn.Module):
    def __init__(self, channels: int):
        super(AdaptiveGlobalRefiner, self).__init__()

        # Pre-aggregation for the channels.
        self.pre_channel_agg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
        )

        # Attention-mask to fuse single-reception and multi-reception features, like in Selective-Stereo: https://github.com/Windsrain/Selective-Stereo.
        self.sam_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.cam_avgpool = nn.AdaptiveAvgPool2d(1)
        self.cam_maxpool = nn.AdaptiveMaxPool2d(1)
        self.cam_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

        # small-reception features.
        self.small_reception_conv = BasicConvIN(channels, channels, kernel_size=3, padding=1, stride=1)

        # large-reception features.
        self.large_repcetion_conv = BasicConvIN(channels, channels, kernel_size=5, padding=2, stride=1)

        # Final fusion features.
        self.final_fusion = nn.Sequential(
            BasicConvIN(channels, channels, kernel_size=3, padding=1, stride=1),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # Save the raw features.
        residual_feats = feats.clone()

        # Compute the channel pre-aggmentation.
        feats = self.pre_channel_agg(feats)

        # Compute the channel attention enhancement.
        cam_avg_out = self.cam_conv(self.cam_avgpool(feats))
        cam_max_out = self.cam_conv(self.cam_maxpool(feats))
        cam_out = cam_avg_out + cam_max_out
        cam_out = self.sigmoid(cam_out)
        feats = cam_out * feats
        # Compute the spatial attention.
        sam_avg_out = torch.mean(feats, dim=1, keepdim=True)
        sam_max_out, _ = torch.max(feats, dim=1, keepdim=True)
        attn_mask = torch.cat([sam_avg_out, sam_max_out], dim=1)
        attn_mask = self.sam_conv(attn_mask)
        attn_mask = self.sigmoid(attn_mask)

        # Compute the small-reception features.
        small_feats = self.small_reception_conv(feats)

        # Compute the large-reception features.
        large_feats = self.large_repcetion_conv(feats)

        # Compute the final refined features.
        refined_feats = small_feats * attn_mask + large_feats * (1 - attn_mask)
        refined_feats = self.final_fusion(residual_feats + refined_feats)

        return refined_feats

class BasicConvIN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, deconv: bool = False, is_3d: bool = False, IN: bool = True,
                 relu: bool = True, **kwargs):
        super(BasicConvIN, self).__init__()

        self.relu = relu
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outs = self.conv(inputs)
        if self.use_in:
            outs = self.IN(outs)
        if self.relu:
            outs = nn.LeakyReLU()(outs)

        return outs