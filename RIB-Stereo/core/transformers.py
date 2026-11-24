import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Tuple, Union
from einops import rearrange


def to_3d(inputs: torch.Tensor, dim: int=1) -> torch.Tensor:
    if dim == 1:
        return rearrange(inputs, "b c h w -> (b h) w c")
    else:
        return rearrange(inputs, "b c h w -> b (h w) c")


def to_4d(inputs: torch.Tensor, b: Union[int, float], h: Union[int, float], w: Union[int, float]=None, dim: int=1) -> torch.Tensor:
    if dim == 1:
        return rearrange(inputs, "(b h) w c -> b c h w", b=b, h=h)
    else:
        return rearrange(inputs, "b (h w) c -> b c h w", h=h, w=w)


class TransformerFeatureRefiner(nn.Module):
    def __init__(self, channels: int, num_heads: int, depth: int, receptive_aug: str = None):
        super(TransformerFeatureRefiner, self).__init__()

        self.depth = depth
        for i in range(depth):
            setattr(
                self,
                f"refinement_{i + 1}",
                TransformerBlock(channels, num_heads, receptive_aug),
            )

    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor) -> Tuple[torch.Tensor]:
        for i in range(self.depth):
            feats1, feats2 = getattr(self, f"refinement_{i + 1}")(feats1.float(), feats2.float())

        return feats1, feats2


class TransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, receptive_aug: str = None):
        super(TransformerBlock, self).__init__()

        self.cross_attn = AttentionLayer(
            in_channels=channels,
            hidden_channels=channels,
            out_channels=channels,
            num_heads=num_heads,
            dropout=0.0,
            pre_norm=True,
            sink_competition=True,
        )
        self.feedforward = FeedForwardLayer(
            in_channels=channels,
            out_channels=channels,
            receptive_aug=receptive_aug,
        )

    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor, bi_cross_attn: bool = True) -> Union[
        torch.Tensor, Tuple[torch.Tensor]]:
        if bi_cross_attn:
            cat_feats1 = torch.cat([feats1, feats2], dim=0)
            cat_feats2 = torch.cat([feats2, feats1], dim=0)
            update_cat_feats1, _ = self.cross_attn(cat_feats1.clone(), cat_feats2)
            cat_feats1 = cat_feats1 + update_cat_feats1
            cat_feats1 = cat_feats1 + self.feedforward(cat_feats1.clone())

            feats1, feats2 = cat_feats1.chunk(2, dim=0)

            return feats1, feats2
        else:
            update_feats1, _ = self.cross_attn(feats1.clone(), feats2)
            feats1 = feats1 + update_feats1
            feats1 = feats1 + self.feedforward(feats1.clone())

            return feats1





class FeedForwardLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            expansion: int = 2,
            receptive_aug: bool = False,
            norm_layer: Any = nn.LayerNorm,
            act_func: Any = nn.SiLU,
    ):
        super(FeedForwardLayer, self).__init__()

        self.receptive_aug = receptive_aug

        self.norm = norm_layer(in_channels)
        self.in_conv = nn.Conv2d(in_channels, expansion * in_channels, kernel_size=1)
        self.act = act_func()

        if self.receptive_aug is not None:
            if self.receptive_aug == "pool":
                self.max_pool_3 = nn.MaxPool2d(kernel_size=(3, 3), stride=1, padding=1)
                self.max_pool_5 = nn.MaxPool2d(kernel_size=(5, 5), stride=1, padding=2)
                self.max_pool_7 = nn.MaxPool2d(kernel_size=(7, 7), stride=1, padding=3)

                self.avg_pool_3 = nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1)
                self.avg_pool_5 = nn.AvgPool2d(kernel_size=(5, 5), stride=1, padding=2)
                self.avg_pool_7 = nn.AvgPool2d(kernel_size=(7, 7), stride=1, padding=3)
            elif self.receptive_aug == "conv":
                self.dw_conv_3 = nn.Conv2d(in_channels=expansion * in_channels, out_channels=expansion * in_channels,
                                           kernel_size=3, stride=1, padding=1, groups=expansion * in_channels)
                self.dw_conv_5 = nn.Conv2d(in_channels=expansion * in_channels, out_channels=expansion * in_channels,
                                           kernel_size=5, stride=1, padding=2, groups=expansion * in_channels)
                self.dw_conv_7 = nn.Conv2d(in_channels=expansion * in_channels, out_channels=expansion * in_channels,
                                           kernel_size=7, stride=1, padding=3, groups=expansion * in_channels)
            else:
                raise ValueError(f"{self.receptive_aug} is not supported! Valid setting is 'pool' or 'conv'.")
            self.pointwise_conv = nn.Sequential(
                nn.Conv2d(in_channels=3 * expansion * in_channels, out_channels=expansion * in_channels, kernel_size=1),
                act_func(),
            )
        else:
            print("No receptive augmentation.")

        self.out_conv = nn.Conv2d(expansion * in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        b, c, h, w = inputs.shape
        inputs = to_4d(self.norm(to_3d(inputs)), b, h)
        inputs = self.in_conv(inputs)
        inputs = self.act(inputs)
        if self.receptive_aug is not None:
            if self.receptive_aug == "pool":
                max_3 = self.max_pool_3(inputs)
                max_5 = self.max_pool_5(inputs)
                max_7 = self.max_pool_7(inputs)
                max_feats = torch.cat([max_3, max_5, max_7], dim=1)

                avg_3 = self.avg_pool_3(inputs)
                avg_5 = self.avg_pool_5(inputs)
                avg_7 = self.avg_pool_7(inputs)
                avg_feats = torch.cat([avg_3, avg_5, avg_7], dim=1)

                inputs = inputs + self.pointwise_conv(max_feats + avg_feats)
            else:
                feats_3 = self.dw_conv_3(inputs)
                feats_5 = self.dw_conv_5(inputs)
                feats_7 = self.dw_conv_7(inputs)

                inputs = inputs + self.pointwise_conv(torch.cat([feats_3, feats_5, feats_7], dim=1))

        inputs = self.out_conv(inputs)

        return inputs

class AttentionLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            num_heads: int = 8,
            dropout: float = 0.0,
            pre_norm: bool = True,
            norm_layer: Any = nn.LayerNorm,
            sink_competition: bool = False,
            qkv_bias: bool = True,
            eps: float = 1e-6,
    ):
        super(AttentionLayer, self).__init__()

        self.eps = eps
        self.pre_norm = pre_norm
        self.sink_competition = sink_competition
        assert (hidden_channels % num_heads) == 0, "hidden_dim and num_heads are not divisible."
        self.scale = (hidden_channels // num_heads) ** -0.5
        self.num_heads = num_heads

        self.norm = norm_layer(in_channels if pre_norm else out_channels, eps=eps)
        self.norm_context = norm_layer(in_channels, eps=eps) if pre_norm else None

        self.to_q = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=qkv_bias)
        self.to_kv = nn.Conv2d(in_channels, hidden_channels * 2, kernel_size=1, bias=qkv_bias)
        self.to_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, feats1: torch.Tensor, feats2: torch.Tensor) -> Tuple[torch.Tensor]:
        b, c, h, w = feats1.shape
        if self.pre_norm:
            feats1 = to_4d(self.norm(to_3d(feats1)), b, h)
            feats2 = to_4d(self.norm_context(to_3d(feats2)), b, h)

        query = to_3d(self.to_q(feats1))
        key, value = to_3d(self.to_kv(feats2)).chunk(2, dim=-1)

        query, key, value = map(
            lambda t: rearrange(t, "bh w (n c) -> (bh n) w c", n=self.num_heads),
            (query, key, value),
        )
        similarity_matrix = torch.einsum("bid, bjd -> bij", query, key) * self.scale

        if self.sink_competition:
            raw_attn = F.softmax(similarity_matrix, dim=-1) + self.eps
            raw_attn = raw_attn / torch.sum(raw_attn, dim=(-1,), keepdim=True)
        else:
            raw_attn = F.softmax(similarity_matrix, dim=-1)

        attn = self.dropout(raw_attn)

        out = torch.einsum("bij, bjd -> bid", attn, value)
        out = rearrange(out, "(bh n) w c -> bh w (n c)", n=self.num_heads)
        out = to_3d(self.to_out(to_4d(out, b, h)))
        if not self.pre_norm:
            out = self.norm(out)
        out = to_4d(out, b, h)

        return out, similarity_matrix





class CrissCrossAttention(nn.Module):
    def __init__(self, in_dim, reduction=8):
        super().__init__()
        self.inter_channels = max(in_dim // reduction, 1)

        self.query_conv = nn.Conv2d(in_dim, self.inter_channels, 1)
        self.key_conv = nn.Conv2d(in_dim, self.inter_channels, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        Q = self.query_conv(x)
        K = self.key_conv(x)
        V = self.value_conv(x)

        row_energy = torch.einsum('bchw,bchv->bhwv', Q, K)
        col_energy = torch.einsum('bchw,bcvw->bhwv', Q, K)
        energy = torch.cat([row_energy, col_energy], dim=-1)

        # 屏蔽自身重复位置
        h_idx = torch.arange(H, device=x.device).view(1, H, 1, 1).expand(B, H, W, 1)
        dup_idx = (W + h_idx).long()
        energy = energy.scatter(-1, dup_idx, float('-inf'))

        attn = F.softmax(energy, dim=-1)
        attn_row = attn[..., :W]
        attn_col = attn[..., W:]

        out_row = torch.einsum('bhwv,bchv->bchw', attn_row, V)
        out_col = torch.einsum('bhwv,bcvw->bchw', attn_col, V)
        out = out_row + out_col
        out = self.gamma * out + x
        return out


class RCCA(nn.Module):
    """双层循环十字注意力，完成单图全局上下文建模"""

    def __init__(self, in_dim, reduction=8):
        super().__init__()
        self.cca = CrissCrossAttention(in_dim, reduction)

    def forward(self, x):
        h1 = self.cca(x)
        h2 = self.cca(h1)
        return h2
