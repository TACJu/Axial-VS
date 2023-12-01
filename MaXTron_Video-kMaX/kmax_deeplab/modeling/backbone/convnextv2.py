# reference: https://github.com/SHI-Labs/OneFormer/blob/main/oneformer/modeling/backbone/convnext.py
# Modified by Qihang Yu

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from torch.cuda.amp import autocast


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        with autocast(enabled=False):
            x = x.float()
            if self.data_format == "channels_last":
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x

class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., out_indices=[0, 1, 2, 3],
                 ):
        super().__init__()

        self.num_features = dims

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.out_indices = out_indices

    def forward_features(self, x):
        outs = {}
        for i in range(4):
            # We add zero padding here for downstream tasks.
            # ref: https://github.com/google-research/deeplab2/blob/main/model/pixel_encoder/convnext.py#L128
            if i == 0:
                x = F.pad(x, (1, 2, 1, 2, 0, 0, 0, 0), "constant", 0)
            else:
                x = F.pad(x, (0, 1, 0, 1, 0, 0, 0, 0), "constant", 0)
            x = self.downsample_layers[i](x)
            if i == 0:
                outs["stem"] = x
            x = self.stages[i](x)
            if i in self.out_indices:
                outs["res{}".format(i + 2)] = x

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        return x

@BACKBONE_REGISTRY.register()
class D2ConvNeXtV2(ConvNeXtV2, Backbone):
    def __init__(self, cfg, input_shape):

        in_chans = cfg.MODEL.CONVNEXTV2.IN_CHANNELS
        depths = cfg.MODEL.CONVNEXTV2.DEPTHS
        dims = cfg.MODEL.CONVNEXTV2.DIMS
        drop_path_rate = cfg.MODEL.CONVNEXTV2.DROP_PATH_RATE
        out_indices = cfg.MODEL.CONVNEXTV2.OUT_INDICES

        super().__init__(
            in_chans=in_chans,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            out_indices=out_indices,
        )

        self._out_features = cfg.MODEL.CONVNEXT.OUT_FEATURES

        self._out_feature_strides = {
            "stem": 2,
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "stem": self.num_features[0],
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"ConvNeXt takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return -1