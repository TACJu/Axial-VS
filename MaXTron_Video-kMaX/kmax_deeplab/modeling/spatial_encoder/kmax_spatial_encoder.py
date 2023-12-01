# Reference: https://github.com/google-research/deeplab2/blob/main/model/pixel_decoder/kmax.py
# Modified by Qihang Yu
from typing import Dict, List

import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY
from torch.cuda.amp import autocast

from .msdeformattn import MSDeformAttnPixelDecoder


@SEM_SEG_HEADS_REGISTRY.register()
class kMaXSpatialEncoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_attn_drop: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        transformer_temporal_layers: int,
        transformer_temporal_attn_type: str,
        transformer_conv_dim: int,
        transformer_in_features: List[str],
        transformer_temporal_in_features: List[str],
        transformer_skip_connect_spatial: bool,
        transformer_skip_connect_temporal: bool,
        transformer_zero_init_value: float,
        num_frames: int,
    ):
        super().__init__()

        self._spatial_module = MSDeformAttnPixelDecoder(input_shape=input_shape,transformer_dropout=transformer_dropout,transformer_attn_drop=transformer_attn_drop,transformer_nheads=transformer_nheads,
                                                                transformer_dim_feedforward=transformer_dim_feedforward,transformer_enc_layers=transformer_enc_layers,
                                                                transformer_temporal_layers=transformer_temporal_layers,transformer_temporal_attn_type=transformer_temporal_attn_type,
                                                                conv_dim=transformer_conv_dim,transformer_in_features=transformer_in_features,
                                                                transformer_temporal_in_features=transformer_temporal_in_features,
                                                                transformer_skip_connect_temporal=transformer_skip_connect_temporal,
                                                                transformer_zero_init_value=transformer_zero_init_value,
                                                                num_frames=num_frames)
        
        self._transformer_skip_connect_spatial = transformer_skip_connect_spatial
        self._transformer_zero_init_value = transformer_zero_init_value

        if self._transformer_skip_connect_spatial:
            transformer_input_shape = {
                k: v for k, v in input_shape.items() if k in transformer_in_features
            }
            transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
            transformer_in_channels = [v.channels for k, v in transformer_input_shape]
            self.gamma = nn.ParameterList([
                nn.Parameter(self._transformer_zero_init_value * torch.ones((channel)), requires_grad=True) for channel in transformer_in_channels[::-1]
            ])

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.IN_FEATURES
        }
        ret["transformer_dropout"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.DROPOUT
        ret["transformer_attn_drop"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.ATTN_DROP
        ret["transformer_nheads"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.NHEADS
        ret["transformer_dim_feedforward"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.DIM_FEEDFORWARD
        ret["transformer_enc_layers"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.ENC_LAYERS
        ret["transformer_temporal_layers"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.TEMPORAL_LAYERS
        ret["transformer_temporal_attn_type"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.TEMPORAL_ATTN_TYPE
        ret["transformer_conv_dim"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.CONV_DIM
        ret["transformer_in_features"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.IN_FEATURES
        ret["transformer_temporal_in_features"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.TEMPORAL_IN_FEATURES
        ret["transformer_skip_connect_spatial"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.SKIP_CONNECT_SPATIAL
        ret["transformer_skip_connect_temporal"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.SKIP_CONNECT_TEMPORAL
        ret["transformer_zero_init_value"] = cfg.MODEL.KMAX_DEEPLAB.SPATIAL_ENC.ZERO_INIT_VALUE
        ret["num_frames"] = cfg.INPUT.NUM_FRAMES
        return ret
    
    def forward_features(self, features):
        spatial_features = self._spatial_module.forward_features(features)
        for _, k in enumerate(spatial_features):
            if self._transformer_skip_connect_spatial:
                features[k] = features[k] + spatial_features[k] * self.gamma[_].unsqueeze(1).unsqueeze(2) # nn.Parameter
            else:
                features[k] = spatial_features[k]
        return features