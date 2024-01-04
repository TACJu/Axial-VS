# Reference: https://github.com/google-research/deeplab2/blob/main/model/pixel_decoder/kmax.py
# Modified by Qihang Yu
from typing import Dict, List

from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .msdeformattn import MSDeformAttnPixelDecoder


@SEM_SEG_HEADS_REGISTRY.register()
class WithinClipTrackingModule(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_attn_drop: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_num_stages: int,
        transformer_spatial_layers: int,
        transformer_temporal_layers: int,
        transformer_temporal_attn_type: str,
        transformer_conv_dims: int,
        transformer_spatial_in_features: List[str],
        transformer_temporal_in_features: List[str],
        num_clip_frames: int,
        cross_clip_training: bool,
    ):
        super().__init__()

        self.within_clip_tracking_module = MSDeformAttnPixelDecoder(input_shape=input_shape, transformer_dropout=transformer_dropout, transformer_attn_drop=transformer_attn_drop,
                                                             transformer_nheads=transformer_nheads, transformer_dim_feedforward=transformer_dim_feedforward, 
                                                             transformer_spatial_layers=transformer_spatial_layers, transformer_temporal_layers=transformer_temporal_layers,
                                                             transformer_temporal_attn_type=transformer_temporal_attn_type, conv_dims=transformer_conv_dims,
                                                             transformer_spatial_in_features=transformer_spatial_in_features, transformer_temporal_in_features=transformer_temporal_in_features,
                                                             transformer_num_stages=transformer_num_stages, num_clip_frames=num_clip_frames, cross_clip_training=cross_clip_training)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.KMAX_DEEPLAB.WITHIN_CLIP_TRACKING_MODULE.SPATIAL_IN_FEATURES
        }
        ret["transformer_dropout"] = cfg.MODEL.KMAX_DEEPLAB.WITHIN_CLIP_TRACKING_MODULE.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.KMAX_DEEPLAB.WITHIN_CLIP_TRACKING_MODULE.NHEADS
        ret["transformer_dim_feedforward"] = cfg.MODEL.KMAX_DEEPLAB.WITHIN_CLIP_TRACKING_MODULE.DIM_FEEDFORWARD
        ret["transformer_spatial_layers"] = cfg.MODEL.KMAX_DEEPLAB.WITHIN_CLIP_TRACKING_MODULE.SPATIAL_LAYERS
        ret["transformer_conv_dims"] = cfg.MODEL.KMAX_DEEPLAB.WITHIN_CLIP_TRACKING_MODULE.CONV_DIMS
        ret["transformer_spatial_in_features"] = cfg.MODEL.KMAX_DEEPLAB.WITHIN_CLIP_TRACKING_MODULE.SPATIAL_IN_FEATURES
        return ret
    
    def forward_features(self, features):
        within_clip_features = self.within_clip_tracking_module.forward_features(features)
        for _, k in enumerate(within_clip_features):
            features[k] = within_clip_features[k]
        return features