# Copyright (c) OpenMMLab. All rights reserved.
import math
import mmengine
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import warnings
from typing import List, Optional, no_type_check

from mmcv.cnn import (PLUGIN_LAYERS, Conv2d, ConvModule, caffe2_xavier_init,
                      normal_init, xavier_init)
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.cnn.bricks.registry import ATTENTION
from mmcv.runner import BaseModule, ModuleList

from mmdet.core.anchor import MlvlPointGenerator
from mmdet.models.utils.transformer import MultiScaleDeformableAttention

from mmengine.model import constant_init, xavier_init
from mmengine.registry import MODELS
from mmengine.utils import deprecated_api_warning
from einops import rearrange

from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
    

@PLUGIN_LAYERS.register_module()
class MSDeformAttnPixelDecoder(BaseModule):
    """Pixel decoder with multi-scale deformable attention.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        strides (list[int] | tuple[int]): Output strides of feature from
            backbone.
        feat_channels (int): Number of channels for feature.
        out_channels (int): Number of channels for output.
        num_outs (int): Number of output scales.
        norm_cfg (:obj:`mmcv.ConfigDict` | dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`mmcv.ConfigDict` | dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`mmcv.ConfigDict` | dict): Config for transformer
            encoder. Defaults to `DetrTransformerEncoder`.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
        init_cfg (:obj:`mmcv.ConfigDict` | dict): Initialization config dict.
    """

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 strides=[4, 8, 16, 32],
                 feat_channels=256,
                 out_channels=256,
                 num_outs=3,
                 norm_cfg=dict(type='GN', num_groups=32),
                 act_cfg=dict(type='ReLU'),
                 encoder=dict(
                     type='DetrTransformerEncoder',
                     num_layers=6,
                     transformerlayers=dict(
                         type='BaseTransformerLayer',
                         attn_cfgs=dict(
                             type='MultiScaleDeformableAxialTrajectoryAttention',
                             embed_dims=256,
                             num_heads=8,
                             num_levels=3,
                             num_temporal_levels=2,
                             num_temporal_layers=1,
                             num_temporal_dim=1024,
                             num_points=4,
                             im2col_step=64,
                             dropout=0.0,
                             batch_first=False,
                             skip_connect=True,
                             attn_drop=0.0,
                             norm_cfg=None,
                             init_cfg=None),
                         feedforward_channels=1024,
                         ffn_dropout=0.0,
                         operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                     init_cfg=None),
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.strides = strides
        self.num_input_levels = len(in_channels)
        self.num_encoder_levels = \
            encoder.transformerlayers.attn_cfgs.num_levels
        assert self.num_encoder_levels >= 1, \
            'num_levels in attn_cfgs must be at least one'
        self.use_temporal_attn = encoder.transformerlayers.attn_cfgs.type == 'MultiScaleDeformableAxialTrajectoryAttention'
        if self.use_temporal_attn:
            self.num_temporal_levels = encoder.transformerlayers.attn_cfgs.num_temporal_levels
            self.level_3d_encodeing = nn.Embedding(self.num_temporal_levels, feat_channels)

        input_conv_list = []
        # from top to down (low to high resolution)
        for i in range(self.num_input_levels - 1,
                       self.num_input_levels - self.num_encoder_levels - 1,
                       -1):
            input_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                norm_cfg=norm_cfg,
                act_cfg=None,
                bias=True)
            input_conv_list.append(input_conv)
        self.input_convs = ModuleList(input_conv_list)

        self.encoder = build_transformer_layer_sequence(encoder)
        self.postional_encoding = build_positional_encoding(
            positional_encoding)
        self.positional_encoding3d = PositionEmbeddingSine3D(positional_encoding.num_feats, normalize=True)
        # high resolution to low resolution
        self.level_encoding = nn.Embedding(self.num_encoder_levels,
                                           feat_channels)

        # fpn-like structure
        self.lateral_convs = ModuleList()
        self.output_convs = ModuleList()
        self.use_bias = norm_cfg is None
        # from top to down (low to high resolution)
        # fpn for the rest features that didn't pass in encoder
        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            lateral_conv = ConvModule(
                in_channels[i],
                feat_channels,
                kernel_size=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=None)
            output_conv = ConvModule(
                feat_channels,
                feat_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.use_bias,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.mask_feature = Conv2d(
            feat_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.num_outs = num_outs
        self.point_generator = MlvlPointGenerator(strides)

    def init_weights(self):
        """Initialize weights."""
        for i in range(0, self.num_encoder_levels):
            xavier_init(
                self.input_convs[i].conv,
                gain=1,
                bias=0,
                distribution='uniform')

        for i in range(0, self.num_input_levels - self.num_encoder_levels):
            caffe2_xavier_init(self.lateral_convs[i].conv, bias=0)
            caffe2_xavier_init(self.output_convs[i].conv, bias=0)

        caffe2_xavier_init(self.mask_feature, bias=0)

        normal_init(self.level_encoding, mean=0, std=1)
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        # init_weights defined in MultiScaleDeformableAttention
        for layer in self.encoder.layers:
            for attn in layer.attentions:
                if isinstance(attn, MultiScaleDeformableAttention):
                    attn.init_weights()

    def forward(self, feats, num_frames):
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            num_frames (int): Number of frames of each clip.

        Returns:
            tuple: A tuple containing the following:

            - mask_feature (Tensor): shape (batch_size, c, h, w).
            - multi_scale_features (list[Tensor]): Multi scale \
                    features, each in shape (batch_size, c, h, w).
        """
        # generate padding mask for each level, for each image
        batch_size = feats[0].shape[0]
        encoder_input_list = []
        padding_mask_list = []
        level_positional_encoding_list = []
        spatial_shapes = []
        reference_points_list = []
        if self.use_temporal_attn:
            level_3d_positional_encoding_list = []

        for i in range(self.num_encoder_levels):
            level_idx = self.num_input_levels - i - 1
            feat = feats[level_idx]
            feat_projected = self.input_convs[i](feat)
            h, w = feat.shape[-2:]

            # no padding
            padding_mask_resized = feat.new_zeros(
                (batch_size, ) + feat.shape[-2:], dtype=torch.bool)
            pos_embed = self.postional_encoding(padding_mask_resized)
            if self.use_temporal_attn:
                if i < self.num_temporal_levels:
                    padding_mask3d_resized = feat.new_zeros(
                        ((batch_size//num_frames, num_frames) + feat.shape[-3:])
                    )
                    pos_embed3d = self.positional_encoding3d(padding_mask3d_resized, fmt='btchw')
                    pos_embed3d = pos_embed3d.permute(0, 1, 3, 4, 2).contiguous()
                    level_embed3d = self.level_3d_encodeing.weight[i]
                    level_pos_embed3d = level_embed3d.view(1, 1, 1, 1, -1) + pos_embed3d
                    level_3d_positional_encoding_list.append(level_pos_embed3d)
            level_embed = self.level_encoding.weight[i]
            level_pos_embed = level_embed.view(1, -1, 1, 1) + pos_embed
            # (h_i * w_i, 2)
            reference_points = self.point_generator.single_level_grid_priors(
                feat.shape[-2:], level_idx, device=feat.device)
            # normalize
            factor = feat.new_tensor([[w, h]]) * self.strides[level_idx]
            reference_points = reference_points / factor

            # shape (batch_size, c, h_i, w_i) -> (h_i * w_i, batch_size, c)
            feat_projected = feat_projected.flatten(2).permute(2, 0, 1)
            level_pos_embed = level_pos_embed.flatten(2).permute(2, 0, 1)
            padding_mask_resized = padding_mask_resized.flatten(1)

            encoder_input_list.append(feat_projected)
            padding_mask_list.append(padding_mask_resized)
            level_positional_encoding_list.append(level_pos_embed)
            spatial_shapes.append(feat.shape[-2:])
            reference_points_list.append(reference_points)
        # shape (batch_size, total_num_query),
        # total_num_query=sum([., h_i * w_i,.])
        padding_masks = torch.cat(padding_mask_list, dim=1)
        # shape (total_num_query, batch_size, c)
        encoder_inputs = torch.cat(encoder_input_list, dim=0)
        level_positional_encodings = torch.cat(
            level_positional_encoding_list, dim=0)
        device = encoder_inputs.device
        # shape (num_encoder_levels, 2), from low
        # resolution to high resolution
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=device)
        # shape (0, h_0*w_0, h_0*w_0+h_1*w_1, ...)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        reference_points = torch.cat(reference_points_list, dim=0)
        reference_points = reference_points[None, :, None].repeat(
            batch_size, 1, self.num_encoder_levels, 1)
        valid_radios = reference_points.new_ones(
            (batch_size, self.num_encoder_levels, 2))
        # shape (num_total_query, batch_size, c)
        if self.use_temporal_attn:
            memory = self.encoder(
                query=encoder_inputs,
                key=None,
                value=None,
                query_pos=level_positional_encodings,
                query_pos3d=level_3d_positional_encoding_list,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                query_key_padding_mask=padding_masks,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_radios=valid_radios)
        else:
            memory = self.encoder(
                query=encoder_inputs,
                key=None,
                value=None,
                query_pos=level_positional_encodings,
                key_pos=None,
                attn_masks=None,
                key_padding_mask=None,
                query_key_padding_mask=padding_masks,
                spatial_shapes=spatial_shapes,
                reference_points=reference_points,
                level_start_index=level_start_index,
                valid_radios=valid_radios)
        # (num_total_query, batch_size, c) -> (batch_size, c, num_total_query)
        memory = memory.permute(1, 2, 0)

        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(memory, num_query_per_level, dim=-1)
        outs = [
            x.reshape(batch_size, -1, spatial_shapes[i][0],
                      spatial_shapes[i][1]) for i, x in enumerate(outs)
        ]

        for i in range(self.num_input_levels - self.num_encoder_levels - 1, -1,
                       -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + F.interpolate(
                outs[-1],
                size=cur_feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
            y = self.output_convs[i](y)
            outs.append(y)
        multi_scale_features = outs[:self.num_outs]

        mask_feature = self.mask_feature(outs[-1])
        return mask_feature, multi_scale_features


class PositionEmbeddingSine3D(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.is_3d = True

    @torch.no_grad()
    def forward(self, x, mask=None, fmt="btchw"):
        assert x.dim() == 5, f"{x.shape} should be a 5-dimensional Tensor, got {x.dim()}-dimensional Tensor instead"
        if fmt == "btchw":
            batch_sz, clip_len, _, height, width = x.shape
        elif fmt == "bcthw":
            batch_sz, _, clip_len, height, width = x.shape
        else:
            raise ValueError(f"Invalid format given: {fmt})")

        if mask is None:
            mask = torch.zeros((batch_sz, clip_len, height, width), device=x.device, dtype=torch.bool)

        not_mask = ~mask
        z_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (z_embed[:, -1:, :, :] + eps) * self.scale
            y_embed = y_embed / (y_embed[:, :, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t_floor_2 = torch.div(dim_t, 2, rounding_mode='trunc')
        dim_t = self.temperature ** (2 * dim_t_floor_2 / self.num_pos_feats)

        dim_t_z = torch.arange((self.num_pos_feats * 2), dtype=torch.float32, device=x.device)
        dim_t_z_floor_2 = torch.div(dim_t_z, 2, rounding_mode='trunc')
        dim_t_z = self.temperature ** (2 * dim_t_z_floor_2 / (self.num_pos_feats * 2))

        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t_z
        pos_x = torch.stack((pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :, 0::2].sin(), pos_z[:, :, :, :, 1::2].cos()), dim=5).flatten(4)
        pos = (torch.cat((pos_y, pos_x), dim=4) + pos_z)

        if fmt == "btchw":
            pos = pos.permute(0, 1, 4, 2, 3)
        elif fmt == "bcthw":
            pos = pos.permute(0, 4, 1, 2, 3)

        return pos


@ATTENTION.register_module()
class MultiScaleDeformableAxialTrajectoryAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_temporal_levels (int): The number of feature map used in
            Axial-Trajectory Attention. Default: 2
        num_temporal_layers (int): The number of temporal layers used in
            Axial-Trajectory Attention. Default: 1
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Default: 1.0.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_temporal_levels: int = 2,
                 num_temporal_layers: int = 1,
                 num_temporal_dim: int = 1024,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 skip_connect: bool = True,
                 attn_drop: float = 0.0,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[mmengine.ConfigDict] = None,
                 value_proj_ratio: float = 1.0):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_temporal_levels = num_temporal_levels
        self.num_temporal_layers = num_temporal_layers
        self.skip_connect = skip_connect
        self.attn_drop = attn_drop
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.temporal_layer = TemporalEncoder(value_proj_size, num_temporal_dim, attn_drop=self.attn_drop, num_temporal_layer=self.num_temporal_layers)
        if self.skip_connect:
            self.gamma = nn.Parameter(1e-6 * torch.ones((value_proj_size)), requires_grad=True)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    @no_type_check
    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiScaleDeformableAttention')
    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                query_pos3d: List[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            query_pos3d (torch.Tensor): The 3d positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """

        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # Need to split output here according to spatial dimensions of different layers, then do axial-trajectory attention
        # from low resolution to high resolution
        num_query_per_level = [e[0] * e[1] for e in spatial_shapes]
        outs = torch.split(output, num_query_per_level, dim=1)
        input_temporal = outs[:self.num_temporal_levels]
        
        output = []
        for i, f in enumerate(input_temporal):
            if self.skip_connect:
                output.append(f + self.gamma * self.temporal_layer(src=f, pos=query_pos3d[i]))
            else:
                output.append(self.temporal_layer(src=f, pos=query_pos3d[i]))
        for _ in outs[self.num_temporal_levels:]:
            output.append(_)
        output = torch.cat(output, dim=1)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class TrajectoryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj_q = nn.Linear(dim, dim, bias=True)
        self.proj_kv = nn.Linear(dim, dim * 2, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query, key, value, num_frames=5):
        # B THW C
        B = query.shape[0]
        F = num_frames
        h = self.num_heads

        # project x to q, k, v values
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Using full attention
        q_dot_k = q @ k.transpose(-2, -1)
        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)

        space_attn = (self.scale * q_dot_k).softmax(dim=-1)
        attn = self.attn_drop(space_attn)
        v_ = rearrange(v, 'b (f n) d -> b f n d', f=F)
        x = torch.einsum('b q f n, b f n d -> b q f d', attn, v_)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, 'b n d f -> b (f n) d', f=F)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, 'b s (h d) -> b h s d', h=h)
        q2 = q2 * self.scale
        k2, v2 = map(
            lambda t: rearrange(t, 'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)
        x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, 'b h s d -> b s (h d)')

        x = self.proj(x)
        return x


class TemporalEncoder(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.0, attn_drop=0.0, activation="relu", n_heads=8, num_temporal_layer=2):
        super().__init__()

        self.temporal_layers = nn.ModuleList([TemporalAxialTrajectoryAttentionLayer(d_model, d_ffn, dropout, attn_drop, activation, n_heads) for _ in range(num_temporal_layer)])

    def forward(self, src: Tensor, pos: Tensor):
        """
        Forward method
        :param src: tensor of shape [B*T, H*W, C]
        :param pos: tensor of shape [B, T, H, W, C]
        :return:
        """
        for layer in self.temporal_layers:
            src = layer(src, pos)

        return src


class TemporalAxialTrajectoryAttentionLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.0, attn_drop=0.0, activation="relu", n_heads=8):
        super().__init__()

        # self attention
        self.height_attn = TrajectoryAttention(d_model, n_heads, dropout)
        self.width_attn = TrajectoryAttention(d_model, n_heads, dropout)
        self.dropout1 = nn.Dropout(attn_drop)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src: Tensor, pos: Tensor):
        """
        Forward method
        :param src: tensor of shape [B*T, H*W, C]
        :param pos: tensor of shape [B, T, H, W, C]
        :param patch_mask_indices: tensor of shape [P, N] (int64) (N = patch area)
        :return:
        """
        batch_sz, clip_len, height, width = pos.shape[:4]

        src = rearrange(src, "(B T) (H W) C -> (B W) (T H) C", T=clip_len, H=height)
        pos = rearrange(pos, "B T H W C -> (B W) (T H) C")

        kq = self.with_pos_embed(src, pos)

        height_attn_output = self.height_attn(query=kq, key=kq, value=src, num_frames=clip_len)  # [BW, TH, C]

        src = src + self.dropout1(height_attn_output)

        src = rearrange(src, "(B W) (T H) C -> (B H) (T W) C", H=height, W=width)
        pos = rearrange(pos, "(B W) (T H) C -> (B H) (T W) C", H=height, W=width)
        
        kq = self.with_pos_embed(src, pos)

        width_attn_output = self.width_attn(query=kq, key=kq, value=src, num_frames=clip_len)  # [BH, TW, C]

        src = src + self.dropout1(width_attn_output)

        src = rearrange(src, "(B H) (T W) C -> (B T) (H W) C", H=height, W=width)

        src = self.norm1(src)
        src = self.forward_ffn(src) # [BT HW C]

        return src
