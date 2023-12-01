# Copyright (c) Facebook, Inc. and its affiliates.
import copy
from einops import rearrange
from typing import Dict, List

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import normal_

from detectron2.layers import ShapeSpec

from .ops.modules import MSDeformAttn
from .pos_embeddings import PositionEmbeddingSine, PositionEmbeddingSine3D
from .temporal_attention import TemporalEncoder


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
    

# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_stages=2,
                 num_spatial_layers=2, num_temporal_layers=4, 
                 temporal_attn_type="axial_trajectory", dim_feedforward=1024, 
                 dropout=0.1, attn_drop=0.1, activation="relu", enc_n_points=4,
                 num_spatial_feature_levels=3, num_temporal_feature_levels=2
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_spatial_layers = num_spatial_layers
        self.num_temporal_layers = num_temporal_layers

        assert num_spatial_layers + num_temporal_layers > 0, "number of layers should be greater than 0"
        if num_spatial_layers > 0:
            assert num_spatial_layers == num_stages, "number of spatial layers should be equal to number of stages"

        if num_spatial_layers > 0 and num_temporal_layers > 0:
            spatial_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_spatial_feature_levels, nhead, enc_n_points)
            temporal_layer = TemporalEncoder(d_model, dim_feedforward, dropout, attn_drop, activation, nhead, temporal_attn_type, num_temporal_layers//num_stages)
            self.encoder = MSDeformAttnTransformerEncoder(spatial_layer, num_spatial_layers, num_spatial_feature_levels, num_temporal_feature_levels, temporal_layer)
        elif num_spatial_layers > 0 and num_temporal_layers == 0:
            spatial_layer = MSDeformAttnTransformerEncoderLayer(d_model, dim_feedforward, dropout, activation, num_spatial_feature_levels, nhead, enc_n_points)
            self.encoder = MSDeformAttnTransformerEncoder(spatial_layer, num_spatial_layers, num_spatial_feature_levels)
        elif num_spatial_layers == 0 and num_temporal_layers > 0:
            temporal_layer = TemporalEncoder(d_model, dim_feedforward, dropout, attn_drop, activation, nhead, temporal_attn_type, num_temporal_layers//num_stages)
            self.encoder = TemporalTransformerEncoder(temporal_layer, num_stages, num_temporal_feature_levels)

        if num_spatial_layers > 0:
            self.level_embed_2d = nn.Parameter(torch.Tensor(num_spatial_feature_levels, d_model))
        if num_temporal_layers > 0:
            self.level_embed_3d = nn.Parameter(torch.Tensor(num_temporal_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_spatial_layers > 0:
            normal_(self.level_embed_2d)
        if self.num_temporal_layers > 0:
            normal_(self.level_embed_3d)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds, pos_embeds_3d):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        if self.num_spatial_layers > 0 and self.num_temporal_layers > 0:
            src_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            lvl_pos_embed_3d_flatten = []
            spatial_shapes = []
            for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                src = src.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embed_2d[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                src_flatten.append(src)
                mask_flatten.append(mask)

            for lvl, pos_embed_3d in enumerate(pos_embeds_3d):
                pos_embed_3d = rearrange(pos_embed_3d, "B T C H W -> B T H W C")
                lvl_pos_embed_3d = pos_embed_3d + self.level_embed_3d[lvl].view(1, 1, 1, 1, -1)
                lvl_pos_embed_3d_flatten.append(lvl_pos_embed_3d)

            src_flatten = torch.cat(src_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

            # encoder
            memory, height_traj_attn, width_traj_attn = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten, lvl_pos_embed_3d_flatten)
        elif self.num_spatial_layers > 0 and self.num_temporal_layers == 0:
            src_flatten = []
            mask_flatten = []
            lvl_pos_embed_flatten = []
            spatial_shapes = []
            for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                src = src.flatten(2).transpose(1, 2)
                mask = mask.flatten(1)
                pos_embed = pos_embed.flatten(2).transpose(1, 2)
                lvl_pos_embed = pos_embed + self.level_embed_2d[lvl].view(1, 1, -1)
                lvl_pos_embed_flatten.append(lvl_pos_embed)
                src_flatten.append(src)
                mask_flatten.append(mask)

            src_flatten = torch.cat(src_flatten, 1)
            mask_flatten = torch.cat(mask_flatten, 1)
            lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
            valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

            # encoder
            memory, height_traj_attn, width_traj_attn = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        elif self.num_spatial_layers == 0 and self.num_temporal_layers > 0:
            src_flatten = []
            lvl_pos_embed_3d_flatten = []
            spatial_shapes = []
            for lvl, src in enumerate(srcs):
                bs, c, h, w = src.shape
                spatial_shape = (h, w)
                spatial_shapes.append(spatial_shape)
                src = src.flatten(2).transpose(1, 2)
                src_flatten.append(src)

            for lvl, pos_embed_3d in enumerate(pos_embeds_3d):
                pos_embed_3d = rearrange(pos_embed_3d, "B T C H W -> B T H W C")
                lvl_pos_embed_3d = pos_embed_3d + self.level_embed_3d[lvl].view(1, 1, 1, 1, -1)
                lvl_pos_embed_3d_flatten.append(lvl_pos_embed_3d)

            spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten[0].device)
            level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

            # encoder
            memory, height_traj_attn, width_traj_attn = self.encoder(src_flatten, lvl_pos_embed_3d_flatten)

        return memory, height_traj_attn, width_traj_attn, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
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

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, spatial_layer, num_stages, transformer_num_spatial_feature_levels, transformer_num_temporal_feature_levels=0, temporal_layer=None):
        super().__init__()
        self.spatial_layers = _get_clones(spatial_layer, num_stages)
        self.transformer_num_spatial_feature_levels = transformer_num_spatial_feature_levels
        self.transformer_num_temporal_feature_levels = transformer_num_temporal_feature_levels

        if transformer_num_temporal_feature_levels > 0:
            self.temporal_layers = _get_clones(temporal_layer, num_stages)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos, padding_mask, pos_3d=None):
        output = src # [BT, LHW, C]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        if self.transformer_num_temporal_feature_levels > 0:
            for _, (spatial_layer, temporal_layer) in enumerate(zip(self.spatial_layers, self.temporal_layers)):
                output = spatial_layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

                split_size_or_sections = [None] * self.transformer_num_spatial_feature_levels
                for i in range(self.transformer_num_spatial_feature_levels):
                    if i < self.transformer_num_spatial_feature_levels - 1:
                        split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
                    else:
                        split_size_or_sections[i] = output.shape[1] - level_start_index[i]
                split_output = torch.split(output, split_size_or_sections, dim=1)
                input_temporal = split_output[:self.transformer_num_temporal_feature_levels]

                output = []
                for i, f in enumerate(input_temporal):
                    temporal_output, height_traj_attn, width_traj_attn = temporal_layer(src=f, pos=pos_3d[i])
                    output.append(temporal_output)
                for _ in split_output[self.transformer_num_temporal_feature_levels:]:
                    output.append(_)
                output = torch.cat(output, dim=1)
        else:
            for _, spatial_layer in enumerate(self.spatial_layers):
                output = spatial_layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
                height_traj_attn = None
                width_traj_attn = None

        return output, height_traj_attn, width_traj_attn
    

class TemporalTransformerEncoder(nn.Module):
    def __init__(self, temporal_layer, num_stages, transformer_num_temporal_feature_levels=2):
        super().__init__()
        self.temporal_layers = _get_clones(temporal_layer, num_stages)
        self.transformer_num_temporal_feature_levels = transformer_num_temporal_feature_levels

    def forward(self, src, pos_3d):
        output = src # L * [BT, HW, C]
        for temporal_layer in self.temporal_layers:
            input_temporal = output[:self.transformer_num_temporal_feature_levels]
            for i, f in enumerate(input_temporal):
                output[i], height_traj_attn, width_traj_attn = temporal_layer(src=f, pos=pos_3d[i])
        output = torch.cat(output, dim=1)
        
        return output, height_traj_attn, width_traj_attn
    

class MSDeformAttnPixelDecoder(nn.Module):
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
        conv_dims: int,
        # spatial & temporal input features
        transformer_spatial_in_features: List[str],
        transformer_temporal_in_features: List[str],
        # video
        num_clip_frames: int,
        # cross-clip training
        cross_clip_training: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_spatial_layers: number of msdeform layers
            transformer_temporal_layers: number of temporal layers
            transformer_temporal_attn_type: type of temporal layers
            conv_dims: number of output channels for the intermediate conv layers
            transformer_spatial_in_features: input features for msdeform layers
            transformer_temporal_in_features: input features for temporal layers
            num_clip_frames: number of frames in a clip
        """
        super().__init__()

        self.transformer_temporal_layers = transformer_temporal_layers
        self.num_clip_frames = num_clip_frames
        transformer_spatial_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_spatial_in_features
        }
        transformer_temporal_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_temporal_in_features
        }
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_spatial_input_shape = sorted(transformer_spatial_input_shape.items(), key=lambda x: x[1].stride)
        transformer_temporal_input_shape = sorted(transformer_temporal_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_spatial_in_features = [k for k, v in transformer_spatial_input_shape]  # starting from "res2" to "res5"
        self.transformer_temporal_in_features = [k for k, v in transformer_temporal_input_shape]  # starting from "res2" to "res5"
        transformer_spatial_in_channels = [v.channels for k, v in transformer_spatial_input_shape]

        self.transformer_num_spatial_feature_levels = len(self.transformer_spatial_in_features)
        self.transformer_num_temporal_feature_levels = len(self.transformer_temporal_in_features)
        if self.transformer_num_spatial_feature_levels > 1:
            input_proj_list = []
            output_proj_list = []
            # from low resolution to high resolution (res5 -> res2)
            for in_channels in transformer_spatial_in_channels[::-1]:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dims, kernel_size=1),
                    nn.GroupNorm(32, conv_dims),
                ))
                output_proj_list.append(nn.Sequential(
                    nn.Conv2d(conv_dims, in_channels, kernel_size=1),
                    nn.GroupNorm(32, in_channels),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
            self.output_proj = nn.ModuleList(output_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_spatial_in_channels[-1], conv_dims, kernel_size=1),
                    nn.GroupNorm(32, conv_dims),
                )])
            self.output_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(conv_dims, transformer_spatial_in_channels[-1], kernel_size=1),
                    nn.GroupNorm(32, transformer_spatial_in_channels[-1]),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        for proj in self.output_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dims,
            dropout=transformer_dropout,
            attn_drop=transformer_attn_drop,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_stages=transformer_num_stages,
            num_spatial_layers=transformer_spatial_layers,
            num_temporal_layers=transformer_temporal_layers,
            temporal_attn_type=transformer_temporal_attn_type,
            num_spatial_feature_levels=self.transformer_num_spatial_feature_levels,
            num_temporal_feature_levels=self.transformer_num_temporal_feature_levels,
        )

        N_steps = conv_dims // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.pe_layer_3d = PositionEmbeddingSine3D(N_steps, normalize=True)

        self.cross_clip_training = cross_clip_training

    def forward_features(self, features):
        BT = features['res5'].shape[0]
        B = BT // self.num_clip_frames if (self.training or self.cross_clip_training) else 1
        T = BT // B

        srcs = []
        pos = []
        pos_3d = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.transformer_spatial_in_features[::-1]):
            # x = features[f].float()  # deformable detr does not support half precision
            x = features[f]
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
            if self.transformer_temporal_layers > 0 and f in self.transformer_temporal_in_features:
                pos_3d.append(self.pe_layer_3d(rearrange(x, '(B T) C H W -> B T C H W', B=B, T=T), fmt='btchw'))

        y, height_traj_attn, width_traj_attn, spatial_shapes, level_start_index = self.transformer(srcs, pos, pos_3d)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_spatial_feature_levels
        for i in range(self.transformer_num_spatial_feature_levels):
            if i < self.transformer_num_spatial_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = {}
        for i, z in enumerate(y):
            out[self.transformer_spatial_in_features[self.transformer_num_spatial_feature_levels-1-i]] = self.output_proj[i](z.transpose(1, 2).contiguous().view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))

        return out, height_traj_attn, width_traj_attn