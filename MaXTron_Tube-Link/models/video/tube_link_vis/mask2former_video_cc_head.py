# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from einops import rearrange
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32, auto_fp16

from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead

from models.video.tube_link_vps.utils import preprocess_video_panoptic_gt
from models.video.tube_link_vis.memory import retry_if_cuda_oom

from scipy.optimize import linear_sum_assignment
from timm.models.layers import trunc_normal_tf_ as trunc_normal_
from torch.cuda.amp import autocast


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
                if len(x.shape) == 4:
                    x = self.weight[:, None, None] * x + self.bias[:, None, None]
                else:
                    x = self.weight[:, None] * x + self.bias[:, None]
                return x


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def get_activation(name):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()


def get_norm(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()

    if name.lower() == 'syncbn':
        return nn.SyncBatchNorm(channels, eps=1e-3, momentum=0.01)
    
    if name.lower() == 'ln':
        return LayerNorm(channels, data_format='channels_first')


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm=None, act=None,
                 conv_type='2d', conv_init='he_normal', norm_init=1.0):
        super().__init__()
        
        if conv_type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)

        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

        if norm is not None:
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class ASPP(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_sizes, dilation_rates, dropout_rate, norm_fn):
        super().__init__()

        self._aspp_conv0 = nn.Conv1d(in_channels, output_channels, kernel_size=kernel_sizes[0], stride=1, dilation=dilation_rates[0], padding='same', padding_mode='replicate')
        self._aspp_conv1 = nn.Conv1d(in_channels, output_channels, kernel_size=kernel_sizes[1], stride=1, dilation=dilation_rates[1], padding='same', padding_mode='replicate')
        self._aspp_conv2 = nn.Conv1d(in_channels, output_channels, kernel_size=kernel_sizes[2], stride=1, dilation=dilation_rates[2], padding='same', padding_mode='replicate')

        if norm_fn != 'none':
            self._proj_conv_bn_act = ConvBN(output_channels * 3, output_channels, kernel_size=1, bias=False, norm=norm_fn, act='gelu', conv_type='1d')
        else:
            self._proj_conv_bn_act = ConvBN(output_channels * 3, output_channels, kernel_size=1, bias=False, act='gelu', conv_type='1d')

        self._proj_drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        results = []
        results.append(self._aspp_conv0(x))
        results.append(self._aspp_conv1(x))
        results.append(self._aspp_conv2(x))

        x = torch.cat(results, dim=1)
        x = self._proj_conv_bn_act(x)
        x = self._proj_drop(x)

        return x

class TrajectoryAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, attn_drop=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = TrajectoryAttention(d_model, nhead, attn_drop=attn_drop)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, seq_len, num_frames):
        tgt2 = self.self_attn(tgt, seq_len, num_frames)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, seq_len, num_frames):
        tgt2 = self.norm(tgt)
        tgt2 = self.self_attn(tgt, seq_len, num_frames)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, x, seq_len, num_frames):
        if self.normalize_before:
            return self.forward_pre(x, seq_len, num_frames)
        return self.forward_post(x, seq_len, num_frames)


class TrajectoryAttention(nn.Module):
    def __init__(self, d_model, nhead, attn_drop):
        super().__init__()
        self.num_heads = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_kv = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, seq_len=128, num_frames=6):
        B, N, C = x.shape
        P = seq_len
        F = num_frames
        h = self.num_heads

        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        # Reshape: 'b n (h d) -> (b h) n d'
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # Using full attention
        q_dot_k = q @ k.transpose(-2, -1)
        q_dot_k = rearrange(q_dot_k, 'b q (f n) -> b q f n', f=F)
        space_attn = (self.scale * q_dot_k).softmax(dim=-1)
        attn = self.attn_drop(space_attn)
        v = rearrange(v, 'b (f n) d -> b f n d', f=F, n=P)
        x = torch.einsum('b q f n, b f n d -> b q f d', attn, v)

        # Temporal attention: query is the similarity-aggregated patch
        x = rearrange(x, '(b h) s f d -> b s f (h d)', b=B)
        x_diag = rearrange(x, 'b (g n) f d -> b g n f d', g=F)
        x_diag = torch.diagonal(x_diag, dim1=-4, dim2=-2)
        x_diag = rearrange(x_diag, f'b n d f -> b (f n) d', f=F)
        q2 = self.proj_q(x_diag)
        k2, v2 = self.proj_kv(x).chunk(2, dim=-1)
        q2 = rearrange(q2, f'b s (h d) -> b h s d', h=h)
        q2 *= self.scale
        k2, v2 = map(
            lambda t: rearrange(t, f'b s f (h d) -> b h s f d', f=F,  h=h), (k2, v2))
        attn = torch.einsum('b h s d, b h s f d -> b h s f', q2, k2)
        attn = attn.softmax(dim=-1)

        x = torch.einsum('b h s f, b h s f d -> b h s d', attn, v2)
        x = rearrange(x, f'b h s d -> b s (h d)')

        x = self.proj(x)
        return x, space_attn

@HEADS.register_module()
class Mask2FormerVideoCCHeadTube(AnchorFreeHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80,
                 num_stuff_classes=53,
                 num_queries=100,
                 train_num_frames=25,
                 train_num_clips=5,
                 test_num_frames=5,
                 num_cc_layers=6,
                 trajectory_drop_out=0.0,
                 kernel_sizes=(3,3,3),
                 atrous_rates=(1,2,3),
                 drop_path_prob=0.1,
                 aspp_norm_fn=None,
                 num_transformer_feat_level=3,
                 point_loss=True,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.train_num_frames = train_num_frames
        self.train_num_clips = train_num_clips
        self.test_num_frames = test_num_frames
        self.num_cc_layers = num_cc_layers
        self.trajectory_drop_out = trajectory_drop_out
        self.kernel_sizes = kernel_sizes
        self.atrous_rates = atrous_rates
        self.drop_path_prob = drop_path_prob
        self.aspp_norm_fn = aspp_norm_fn
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.point_loss = point_loss
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.\
            attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)


        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        self.transformer_trajectory_self_attention_layers = nn.ModuleList()
        self.conv_short_aggregate_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()

        for _ in range(self.num_cc_layers):
            self.transformer_trajectory_self_attention_layers.append(
                TrajectoryAttentionLayer(
                    d_model=256,
                    nhead=8,
                    dropout=0.0,
                    attn_drop=self.trajectory_drop_out,
                    normalize_before=False,
                )
            )

            self.conv_short_aggregate_layers.append(
                ASPP(256, 256, self.kernel_sizes, self.atrous_rates, self.drop_path_prob, self.aspp_norm_fn)
            )

            self.conv_norms.append(nn.LayerNorm(256))

        self.activation_proj = nn.Linear(256, 1)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        for p in self.pixel_decoder.parameters():
            p.requires_grad_(False)
        for p in self.transformer_decoder.parameters():
            p.requires_grad_(False)
        for p in self.query_embed.parameters():
            p.requires_grad_(False)
        for p in self.query_feat.parameters():
            p.requires_grad_(False)
        for p in self.level_embed.parameters():
            p.requires_grad_(False)
        for p in self.decoder_input_projs.parameters():
            p.requires_grad_(False)

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_seg, gt_instance_ids, img_metas):
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_instance_ids (Tensor | None): instance id for each object
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(gt_labels_list)
        num_stuff_list = [self.num_stuff_classes] * len(gt_labels_list)
        if gt_semantic_seg is None:
            gt_semantic_seg = [None] * len(gt_labels_list)

        targets = multi_apply(
            preprocess_video_panoptic_gt,
            gt_labels_list,
            gt_masks_list,
            gt_semantic_seg,
            gt_instance_ids,
            num_things_list,
            num_stuff_list,
            img_metas
        )
        labels, masks = targets
        return labels, masks

    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list, pos_inds_list, neg_inds_list) = \
            multi_apply(self._get_target_single,
                        cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg


    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
        # Note that gt_masks are with num_gts, num_frames, h, w
        # but mask_preds are with num_frames, num_queries, h, w
        tmp_nf, tmp_h = gt_masks.size()[1:3]
        gt_masks = gt_masks.flatten(1, 2)  # num_gts, num_frames * h, w (turn to a long image)
        mask_pred = mask_pred.transpose(1, 0).flatten(1, 2)  # num_query, num_frames *h, w (turn to a long image)
        # sample points on both prediction and gt masks to calculate the cost
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]

        point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)

        mask_points_pred = point_sample(mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(1)
        # shape (num_gts, num_points)
        gt_points_masks = point_sample(gt_masks.unsqueeze(1).float(), point_coords.repeat(num_gts, 1, 1)).squeeze(1)

        # assign and sample
        assign_result = self.assigner.assign(cls_score, mask_points_pred, gt_labels, gt_points_masks, img_meta=None)
        sampling_result = self.sampler.sample(assign_result, mask_pred, gt_masks)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label target
        labels = gt_labels.new_full((num_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones((num_queries, ))

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((num_queries, ))
        mask_weights[pos_inds] = 1.0

        return labels, label_weights, mask_targets.unflatten(1, (tmp_nf, tmp_h)), mask_weights, pos_inds, neg_inds

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
             gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list, gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list, num_total_pos, num_total_neg) = \
            self.get_targets(cls_scores_list, mask_preds_list, gt_labels_list, gt_masks_list, img_metas)
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classification loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)  # (b*n, c)
        labels = labels.flatten(0, 1) # (b*n, )
        label_weights = label_weights.flatten(0, 1)

        class_weight = cls_scores.new_tensor(self.class_weight)
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum())

        num_total_masks = reduce_mean(cls_scores.new_tensor([num_total_pos]))
        num_total_masks = max(num_total_masks, 1)

        # HY : Note that mask_pred is with bs, num_frames, num_queries, h, w
        # bs, num_frames, num_queries, h, w -> bs, num_queries, num_frames * h, w
        mask_preds = mask_preds.transpose(2, 1).flatten(2, 3)
        # HY : mask_targets is with n_gt, num_frames, h, w
        mask_targets = mask_targets.flatten(1, 2)
        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        if self.point_loss:
            with torch.no_grad():
                points_coords = get_uncertain_point_coords_with_randomness(
                    mask_preds.unsqueeze(1), None, self.num_points,
                    self.oversample_ratio, self.importance_sample_ratio)
                # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
                mask_point_targets = point_sample(mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)
            # shape (num_queries, h, w) -> (num_queries, num_points)
            mask_point_preds = point_sample(mask_preds.unsqueeze(1), points_coords).squeeze(1)
        else:
            _, h, w = mask_targets.size()
            mask_point_targets = mask_targets.flatten(1, 2)
            mask_point_preds = F.interpolate(mask_preds.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1).flatten(1, 2)

        # dice loss
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        num_points = mask_point_preds.size(1)

        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)

        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            avg_factor=num_total_masks * num_points
        )

        return loss_cls, loss_mask, loss_dice

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries,h, w).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (num_queries, batch_size, c)
        cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward_head_video(self, decoder_out, mask_feature, attn_mask_target_size):
        # similar to self.forward_head but for video.
        # NBC -> BNC for decoder_out
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum('bqc,btchw->btqhw', mask_embed, mask_feature)
        bs, nf = mask_pred.size()[:2]
        attn_mask = F.interpolate(
            mask_pred.flatten(0, 1),
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False).unflatten(0, (bs, nf))
        # video : (b, num_frame, n, h, w) -> (b * num_head, num_frame, n, h * w)
        attn_mask = attn_mask.flatten(3).unsqueeze(1).repeat((1, self.num_heads, 1, 1, 1)).flatten(0, 1)
        # video : (b * num_head, num_frame, n, h * w) - > (b * num_head, n, nf * h * w) num_head=8 by default
        attn_mask = attn_mask.transpose(1, 2).flatten(2)
        attn_mask = attn_mask.sigmoid() < 0.5  # default setting by torch.
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward_head_clips(self, decoder_out, mask_feature):
        # similar to self.forward_head but for video.
        # NBC -> BNC for decoder_out
        num_clips = decoder_out.shape[0]
        num_frames = mask_feature.shape[1]
        frames_per_clip = num_frames // num_clips

        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        decoder_out = decoder_out.permute(1,3,0,2,4)

        outputs_class = self.pred_class(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        
        outputs_masks = []
        for _ in range(num_clips):
            outputs_mask = torch.einsum("lbqc,btchw->lbtqhw", mask_embed[:,:,_], mask_feature[:,frames_per_clip*_:frames_per_clip*(_+1)])
            outputs_masks.append(outputs_mask)
        outputs_masks = torch.cat(outputs_masks, dim=2)
        outputs_masks = outputs_masks.unbind(0)

        return outputs_class, outputs_masks

    def pred_class(self, decoder_output):
        """
        fuse the objects queries of all frames and predict an overall score based on the fused objects queries
        :param decoder_output: instance queries, shape is (l, b, t, q, c)
        """
        T = decoder_output.size(2)

        # compute the weighted average of the decoder_output
        activation = self.activation_proj(decoder_output).softmax(dim=2)  # (l, b, t, q, 1)
        class_output = (decoder_output * activation).sum(dim=2, keepdim=True)  # (l, b, 1, q, c)

        # to unify the output format, duplicate the fused features T times
        class_output = class_output.squeeze(2) # (l b q c)
        outputs_class = self.cls_embed(class_output)
        return outputs_class.unbind(0)

    def forward(self, feats, img_metas, clip_feature_frames):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 h, w).
        """

        self.pixel_decoder.eval()
        self.transformer_decoder.eval()

        batch_size = len(img_metas)
        # num_frame = len(img_metas[0])

        num_clips = len(feats)
        frames_per_clip = clip_feature_frames
        unmatched_query_feat = []
        # unmatched_cls_pred = []
        # unmatched_mask_pred = []
        matched_query_feat = []
        # matched_cls_pred = []
        # matched_mask_pred = []
        mask_features_list = []

        for nc in range(num_clips):
            mask_features, multi_scale_memorys = self.pixel_decoder(feats[nc], frames_per_clip) # feats: list of multi-scale feature: each shape is (b*T, c, h, w)
            bs_nf, _c, _h, _w = mask_features.size()
            mask_features = mask_features.reshape((batch_size, frames_per_clip, _c, _h, _w))
            mask_features_list.append(mask_features)
            for idx in range(len(multi_scale_memorys)):
                bs_nf, _c, _h, _w = multi_scale_memorys[idx].size()
                multi_scale_memorys[idx] = multi_scale_memorys[idx].reshape((batch_size, frames_per_clip, _c, _h, _w))
            # multi_scale_memorys (from low resolution to high resolution)
            decoder_inputs = []
            decoder_positional_encodings = []
            for i in range(self.num_transformer_feat_level):
                # assuming self.decoder_input_projs is identity
                decoder_input = multi_scale_memorys[i]
                # image : shape (batch_size, c, h, w) -> (h*w, batch_size, c)
                # video : shape (bs, nf, c, h, w) -> (nf * h*w, bs, c)
                decoder_input = decoder_input.flatten(3).permute(1, 3, 0, 2).flatten(start_dim=0, end_dim=1)
                level_embed = self.level_embed.weight[i].view(1, 1, -1)
                decoder_input = decoder_input + level_embed
                mask = decoder_input.new_zeros((batch_size, frames_per_clip)+multi_scale_memorys[i].shape[-2:], dtype=torch.bool)
                decoder_positional_encoding = self.decoder_positional_encoding(mask)
                # image : shape (batch_size, c, h, w) -> (h*w, batch_size, c)
                # video : shape (bs, nf, c, h, w) -> (nf * h*w, bs, c)
                decoder_positional_encoding = decoder_positional_encoding.flatten(
                    3).permute(1, 3, 0, 2).flatten(start_dim=0, end_dim=1)
                decoder_inputs.append(decoder_input)
                decoder_positional_encodings.append(decoder_positional_encoding)
            # shape (num_queries, c) -> (num_queries, batch_size, c)
            query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
            query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

            # List to store mask classification and prediction
            cls_pred_list = []
            mask_pred_list = []
            # init mask prediction (the same as K-Net)
            cls_pred, mask_pred, attn_mask = self.forward_head_video(
                query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
            # cls_pred_list.append(cls_pred)
            # mask_pred_list.append(mask_pred)

            # explore the spatial-temporal multi-scale feature to generate mask classification
            for i in range(self.num_transformer_decoder_layers):  # default: 9
                level_idx = i % self.num_transformer_feat_level  # default: 3 (last three features in the pyramid)
                # if a mask is all True (all background), then set it all False.
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False

                # cross_attn + self_attn
                layer = self.transformer_decoder.layers[i]
                attn_masks = [attn_mask, None]
                query_feat = layer(
                    query=query_feat,
                    key=decoder_inputs[level_idx],
                    value=decoder_inputs[level_idx],
                    query_pos=query_embed,
                    key_pos=decoder_positional_encodings[level_idx],
                    attn_masks=attn_masks,
                    query_key_padding_mask=None,
                    # here we do not apply masking on padded region
                    key_padding_mask=None)
                cls_pred, mask_pred, attn_mask = self.forward_head_video(
                    query_feat, mask_features, multi_scale_memorys[
                        (i + 1) % self.num_transformer_feat_level].shape[-2:])
            
            # we only store the last cls_pred and mask_pred
            # cls_pred_list.append(cls_pred)
            # mask_pred_list.append(mask_pred)
            
            unmatched_query_feat.append(query_feat.permute(1,0,2).contiguous().squeeze(0).detach()) # [num_clips * (N C)]
            # unmatched_cls_pred.append(cls_pred.squeeze(0)) # [num_clips * (N C)]
            # unmatched_mask_pred.append(mask_pred.squeeze(0).permute(1,0,2,3).contiguous()) # [num_clips * (N T H W)]

        matched_query_feat.append(unmatched_query_feat[0]) 
        # matched_cls_pred.append(unmatched_cls_pred[0]) 
        # matched_mask_pred.append(unmatched_mask_pred[0]) 

        for cur_clip_idx in range(1, num_clips):
            indices = self.match_from_embds(matched_query_feat[-1], unmatched_query_feat[cur_clip_idx])
            matched_query_feat.append(unmatched_query_feat[cur_clip_idx][indices])
            # matched_cls_pred.append(unmatched_cls_pred[cur_clip_idx][indices])
            # matched_mask_pred.append(unmatched_mask_pred[cur_clip_idx][indices])
        
        clip_query = torch.stack(matched_query_feat, dim=0).unsqueeze(0) # B * num_clips * N * C
        # matched_cls_pred = (sum(matched_cls_pred) / num_clips) # N C
        # matched_mask_pred = torch.cat(matched_mask_pred, dim=1).permute(1,0,2,3) # T N H W

        mask_features = torch.cat(mask_features_list, dim=1)

        cls_pred_list = []
        mask_pred_list = []
        outputs = []

        B, T, Q, C = clip_query.shape

        clip_query = rearrange(clip_query, 'b t q c -> b c t q')
        for i in range(self.num_cc_layers):
            clip_query = rearrange(clip_query, 'b c t q -> b (t q) c')
            clip_query = self.transformer_trajectory_self_attention_layers[i](
                clip_query, seq_len=Q, num_frames=T,
            )
            clip_query = rearrange(clip_query, 'b (t q) c -> t (b q) c', t=T)

            clip_query = clip_query.permute(1, 2, 0)  # (bq, c, t)

            clip_query = self.conv_norms[i](
                (self.conv_short_aggregate_layers[i](clip_query) + clip_query).transpose(1, 2)
            ).transpose(1, 2)

            clip_query = clip_query.reshape(
                B, Q, C, T
            ).permute(0, 2, 3, 1)  # (b, c, t, q)

            outputs.append(clip_query)

        outputs = torch.stack(outputs, dim=0).permute(3, 0, 4, 1, 2) # (l, b, c, t, q) -> (t, l, q, b, c)
        cls_pred_list, mask_pred_list = self.forward_head_clips(outputs, mask_features) # (l b n c), (l b t n h w)

        return cls_pred_list, mask_pred_list

    def forward_train(self,
                      feats,
                      img_metas,
                      clip_feature_frames,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg,
                      gt_instance_ids,
                      gt_bboxes_ignore=None):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_instance_ids : instance id for matching
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # not consider ignoring bboxes
        assert gt_bboxes_ignore is None

        # forward
        all_cls_scores, all_mask_preds = self(feats, img_metas, clip_feature_frames)

        # preprocess ground truth
        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks, gt_semantic_seg, gt_instance_ids, img_metas)

        # loss
        losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)

        return losses
    

    def simple_test(self, feats, img_metas, clip_feature_frames, **kwargs):
        """Test without augmentaton.

        Args:
            feats (list[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two tensors.

            - mask_cls_results (Tensor): Mask classification logits,\
                shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            - mask_pred_results (Tensor): Mask logits, shape \
                (batch_size, num_queries, h, w).
        """
        all_cls_scores, all_mask_preds = self(feats, img_metas, clip_feature_frames)

        # use the mask from the last stage to generate final prediction
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]

        # upsample masks 
        img_shape = img_metas[0][0]['batch_input_shape']
        bs, nf = mask_pred_results.size()[:2]
        mask_pred_results = F.interpolate(
            mask_pred_results.flatten(0, 1),
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)
        mask_pred_results = mask_pred_results.unflatten(0, (bs, nf))

        return mask_cls_results, mask_pred_results

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0,1))

        cost_embd = 1 - cos_sim

        C = 1.0 * cost_embd
        C = C.cpu()
        indices = linear_sum_assignment(C.transpose(0,1))
        indices = indices[1]

        return indices


def upsmaple(mask_cls_results, mask_pred_results, query_feat, img_shape):
    bs, nf = mask_pred_results.size()[:2]
    mask_pred_results = F.interpolate(
        mask_pred_results.flatten(0, 1),
        size=(img_shape[0], img_shape[1]),
        mode='bilinear',
        align_corners=False)
    mask_pred_results = mask_pred_results.unflatten(0, (bs, nf))

    return mask_cls_results, mask_pred_results, query_feat