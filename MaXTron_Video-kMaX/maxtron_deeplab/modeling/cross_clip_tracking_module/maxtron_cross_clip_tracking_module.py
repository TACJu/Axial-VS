# Reference: https://github.com/google-research/deeplab2/blob/main/model/transformer_decoder/kmax.py
# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
# Modified by Qihang Yu

from typing import Optional, List

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from einops import rearrange
from timm.models.layers import trunc_normal_tf_ as trunc_normal_

from kmax_deeplab.modeling.pixel_decoder.kmax_pixel_decoder import get_norm, ConvBN
from maxtron_deeplab.modeling.transformer_decoder.maxtron_transformer_decoder import add_bias_towards_void


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MaXTronCCPredictor(nn.Module):
    def __init__(self, num_classes=133+1):
        super().__init__()

        self._transformer_mask_head = ConvBN(256, 128, kernel_size=1, bias=False, norm='syncbn', act=None, conv_type='1d')
        self._transformer_class_head = ConvBN(256, num_classes, kernel_size=1, norm=None, act=None, conv_type='1d')
        self._transformer_class_activation_head = ConvBN(256, 1, kernel_size=1, norm=None, act=None, conv_type='1d')
        trunc_normal_(self._transformer_class_head.conv.weight, std=0.01)
        trunc_normal_(self._transformer_class_activation_head.conv.weight, std=0.01)
        self._transformer_class_head.conv.bias.data.fill_(0.0)
        self._transformer_class_activation_head.conv.bias.data.fill_(0.0)

        self._pixel_space_mask_batch_norm = get_norm('syncbn', channels=1)
        nn.init.constant_(self._pixel_space_mask_batch_norm.weight, 0.1)

    def forward(self, mask_embeddings, class_embeddings, pixel_feature, num_clips, num_clip_frames):
        # cluster_centers: BT' x C x N
        # pixel feature: BT' x C x TH x W
        cluster_class_activation_logits = self._transformer_class_activation_head(class_embeddings)
        cluster_class_activation_logits = cluster_class_activation_logits.softmax(dim=0)
        class_embeddings = (class_embeddings * cluster_class_activation_logits).sum(dim=0).unsqueeze(0) # B(1)xCxN
        cluster_class_logits = self._transformer_class_head(class_embeddings).permute(0, 2, 1).contiguous() # BxCxN->BxNxC, to align with mask2former format
        cluster_class_logits = add_bias_towards_void(cluster_class_logits)
        cluster_mask_kernel = self._transformer_mask_head(mask_embeddings)
        if self.training:
            mask_logits = torch.einsum('bchw,bcn->bnhw', pixel_feature, cluster_mask_kernel)
            mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)
            mask_logits = rearrange(mask_logits, '(B T) C (V H) W -> B C (T V) H W', T=num_clips, V=num_clip_frames)
        else:
            cluster_class_logits = cluster_class_logits.cpu()

            video_mask_logits = []
            for i in range(pixel_feature.shape[0]):
                clip_cluster_mask_kernel = cluster_mask_kernel[i].unsqueeze(0) # 1xCxN
                clip_pixel_feature = pixel_feature[i].unsqueeze(0) # 1xCxTHxW
                clip_mask_logits = torch.einsum('bchw,bcn->bnhw', clip_pixel_feature, clip_cluster_mask_kernel)
                video_mask_logits.append(clip_mask_logits)
            mask_logits = torch.cat(video_mask_logits, dim=0)
            mask_logits = self._pixel_space_mask_batch_norm(mask_logits.unsqueeze(dim=1)).squeeze(dim=1)
            mask_logits = rearrange(mask_logits, '(B T) C (V H) W -> B C (T V) H W', T=num_clips, V=num_clip_frames)
            mask_logits = mask_logits.cpu().to(torch.float32)

        return {
            'class_logits': cluster_class_logits,
            'mask_logits': mask_logits,
        }


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
        tgt2 = self.self_attn(tgt, seq_len, num_frames)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, seq_len, num_frames):
        tgt2 = self.norm(tgt)
        tgt2 = self.self_attn(tgt, seq_len, num_frames)
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, x, seq_len, num_frames):
        if self.normalize_before:
            return self.forward_pre(x, seq_len, num_frames)
        return self.forward_post(x, seq_len, num_frames)


class ASPP(nn.Module):
    def __init__(self, in_channels, output_channels, kernel_sizes, atrous_rates, dropout_rate, norm_fn):
        super().__init__()

        self._aspp_conv0 = nn.Conv1d(in_channels, output_channels, kernel_size=kernel_sizes[0], stride=1, dilation=atrous_rates[0], padding='same', padding_mode='replicate')
        self._aspp_conv1 = nn.Conv1d(in_channels, output_channels, kernel_size=kernel_sizes[1], stride=1, dilation=atrous_rates[1], padding='same', padding_mode='replicate')
        self._aspp_conv2 = nn.Conv1d(in_channels, output_channels, kernel_size=kernel_sizes[2], stride=1, dilation=atrous_rates[2], padding='same', padding_mode='replicate')
        
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
    

class CrossClipTrackingModule(nn.Module):
    def __init__(
        self,
        *,
        num_layers: int,
        num_classes: int,
        attn_drop: float,
        aspp_drop: float,
        kernel_sizes: List[int],
        atrous_rates: List[int],
        norm_fn: str,
        num_clip_frames: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        self.kernel_sizes = kernel_sizes
        self.atrous_rates = atrous_rates
        self.attn_drop = attn_drop
        self.aspp_drop = aspp_drop
        self.norm_fn = norm_fn
        self.num_clip_frames = num_clip_frames
        
        self.num_heads = 8
        self.num_layers = num_layers
        self.transformer_trajectory_self_attention_layers = nn.ModuleList()
        self.conv_short_aggregate_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_trajectory_self_attention_layers.append(
                TrajectoryAttentionLayer(
                    d_model=256,
                    nhead=8,
                    dropout=0.0,
                    attn_drop=self.attn_drop,
                    normalize_before=False,
                )
            )

            self.conv_short_aggregate_layers.append(
                ASPP(256, 256, self.kernel_sizes, self.atrous_rates, self.aspp_drop, self.norm_fn)
            )

            self.conv_norms.append(nn.LayerNorm(256))

        # init heads
        self._class_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',
                                                  conv_type='1d')

        self._mask_embedding_projection = ConvBN(256, 256, kernel_size=1, bias=False, norm='syncbn', act='gelu',
                                                  conv_type='1d')

        self._predictor = MaXTronCCPredictor(num_classes=num_classes+1)


    def forward(self, clip_query, panoptic_features):
        # input: aligned clip_query
        _, Q, T, _ = clip_query.shape
        panoptic_features = rearrange(panoptic_features, 'B C (T V) H W -> (B T) C (V H) W', V=self.num_clip_frames)

        predictions_class = []
        predictions_mask = []

        for i in range(self.num_layers):
            clip_query = rearrange(clip_query, 'b q t c -> b (t q) c', t=T)

            # do long temporal attention
            clip_query = self.transformer_trajectory_self_attention_layers[i](
                clip_query, seq_len=Q, num_frames=T,
            )
            clip_query = rearrange(clip_query, 'b (t q) c -> (b q) c t', t=T)

            # do short temporal conv
            clip_query = self.conv_norms[i](
                (self.conv_short_aggregate_layers[i](clip_query) + clip_query).transpose(1, 2)
            )
            
            clip_query = rearrange(clip_query, '(b q) t c -> b q t c', q=Q)
            video_query = rearrange(clip_query, 'b q t c -> (b t) c q')

            class_embeddings = self._class_embedding_projection(video_query)
            mask_embeddings = self._mask_embedding_projection(video_query) # BT x C x Q

            prediction_result = self._predictor(
                mask_embeddings=mask_embeddings,
                class_embeddings=class_embeddings,
                pixel_feature=panoptic_features,
                num_clips=T,
                num_clip_frames=self.num_clip_frames,
            )

            predictions_class.append(prediction_result['class_logits'])
            predictions_mask.append(prediction_result['mask_logits'])
        
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class, predictions_mask
            ),      
        }

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        target_size = outputs_seg_masks[-1].shape[-3:]
        align_corners = (target_size[-1] % 2 == 1)
        return [
            {"pred_logits": a, "pred_masks": F.interpolate(b, size=target_size, mode="trilinear", align_corners=align_corners)}
            for a, b, in zip(outputs_class[:-1], outputs_seg_masks[:-1])
        ]
