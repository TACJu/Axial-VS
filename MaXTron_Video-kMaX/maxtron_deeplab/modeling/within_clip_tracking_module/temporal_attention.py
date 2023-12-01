from einops import rearrange
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    def forward(self, query, key, value, num_frames=2):
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
        return x, space_attn


class TemporalEncoder(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.0, attn_drop=0.0, activation="relu", n_heads=8, 
                 temporal_attn_type="trajectory", num_temporal_layer=2):
        super().__init__()

        if temporal_attn_type == "trajectory":
            self.temporal_layers = nn.ModuleList([TemporalTrajectoryAttentionLayer(d_model, d_ffn, dropout, attn_drop, activation, n_heads) for _ in range(num_temporal_layer)])
        elif temporal_attn_type == "axial-trajectory":
            self.temporal_layers = nn.ModuleList([TemporalAxialTrajectoryAttentionLayer(d_model, d_ffn, dropout, attn_drop, activation, n_heads) for _ in range(num_temporal_layer)])

    def forward(self, src: Tensor, pos: Tensor):
        """
        Forward method
        :param src: tensor of shape [B*T, H*W, C]
        :param pos: tensor of shape [B, T, H, W, C]
        :return:
        """
        for layer in self.temporal_layers:
            src, height_traj_attn, width_traj_attn = layer(src, pos)

        return src, height_traj_attn, width_traj_attn


class TemporalTrajectoryAttentionLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.0, attn_drop=0.0, activation="relu", n_heads=8):
        super().__init__()

        # self attention
        self.temporal_attn = TrajectoryAttention(d_model, n_heads, dropout)
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
        batch_sz, clip_len = pos.shape[:2]

        src = rearrange(src, "(B T) HW C -> B (T HW) C", B=batch_sz, T=clip_len)
        pos = rearrange(pos, "B T H W C -> B (T H W) C")

        kq = self.with_pos_embed(src, pos)

        attn_output = self.temporal_attn(query=kq, key=kq, value=src, num_frames=clip_len)[0]  # [B, THW C]

        src = src + self.dropout1(attn_output)

        src = rearrange(src, "B (T HW) C -> (B T) HW C", T=clip_len)

        src = self.norm1(src)
        src = self.forward_ffn(src) # [BT HW C]

        return src, None, None


class TemporalAxialTrajectoryAttentionLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.0, attn_drop=0.0, activation="relu", n_heads=8):
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

        height_attn_output, height_traj_attn = self.height_attn(query=kq, key=kq, value=src, num_frames=clip_len)  # [BW, TH, C]

        src = src + self.dropout1(height_attn_output)

        src = rearrange(src, "(B W) (T H) C -> (B H) (T W) C", H=height, W=width)
        pos = rearrange(pos, "(B W) (T H) C -> (B H) (T W) C", H=height, W=width)
        
        kq = self.with_pos_embed(src, pos)

        width_attn_output, width_traj_attn = self.width_attn(query=kq, key=kq, value=src, num_frames=clip_len)  # [BH, TW, C]

        src = src + self.dropout1(width_attn_output)

        src = rearrange(src, "(B H) (T W) C -> (B T) (H W) C", H=height, W=width)

        src = self.norm1(src)
        src = self.forward_ffn(src) # [BT HW C]

        return src, height_traj_attn, width_traj_attn