num_things_classes = 40
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes


# note: start from 10, the swin schedule changes
_base_ = [
    '../_base_/datasets/yvis_2021.py',
    '../_base_/models/mask2former_tube_r50.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/mask2former_swin_schedules_iter.py',
]


depths = [2, 2, 18, 2]
model=dict(
    fix_backbone=False,
    type='TubeLinkVideoVIS',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=depths,
        num_heads=[6, 12, 24, 48],
        window_size=12,
        pretrain_img_size=384,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=True,
        convert_weights=True,
        frozen_stages=-1,
    ),
    panoptic_head=dict(
        type='Mask2FormerVideoCCHeadTube',
        in_channels=[192, 384, 768, 1536],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=100,
        train_num_frames=30,
        train_num_clips=6,
        test_num_frames=6,
        num_cc_layers=4,
        trajectory_drop_out=0.1,
        kernel_sizes=(3,3,3),
        atrous_rates=(1,2,3),
        drop_path_prob=0.1,
        aspp_norm_fn='ln',
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
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
                        attn_drop=0.1,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_classes + [0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        loss_sem_seg=None,
    ),
)

# load tube_link_vps coco swin-l
load_from = "/mnt/bn/jieneng-eu-nas4web/ju/ckpt/Tube-Link/ytvis21/swin-l_66_sc_attn_drop_0.0/iter_10000.pth"

work_dir = 'work_dir/ytvis21/swin-l_maxtron_cc'


lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[2500, 7500],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_iters=500,
    warmup_ratio=0.001,
)

max_iters = 10000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

# no use following
interval = 2500
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=5
)

# Before 365001th iteration, we do evaluation every 5000 iterations.
# After 365000th iteration, we do evaluation every 368750 iterations,
# which means that we do evaluation at the end of training.
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
evaluation = dict()

"""
g8at configs/video/exp_tubeminvis/y21_swin_l_010_tubemin_2_5k_5k_10k.py work_dir/y21_swinl_010_nofix_tubemin_2_5k_5k_10k/latest.pth --format-only --eval-options resfile_path='work_dir/y21_swinl_010_nofix_tubemin_2_5k_5k_10k'
"""
