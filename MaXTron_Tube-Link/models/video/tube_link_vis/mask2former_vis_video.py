# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
from mmdet.utils import get_root_logger

from mmdet.core import bbox2result, encode_mask_results
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors.single_stage import SingleStageDetector


@DETECTORS.register_module()
class TubeLinkVideoVIS(SingleStageDetector):

    async def async_simple_test(self, img, img_metas, **kwargs):
        raise NotImplementedError

    def __init__(self,
                 backbone,
                 neck=None,
                 panoptic_head=None,
                 panoptic_fusion_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 fix_backbone=False,
                 overlap=0,
                 ):
        super(SingleStageDetector, self).__init__(init_cfg=init_cfg)
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)

        panoptic_head_ = copy.deepcopy(panoptic_head)
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head = build_head(panoptic_head_)

        panoptic_fusion_head_ = copy.deepcopy(panoptic_fusion_head)
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = build_head(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.logger = get_root_logger()
        self.logger.info("[Unified Video Segmentation] Using customized tube_link_vps segmentor.")

        self.fix_backbone = fix_backbone
        if self.fix_backbone:
            self.backbone.train(mode=False)
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False
                self.logger.info(name + " is fixed.")

        self.clip_feature_frames = self.panoptic_head.train_num_frames // self.panoptic_head.train_num_clips
        self.overlap = overlap

        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward_dummy(self, img, img_metas):
        raise NotImplementedError

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks,
                      gt_semantic_seg=None,
                      gt_bboxes_ignore=None,
                      *,
                      ref_img=None,
                      ref_img_metas=None,
                      ref_gt_bboxes=None,
                      ref_gt_labels=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      ref_gt_semantic_seg=None,
                      ref_gt_instance_ids=None,
                      **kargs):

        self.backbone.eval()

        # add batch_input_shape in img_metas
        batch_input_shape = tuple(img.size()[-2:])
        for img_meta in img_metas:
            img_meta['batch_input_shape'] = batch_input_shape
        for ref_meta in ref_img_metas:
            for meta in ref_meta:
                meta['batch_input_shape'] = batch_input_shape
        # step 1 : extract volume img features
        bs, num_frame, three, h, w = ref_img.size()  # (b,T,3,h,w)

        video_x = []
        for i in range(self.panoptic_head.train_num_clips):
            ref_video_i = ref_img[:,self.clip_feature_frames*i:self.clip_feature_frames*(i+1)].reshape((bs * self.clip_feature_frames, three, h, w))
            video_x_i = self.extract_feat(ref_video_i)
            video_x.append(video_x_i)

        # step 2: forward the volume features
        losses = self.panoptic_head.forward_train(
            video_x,
            ref_img_metas,
            self.clip_feature_frames,
            ref_gt_bboxes,
            ref_gt_labels,
            ref_gt_masks,
            ref_gt_semantic_seg,
            ref_gt_instance_ids,
            gt_bboxes_ignore=None
        )

        return losses

    def simple_test(self, img, img_metas, ref_img, ref_img_metas, **kwargs):
        device = img.device
        del img
        bs, num_frame, three, h, w = ref_img.size()

        video_x = []
        # (b, t, 3, h, w)
        if num_frame > self.clip_feature_frames:
            if num_frame % self.clip_feature_frames == 0:
                num_clip = num_frame // self.clip_feature_frames
                pad_size = 0
            else:
                num_clip = num_frame // self.clip_feature_frames + 1
                pad_size = self.clip_feature_frames - (num_frame % self.clip_feature_frames)
            step_size = self.clip_feature_frames
            self.logger.info("#frames: {}; #clips: {}; clip_size: {}".format(num_frame, num_clip, step_size))
            for i in range(num_clip):
                start = i*step_size
                end = min(num_frame, (i+1) * step_size)
                if i == num_clip - 1 and pad_size != 0:
                    ref_video = torch.cat((ref_img[:, start:end], ref_img[:, end-1].unsqueeze(1).repeat(1,pad_size,1,1,1)), dim=1).reshape((bs * self.clip_feature_frames, three, h, w))
                else:
                    ref_video = ref_img[:, start:end].reshape(
                        (bs * (end - start), three, h, w))
                clip_x = self.extract_feat(ref_video)
                assert len(clip_x) == 4
                video_x.append(clip_x)
        elif num_frame == self.clip_feature_frames:
            ref_video = ref_img.reshape((bs * num_frame, three, h, w))
            video_x.append(self.extract_feat(ref_video))
            pad_size = 0
        else:
            pad_size = self.clip_feature_frames - num_frame
            ref_video = torch.cat((ref_img, ref_img[:, num_frame-1].unsqueeze(1).repeat(1,pad_size,1,1,1)), dim=1).reshape((bs * self.clip_feature_frames, three, h, w))
            video_x.append(self.extract_feat(ref_video))

        del ref_video
        del ref_img

        mask_cls_results, mask_pred_results = self.panoptic_head.simple_test(video_x, ref_img_metas, self.clip_feature_frames, **kwargs)

        if pad_size != 0:
            mask_pred_results = mask_pred_results[:,:-pad_size]

        results = [[] for _ in range(bs)]

        # for each frame results
        for frame_id in range(num_frame):
            # fuse the final panoptic segmentation results.
            result = self.panoptic_fusion_head.simple_test(
                mask_cls_results,
                mask_pred_results[:, frame_id],
                [ref_img_metas[idx][frame_id] for idx in range(bs)],
                **kwargs
            )

            for i in range(len(result)):
                if 'pan_results' in result[i]:

                    result[i]['pan_results'] = result[i]['pan_results'].detach(
                    ).cpu().numpy()

                if 'ins_results' in result[i]:
                    labels_per_image, bboxes, mask_pred_binary, _ = result[i]['ins_results']
                    # add the id in the box field.
                    bboxes = torch.cat(
                        [torch.arange(len(bboxes), dtype=bboxes.dtype, device=bboxes.device)[:, None] + 1,
                         bboxes], dim=1)
                    # sort by the score
                    inds = torch.argsort(bboxes[:, -1], descending=True)
                    labels_per_image = labels_per_image[inds][:30] # only keep final top-10 in each image
                    bboxes = bboxes[inds][:30]
                    mask_pred_binary = mask_pred_binary[inds][:30]
                    bbox_results = bbox2result(bboxes, labels_per_image, self.num_things_classes)
                    mask_results = [[] for _ in range(self.num_things_classes)]
                    for j, label in enumerate(labels_per_image):
                        mask = mask_pred_binary[j].detach().cpu().numpy()
                        mask_results[label].append(mask)
                    result[i]['ins_results'] = bbox_results, mask_results  # default format as instance segmentation.

                results[i].append(result[i])

        if self.num_stuff_classes == 0:
            # HY : starting from here, the codes are for video instance segmentation.
            # THe codes for vis does not support vps anymore.
            for i in range(len(results)):
                for j in range(len(results[i])):
                    bbox_results, mask_results = results[i][j]['ins_results']
                    results[i][j]['ins_results'] = (bbox_results, encode_mask_results(mask_results))

        return results

    def forward_test(self, imgs, img_metas, **kwargs):
        """Currently video seg model does not support aug test.
        So we only add batch input shape here
        """
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_metas)
            for img_id in range(batch_size):
                img_metas[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
        for ref_img, ref_img_meta in zip(kwargs['ref_img'], kwargs['ref_img_metas']):
            batch_size = len(kwargs['ref_img_metas'])
            for batch_id in range(batch_size):
                num_frame = len(ref_img_meta)
                for frame_id in range(num_frame):
                    kwargs['ref_img_metas'][batch_id][frame_id]['batch_input_shape'] = tuple(ref_img.size()[-2:])

        return self.simple_test(img=imgs, img_metas=img_metas, **kwargs)

    def aug_test(self, imgs, img_metas, **kwargs):
        raise NotImplementedError
    

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        super().train(mode=mode)
        if self.fix_backbone:
            self.backbone.train(mode=False)
        return self
