# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
# Reference: https://github.com/google-research/deeplab2/blob/main/model/kmax_deeplab.py
# Reference: https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py
# Modified by Qihang Yu

import math
from typing import Tuple, List
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.cross_clip_tracking_module.maxtron_cross_clip_tracking_module import CrossClipTrackingModule
from .modeling.cc_criterion import MaXTronCCSetCriterion
from .modeling.matcher import VideoHungarianMatcher

from scipy.optimize import linear_sum_assignment


@META_ARCH_REGISTRY.register()
class MaXTronCCDeepLab(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        cross_clip_tracking_module: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        pixel_confidence_threshold: float,
        class_threshold_thing: float,
        class_threshold_stuff: float,
        overlap_threshold: float,
        reorder_class_weight: float,
        reorder_mask_weight: float,
        metadata,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_shape: List[int],
        channel_last_format: bool,
        # video
        num_clip_frames: int,
        num_video_frames: int,
        post_processing_type: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            cross_clip_module: a module that enhances the cross-clip consistency
            criterion: a module that defines the loss
            num_queries: int, number of queries
            pixel_confidence_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            num_clip_frames: number of frames in a clip
            num_video_frames: number of frames in a short video
            post_processing_type: mask-wise for VPQ or pixel-wise for STQ
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.cross_clip_tracking_module = cross_clip_tracking_module
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.pixel_confidence_threshold = pixel_confidence_threshold
        self.class_threshold_thing = class_threshold_thing
        self.class_threshold_stuff = class_threshold_stuff
        self.reorder_class_weight = reorder_class_weight
        self.reorder_mask_weight = reorder_mask_weight
        self.channel_last_format = channel_last_format
        self.metadata = metadata
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.input_shape = input_shape
        self.num_clip_frames = num_clip_frames
        self.num_video_frames = num_video_frames
        self.post_processing_type = post_processing_type

        # Freeze backbone and sem_seg_head
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sem_seg_head.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())
        cross_clip_tracking_module = CrossClipTrackingModule(
            num_layers=cfg.MODEL.MAXTRON.CROSS_CLIP_TRACKING_MODULE.NUM_LAYERS,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            attn_drop=cfg.MODEL.MAXTRON.CROSS_CLIP_TRACKING_MODULE.ATTN_DROP,
            aspp_drop=cfg.MODEL.MAXTRON.CROSS_CLIP_TRACKING_MODULE.ASPP_DROP,
            kernel_sizes=cfg.MODEL.MAXTRON.CROSS_CLIP_TRACKING_MODULE.KERNEL_SIZES,
            atrous_rates=cfg.MODEL.MAXTRON.CROSS_CLIP_TRACKING_MODULE.ATROUS_RATES,
            norm_fn=cfg.MODEL.MAXTRON.CROSS_CLIP_TRACKING_MODULE.NORM_FN,
            num_clip_frames=cfg.INPUT.NUM_CLIP_FRAMES,
        )

        # Loss parameters:
        deep_supervision = cfg.MODEL.KMAX_DEEPLAB.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.KMAX_DEEPLAB.NO_OBJECT_WEIGHT
        share_final_matching = cfg.MODEL.KMAX_DEEPLAB.SHARE_FINAL_MATCHING

        # loss weights
        class_weight = cfg.MODEL.KMAX_DEEPLAB.CLASS_WEIGHT
        dice_weight = cfg.MODEL.KMAX_DEEPLAB.DICE_WEIGHT
        mask_weight = cfg.MODEL.KMAX_DEEPLAB.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(masking_void_pixel=cfg.MODEL.KMAX_DEEPLAB.MASKING_VOID_PIXEL)

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MAXTRON.CROSS_CLIP_TRACKING_MODULE.NUM_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = MaXTronCCSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            share_final_matching=share_final_matching,
            process_semantic=cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_WEIGHT > 0,
            pixel_insdis_temperature=cfg.MODEL.KMAX_DEEPLAB.PIXEL_INSDIS_TEMPERATURE,
            pixel_insdis_sample_k=cfg.MODEL.KMAX_DEEPLAB.PIXEL_INSDIS_SAMPLE_K,
            aux_semantic_temperature=cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_TEMPERATURE,
            aux_semantic_sample_k=cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_SAMPLE_K,
            masking_void_pixel=cfg.MODEL.KMAX_DEEPLAB.MASKING_VOID_PIXEL,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "cross_clip_tracking_module": cross_clip_tracking_module,
            "criterion": criterion,
            "num_queries": cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NUM_OBJECT_QUERIES,
            "pixel_confidence_threshold": cfg.MODEL.MAXTRON.TEST.PIXEL_CONFIDENCE_THRESHOLD,
            "class_threshold_thing": cfg.MODEL.MAXTRON.TEST.CLASS_THRESHOLD_THING,
            "class_threshold_stuff": cfg.MODEL.MAXTRON.TEST.CLASS_THRESHOLD_STUFF,
            "overlap_threshold": cfg.MODEL.MAXTRON.TEST.OVERLAP_THRESHOLD,
            "reorder_class_weight": cfg.MODEL.MAXTRON.TEST.REORDER_CLASS_WEIGHT,
            "reorder_mask_weight": cfg.MODEL.MAXTRON.TEST.REORDER_MASK_WEIGHT,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "input_shape": cfg.INPUT.IMAGE_SIZE,
            "channel_last_format": cfg.MODEL.KMAX_DEEPLAB.CHANNEL_LAST_FORMAT,
            # video
            "num_clip_frames": cfg.INPUT.NUM_CLIP_FRAMES,
            "num_video_frames": cfg.INPUT.NUM_VIDEO_FRAMES,
            "post_processing_type": cfg.MODEL.MAXTRON.TEST.POST_PROCESSING_TYPE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        is_real_pixels = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
            for real_pixels in video["is_real_pixels"]:
                is_real_pixels.append(real_pixels.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
        # Set all padded pixel values to 0.
        images = [x * y.to(x) for x, y in zip(images, is_real_pixels)]
        del is_real_pixels

        self.backbone.eval()
        self.sem_seg_head.eval()

        for idx in range(len(images)):
            cur_height, cur_width = images[idx].shape[-2:]
            if self.training:
                padding = (0, max(0, self.input_shape[1] - cur_width), 0, max(0, self.input_shape[0] - cur_height), 0, 0)
            else:
                input_height, input_width = self.input_shape[0], self.input_shape[1]
                align_corners = input_width % 2 == 1
                height_scale_factor = input_height / cur_height
                width_scale_factor = input_width / cur_width
                if height_scale_factor < width_scale_factor:
                    scale_height = False
                    scale_factor = height_scale_factor
                else:
                    scale_height = True
                    scale_factor = width_scale_factor
                scaled_width = cur_width
                scaled_height = cur_height
                if scale_factor < 1:
                    if scale_height:
                        scaled_width = input_width
                        scaled_height = round(cur_height * scale_factor)
                        images[idx] = F.interpolate(images[idx].unsqueeze(0), size=(scaled_height, scaled_width), mode="bilinear", align_corners=align_corners).squeeze(0)
                    else:
                        scaled_height = input_height
                        scaled_width = round(cur_width * scale_factor)
                        images[idx] = F.interpolate(images[idx].unsqueeze(0), size=(scaled_height, scaled_width), mode="bilinear", align_corners=align_corners).squeeze(0)
                padding = (0, max(0, input_width - scaled_width), 0, max(0, input_height - scaled_height), 0, 0)
            images[idx] = F.pad(images[idx], padding, value=0)

        if not self.training:
            pad_video_length = 0
            while (len(images) % self.num_clip_frames != 0):
                images.append(images[-1])
                pad_video_length += 1

        images = ImageList.from_tensors(images, -1)

        input_tensor = images.tensor
        if self.channel_last_format:
            input_tensor = input_tensor.to(memory_format=torch.channels_last)

        with torch.no_grad():
            clip_outputs = self.segmentor_windows_inference(input_tensor, window_size=self.num_clip_frames)
            pred_mask_embeddings = clip_outputs['pred_mask_embeddings']
            pixel_feature = clip_outputs['pixel_feature']
            pred_cluster_centers = clip_outputs['cluster_centers']

            matched_cluster_centers = []
            batch_size = pred_mask_embeddings.shape[0]
            for b in range(batch_size):
                curr_unmatched_pred_mask_embeddings = pred_mask_embeddings[b]
                curr_unmatched_cluster_centers = pred_cluster_centers[b]

                curr_matched_pred_mask_embeddings = []
                curr_matched_cluster_centers = []

                curr_matched_pred_mask_embeddings.append(curr_unmatched_pred_mask_embeddings[0])
                curr_matched_cluster_centers.append(curr_unmatched_cluster_centers[0])

                for i in range(1, len(curr_unmatched_pred_mask_embeddings)):
                    indices = self.match_from_embds(curr_matched_pred_mask_embeddings[-1], curr_unmatched_pred_mask_embeddings[i])
                    curr_matched_pred_mask_embeddings.append(curr_unmatched_pred_mask_embeddings[i][indices, :])
                    curr_matched_cluster_centers.append(curr_unmatched_cluster_centers[i][indices, :])

                curr_matched_cluster_centers = torch.stack(curr_matched_cluster_centers, dim=1)
                matched_cluster_centers.append(curr_matched_cluster_centers)

        matched_cluster_centers = torch.stack(matched_cluster_centers, dim=0)
        video_outputs = self.cross_clip_tracking_module(matched_cluster_centers, pixel_feature)

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)

            with autocast(enabled=False):
                # bipartite matching-based loss
                for output_key in ["pred_masks", "pred_logits"]:
                    video_outputs[output_key] = video_outputs[output_key].float()
                if "aux_outputs" in video_outputs:
                    for i in range(len(video_outputs["aux_outputs"])):
                        for output_key in ["pred_masks", "pred_logits"]:
                            video_outputs["aux_outputs"][i][output_key] = video_outputs["aux_outputs"][i][output_key].float()

                losses = self.criterion(video_outputs, targets)
                    
                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                return losses 
        else:
            panoptic_segs = []
            segment_infos = []

            mask_cls_results = video_outputs["pred_logits"]
            mask_pred_results = video_outputs["pred_masks"]
            mask_pred_embeddings = pred_mask_embeddings

            mask_cls_result = mask_cls_results[0] # B N C -> N C
            mask_pred_result = mask_pred_results[0] # B N T H W -> N T H W
            mask_pred_embeddings = mask_pred_embeddings[0][0] # not important

            if pad_video_length > 0:
                mask_pred_result = mask_pred_result[:,:-pad_video_length]

            del video_outputs

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation
            align_corners = (images.tensor.shape[-1] % 2 == 1) 

            height = input_per_image.get("height", image_size[0])[0]  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])[0]

            mask_pred_result = retry_if_cuda_oom(self.video_seg_post_processing)(mask_pred_result, align_corners, images.tensor.shape[-2], images.tensor.shape[-1],
                                                                                 scale_factor, scaled_height, scaled_width, height, width)

            if self.post_processing_type == "mask-wise":
                panoptic_imgs, segments_info = retry_if_cuda_oom(self.panoptic_mask_inference)(mask_cls_result, mask_pred_result, mask_pred_embeddings)

            panoptic_segs.append(panoptic_imgs)
            segment_infos.append(segments_info)
            
            # to warp it in a way that evaluator likes
            return [[panoptic_segs, segment_infos]]

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))
        cost_embd = 1 - cos_sim
        C = 1.0 * cost_embd
        C = C.cpu()
        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target
        return indices

    def segmentor_windows_inference(self, images_tensor, window_size=2):
        BT = images_tensor.shape[0]
        B = BT // self.num_video_frames if self.training else 1
        T = BT // B

        clip_outputs = {}
        images_tensor = rearrange(images_tensor, '(b t) c h w -> b t c h w', t=T)
        iters = T // window_size
        if T % window_size != 0:
            iters += 1

        outs_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size

            input_clip_tensor = rearrange(images_tensor[:, start_idx:end_idx], 'b t c h w -> (b t) c h w')
            features = self.backbone(input_clip_tensor)
            output = self.sem_seg_head(features)

            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(output['aux_outputs'])):
                del output['aux_outputs'][j]['pred_masks'], output['aux_outputs'][j]['pred_logits'], output['aux_outputs'][j]['pixel_feature']

            outs_list.append(output)

        clip_outputs['pred_mask_embeddings'] = torch.stack([x['pred_mask_embeddings'] for x in outs_list], dim=1).detach()
        clip_outputs['cluster_centers'] = torch.stack([x['cluster_centers'] for x in outs_list], dim=1).detach()
        clip_outputs['pred_masks'] = torch.stack([x['pred_masks'] for x in outs_list], dim=1).detach()
        clip_outputs['pred_logits'] = torch.stack([x['pred_logits'] for x in outs_list], dim=1).detach()
        clip_outputs['pixel_feature'] = torch.cat([x['pixel_feature'] for x in outs_list], dim=2).detach()
        
        return clip_outputs

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        h_gt, w_gt = math.ceil(h_pad/4), math.ceil(w_pad/4)
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_video_frames, h_gt, w_gt]
            semantic_mask_shape = [self.num_video_frames, h_gt, w_gt]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            semantic_gt_masks_per_video = torch.zeros(semantic_mask_shape, dtype=torch.int64, device=self.device)

            gt_ids_per_video = []
            gt_classes_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_classes_per_video.append(targets_per_frame.gt_classes[:, None])
                gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks
                semantic_gt_masks_per_video[f_i, :h, :w] = targets_per_video["sem_seg_gt"][f_i]
            
            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)
            gt_ids_per_video = gt_ids_per_video[valid_idx]
            gt_ids_per_video = gt_ids_per_video.max(dim=1)[0]

            gt_classes_per_video = torch.cat(gt_classes_per_video, dim=1)
            gt_classes_per_video = gt_classes_per_video[valid_idx]
            gt_classes_per_video = gt_classes_per_video.max(dim=1)[0]
            
            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video, "semantic_masks": semantic_gt_masks_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def video_seg_post_processing(self, mask_pred_result, align_corners, image_height, image_width, scale_factor, new_height, new_width, height, width):
        mask_pred_result = F.interpolate(
            mask_pred_result,
            size=(image_height, image_width),
            mode="bilinear",
            align_corners=align_corners,
        )

        if scale_factor < 1:
            mask_pred_result = mask_pred_result[:, :, :new_height, :new_width]
            mask_pred_result = F.interpolate(
                mask_pred_result, size=(height, width), mode="bilinear", align_corners=align_corners
            )
        else:
            mask_pred_result = mask_pred_result[:, :, :height, :width]

        return mask_pred_result
    
    def panoptic_mask_inference(self, mask_cls, mask_pred, mask_embedding):
        # mask_cls: N x C
        # mask_pred: N x T x H x W
        # mask_embedding: N x C
        # some hyper-params
        num_mask_slots = mask_pred.shape[0]
        cls_threshold_thing = self.class_threshold_thing
        cls_threshold_stuff = self.class_threshold_stuff
        pixel_confidence_threshold = self.pixel_confidence_threshold
        overlap_threshold = self.overlap_threshold
        reorder_class_weight = self.reorder_class_weight
        reorder_mask_weight = self.reorder_mask_weight

        # https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py#L675
        # https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py#L199
        cls_scores, cls_labels = F.softmax(mask_cls, dim=-1)[..., :-1].max(-1) # N
        mask_scores = F.softmax(mask_pred, dim=0)
        
        binary_masks = mask_scores > pixel_confidence_threshold # N x T x H x W
        mask_scores_flat = mask_scores.flatten(1) # N x THW
        binary_masks_flat = binary_masks.flatten(1).float() # N x THW
        pixel_number_flat = binary_masks_flat.sum(1) # N
        mask_scores_flat = (mask_scores_flat * binary_masks_flat).sum(1) / torch.clamp(pixel_number_flat, min=1.0) # N

        reorder_score = (cls_scores ** reorder_class_weight) * (mask_scores_flat ** reorder_mask_weight) # N
        reorder_indices = torch.argsort(reorder_score, dim=-1, descending=True)
        
        # prepare ids
        thing_ids=list(self.metadata.thing_dataset_id_to_contiguous_id.values())
        stuff_ids=list(self.metadata.stuff_dataset_id_to_contiguous_id.values())
        
        id_cont_to_ids_dic = {}
        all_ids = sorted(thing_ids + stuff_ids)
        for ii, id_ in enumerate(all_ids):
            id_cont_to_ids_dic[ii] = id_

        # prepare midway results
        panoptic_seg = torch.zeros((mask_pred.shape[1], mask_pred.shape[2], mask_pred.shape[3]), dtype=torch.int32, device=mask_pred.device)
        segments_info = []

        # prepare final results
        panoptic_seg_mask = torch.ones((mask_pred.shape[1], mask_pred.shape[2], mask_pred.shape[3]), dtype=torch.int32, device=mask_pred.device) * (-1)
        dic_tmp = {}
        dic_cat_idemb = {}

        current_segment_id = 0
        stuff_memory_list = {}
        for i in range(num_mask_slots):
            cur_idx = reorder_indices[i].item() # 1
            cur_binary_mask = binary_masks[cur_idx] # T x H x W
            cur_mask_embedding = mask_embedding[cur_idx] # C
            cur_cls_score = cls_scores[cur_idx].item() # 1 
            cur_cls_label = cls_labels[cur_idx].item() # 1
            is_thing = cur_cls_label in thing_ids
            is_confident = (is_thing and cur_cls_score > cls_threshold_thing) or (
                (not is_thing) and cur_cls_score > cls_threshold_stuff)

            original_pixel_number = cur_binary_mask.float().sum()
            new_binary_mask = torch.logical_and(cur_binary_mask, (panoptic_seg == 0))
            new_pixel_number = new_binary_mask.float().sum()
            is_not_overlap_too_much = new_pixel_number > (original_pixel_number * overlap_threshold)

            if is_confident and is_not_overlap_too_much:
                # merge stuff regions
                if not is_thing:
                    if int(cur_cls_label) in stuff_memory_list.keys():
                        panoptic_seg[new_binary_mask] = stuff_memory_list[int(cur_cls_label)]
                        continue
                    else:
                        stuff_memory_list[int(cur_cls_label)] = current_segment_id + 1

                current_segment_id += 1                
                panoptic_seg[new_binary_mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(is_thing),
                        "category_id": int(cur_cls_label),
                    }
                )

                if is_thing:
                    cat_id_ = id_cont_to_ids_dic[int(cur_cls_label)]
                    if (cat_id_, True) not in dic_tmp:
                        dic_tmp[(cat_id_,True)] = []
                        dic_tmp[(cat_id_,True)].append((current_segment_id, cur_mask_embedding))
                    else:
                        if current_segment_id not in dic_tmp[(cat_id_,True)]:
                            dic_tmp[(cat_id_,True)].append((current_segment_id, cur_mask_embedding))
                else:
                    cat_id_ = id_cont_to_ids_dic[int(cur_cls_label)]
                    if (cat_id_,False) not in dic_tmp:
                        dic_tmp[(cat_id_,False)] = []
                        dic_tmp[(cat_id_,False)].append(current_segment_id)
                    else:
                        if current_segment_id not in dic_tmp[(cat_id_,False)]:
                            dic_tmp[(cat_id_,False)].append(current_segment_id)

        for tmp_, curr_seg_id_list in dic_tmp.items():
            cat_id_, isthing = tmp_
            if isthing:
                dic_cat_idemb[cat_id_] = []
                for ii, (cur_seg_id, id_emb) in enumerate(curr_seg_id_list):
                    new_id = cat_id_ * self.metadata.label_divisor + ii
                    panoptic_seg_mask[panoptic_seg==cur_seg_id] = new_id
                    dic_cat_idemb[cat_id_].append(F.normalize(id_emb, p=2, dim=0))
            else:
                for cur_seg_id in curr_seg_id_list:
                    panoptic_seg_mask[panoptic_seg==cur_seg_id] = cat_id_

        return panoptic_seg_mask, dic_cat_idemb