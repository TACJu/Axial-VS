# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py
# modified by Ju He
import os
import copy
import logging

import numpy as np
import torch
import random

from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Boxes, Instances

__all__ = ["VIPSegPanoptickMaXDeepLabDatasetMapper"]


def build_transform_gen(cfg, is_train, scale_ratio=1.0):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE
    # assert is_train

    min_scale = cfg.INPUT.MIN_SCALE * scale_ratio
    max_scale = cfg.INPUT.MAX_SCALE * scale_ratio

    # Augmnetation order majorlly follows deeplab2: resize -> autoaug (color jitter) -> random pad/crop -> flip
    # But we alter it to  resize -> color jitter -> flip -> pad/crop, as random pad is not supported in detectron2.
    # The order of flip and pad/crop does not matter as we are doing random padding/crop anyway.
    if is_train:
        augmentation = [
            # Unlike deeplab2 in tf, here the interp will be done in uin8 instead of float32.
            T.ResizeScale(
                min_scale=min_scale, max_scale=max_scale, target_height=image_size[0], target_width=image_size[1]
            ), # perofrm on uint8 or float32
            ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT), # performed on uint8
            # Unlike deeplab2 in tf, here the padding value for image is 128, for label is 255. Besides, padding here will only pad right and bottom.
            # T.FixedSizeCrop(crop_size=(image_size, image_size)),

            # We only perform crop, and do padding manually as the padding value matters. This will crop the image to min(h, image_size).
            T.RandomCrop(crop_type="absolute", crop_size=(image_size[0], image_size[1])),
            T.RandomFlip(),
        ]
    else:
        augmentation = []

    return augmentation


class VIPSegPanopticMaXTronDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by kMaX-DeepLab.

    This dataset mapper applies the same transformation as DETR for VIPSeg panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        tfm_gens_copy_paste,
        image_format,
        image_size,
        # video
        train_clip_length,
        random_reverse,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            tfm_gens_copy_paste: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        self.tfm_gens_copy_paste = tfm_gens_copy_paste
        if is_train:
            logging.getLogger(__name__).info(
                "[VIPSegPanopticDeepLab2DatasetMapper] Full TransformGens used in training: {}, {}".format(
                    str(self.tfm_gens), str(self.tfm_gens_copy_paste)
                )
            )
        else:
            logging.getLogger(__name__).info(
                "[VIPSegPanopticDeepLab2DatasetMapper] Full TransformGens used in testing: {}".format(
                    str(self.tfm_gens)
                )
            )
        self.img_format = image_format
        self.is_train = is_train
        self.image_size = image_size

        # video
        self.train_clip_length = train_clip_length
        self.random_reverse = random_reverse

        detectron2_datasets_dir = os.getenv("DETECTRON2_DATASETS", "./datasets")
        image_dir = os.path.join(detectron2_datasets_dir, "VIPSeg/VIPSeg_720P/images")
        gt_dir = os.path.join(detectron2_datasets_dir, "VIPSeg/VIPSeg_720P/panomasksRGB")
        json_file = os.path.join(detectron2_datasets_dir, "VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_train.json")

        from ..datasets import register_panovspw_vps_video
        self.dataset_dict_all = register_panovspw_vps_video.load_video_vspw_vps_json(
            json_file, image_dir, gt_dir
        )
        self.filename2idx = {}
        self.videoid2idx = {}
        for video_idx, video_dataset_dict in enumerate(self.dataset_dict_all):
            video_id = video_dataset_dict["video_id"]
            self.videoid2idx[video_id] = video_idx
            self.filename2idx[video_id] = {}
            for idx, image_file_name in enumerate(video_dataset_dict["file_names"]):
                self.filename2idx[video_id][image_file_name.split('/')[-1].replace('.jpg', '')] = idx


    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        tfm_gens_copy_paste = build_transform_gen(cfg, is_train, scale_ratio=0.5)
        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "tfm_gens_copy_paste": tfm_gens_copy_paste,
            "image_format": cfg.INPUT.FORMAT,
            "image_size": cfg.INPUT.IMAGE_SIZE,
            # video
            "train_clip_length": cfg.INPUT.NUM_VIDEO_FRAMES,
            "random_reverse": cfg.INPUT.RANDOM_REVERSE,
        }
        return ret

    def read_dataset_dict(self, dataset_dict, is_copy_paste=False):
        """
        Args:
            dataset_dict (dict): Metadata of one clip, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict_list = []
        pan_seg_gt_list = []

        for idx in range(len(dataset_dict)):
            current_dataset_dict = copy.deepcopy(dataset_dict[idx])
            image = utils.read_image(current_dataset_dict["file_name"], format=self.img_format)
            utils.check_image_size(current_dataset_dict, image)

            if self.is_train:
                if idx == 0:
                    if not is_copy_paste:
                        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
                    else:
                        image, transforms = T.apply_transform_gens(self.tfm_gens_copy_paste, image)
                else:
                    image = transforms.apply_image(image)

            current_dataset_dict["image"] = np.ascontiguousarray(image.transpose(2, 0, 1))

            if not self.is_train:
                # If this is for test, we can directly return the unpadded image, as the padding
                # will be handled by size_divisibility
                current_dataset_dict.pop("annotations", None)
                dataset_dict_list.append(current_dataset_dict)
                pan_seg_gt_list = None
            else:
                # We pad the image manually, for copy-paste purpose.
                padded_image = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=current_dataset_dict["image"].dtype)
                new_h, new_w = current_dataset_dict["image"].shape[1:]
                offset_h, offset_w = 0, 0 # following the d2 panoptic deeplab implementaiton to only perform bottom/right padding.
                padded_image[:, offset_h:offset_h+new_h, offset_w:offset_w+new_w] = current_dataset_dict["image"]
                current_dataset_dict["image"] = padded_image

                pan_seg_gt = utils.read_image(current_dataset_dict.pop("pan_seg_file_name"), "RGB")

                # apply the same transformation to panoptic segmentation
                pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

                pan_seg_gt = rgb2id(pan_seg_gt) # int32 # H x W
                # similarily, we manually pad the label, and we use label -1 to indicate those padded pixels.
                # In this way, we can masking out the padded pixels values to -1 after normalization, which aligns the
                # behavior between training and testing.
                padded_pan_seg_gt = np.zeros((self.image_size[0], self.image_size[1]), dtype=pan_seg_gt.dtype)
                is_real_pixels = np.zeros((self.image_size[0], self.image_size[1]), dtype=bool)
                padded_pan_seg_gt[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = pan_seg_gt
                is_real_pixels[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = True
                current_dataset_dict["is_real_pixels"] = is_real_pixels
                pan_seg_gt = padded_pan_seg_gt

                dataset_dict_list.append(current_dataset_dict)
                pan_seg_gt_list.append(pan_seg_gt)
            
        return dataset_dict_list, pan_seg_gt_list

    def call_copypaste(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # Read clip frames.
        dataset_dict_list, pan_seg_gt_list = self.read_dataset_dict(dataset_dict, is_copy_paste=False)

        if not self.is_train:
            video_dict = {}

            # update the video_dict with keys needed during training or inference
            video_dict["image"] = []
            video_dict["file_names"] = []
            video_dict["video_id"] = []
            video_dict["height"] = []
            video_dict["width"] = []
            video_dict["is_real_pixels"] = []

            for dataset_dict_each in dataset_dict_list:
                video_dict["file_names"].append(dataset_dict_each["file_name"])
                video_dict["video_id"].append(dataset_dict_each["video_id"])
                video_dict["height"].append(dataset_dict_each["height"])
                video_dict["width"].append(dataset_dict_each["width"])
                image = torch.as_tensor(dataset_dict_each["image"])
                video_dict["image"].append(image)
                video_dict["is_real_pixels"].append(torch.ones_like(image).bool())

            return video_dict

        # Read copy-paste image.
        # It should be sometinng like xxx/xxx/xxx/000000139.jpg, etc. we use the last number as a bias to random number.
        # Currently only sample from the same video
        copy_paste_video_idx = random.randint(0, len(self.dataset_dict_all) - 1)
        copy_paste_dataset_dict = self.dataset_dict_all[copy_paste_video_idx]
        copy_paste_video_length = len(copy_paste_dataset_dict["file_names"])
        if self.train_clip_length < copy_paste_video_length:
            copy_paste_start_idx = random.randrange(copy_paste_video_length - self.train_clip_length)
            copy_paste_index_list = [copy_paste_start_idx + _ for _ in range(self.train_clip_length)]
        else:
            copy_paste_index_list = list(range(copy_paste_video_length))
            copy_paste_index_list += [copy_paste_index_list[-1] for _ in range(self.train_clip_length - len(copy_paste_index_list))]
        if self.random_reverse and random.random() < 0.5:
            copy_paste_index_list = copy_paste_index_list[::-1]
        
        copy_paste_video_dataset_dict = []
        for idx in copy_paste_index_list:
            img_dataset_dict = {}
            img_dataset_dict["file_name"] = copy_paste_dataset_dict["file_names"][idx]
            img_dataset_dict["width"] = copy_paste_dataset_dict["width"]
            img_dataset_dict["height"] = copy_paste_dataset_dict["height"]
            img_dataset_dict["video_id"] = copy_paste_dataset_dict["video_id"]
            img_dataset_dict["pan_seg_file_name"] = copy_paste_dataset_dict["pan_seg_file_names"][idx]
            img_dataset_dict["segments_info"] = copy_paste_dataset_dict["segments_infos"][idx]
            copy_paste_video_dataset_dict.append(img_dataset_dict)

        dataset_dict_list_copy_paste, pan_seg_gt_list_copy_paste = self.read_dataset_dict(copy_paste_video_dataset_dict, is_copy_paste=True)

        # Choose copy-paste id from the first frame
        all_ids = []
        thing_ids = []
        clip_copy_label_list = []
        segments_info_copy_paste = dataset_dict_list_copy_paste[0]["segments_info"]
        for segment_info_copy_paste in segments_info_copy_paste:
            if not segment_info_copy_paste["iscrowd"]:
                # -1 is reserved for padded pixels.
                if segment_info_copy_paste["id"] in [-1, 0]:
                    print(segment_info_copy_paste)
                    raise ValueError("id should not be -1, 0")
                all_ids.append(segment_info_copy_paste["id"])
                if segment_info_copy_paste["isthing"]: # All thing classes are copy-pasted.
                    thing_ids.append(segment_info_copy_paste["id"])

        # Shuffle and randomly select kept label ids.
        random.shuffle(all_ids)
        keep_number = random.randint(0, len(all_ids))
        for index, label_id in enumerate(all_ids):
            # randomly copy labels, but keep all thing classes.
            if index < keep_number or label_id in thing_ids:
                clip_copy_label_list.append(label_id)

        # prepare all possible ids
        _ids = set()
        for curr_dataset_dict in dataset_dict_list:
            for segment_info in curr_dataset_dict["segments_info"]:
                _ids.add(segment_info["id"])
        for copy_paste_id in clip_copy_label_list:
            _ids.add(copy_paste_id)
        merged_ids = dict()
        for i, _id in enumerate(_ids):
            merged_ids[_id] = i
        ids_length = len(merged_ids)

        video_dataset_dict = {}
        video_dataset_dict["video_id"] = dataset_dict_list[0]["video_id"]
        video_dataset_dict["width"] = dataset_dict_list[0]["width"]
        video_dataset_dict["height"] = dataset_dict_list[0]["height"]
        video_dataset_dict["image"] = []
        video_dataset_dict["is_real_pixels"] = []
        video_dataset_dict["instances"] = []
        video_dataset_dict["sem_seg_gt"] = []

        for idx in range(len(dataset_dict_list)):
            # Copy data_dict_copy_paste onto data_dict. 0 means keep original pixel, 1 means use copy-paste pixel.
            copy_paste_masks = np.zeros((pan_seg_gt_list[idx].shape[-2], pan_seg_gt_list[idx].shape[-1]))
            segments_info_copy_paste = dataset_dict_list_copy_paste[idx]["segments_info"]

            for label_id in clip_copy_label_list:
                copy_paste_masks[pan_seg_gt_list_copy_paste[idx] == label_id] = 1

            # random probability to copy-paste. (probability is 0.50)
            if random.random() < 0.0:
                copy_paste_masks = np.zeros_like(copy_paste_masks)

            # We merge the image and copy-paste image based on the copy-paste mask.
            # 3 x H x W
            merged_image = (dataset_dict_list[idx]["image"] * (1.0 - copy_paste_masks).astype(dataset_dict_list[idx]["image"].dtype) +
                                    dataset_dict_list_copy_paste[idx]["image"] * copy_paste_masks.astype(dataset_dict_list[idx]["image"].dtype))
            video_dataset_dict["image"].append(torch.as_tensor(merged_image))
            merged_is_real_pixels = (dataset_dict_list[idx]["is_real_pixels"] * (1.0 - copy_paste_masks).astype(dataset_dict_list[idx]["is_real_pixels"].dtype) +
                                    dataset_dict_list_copy_paste[idx]["is_real_pixels"] * copy_paste_masks.astype(dataset_dict_list[idx]["is_real_pixels"].dtype))
            video_dataset_dict["is_real_pixels"].append(torch.as_tensor(merged_is_real_pixels))
            # We set all ids in copy-paste image to be negative, so that there will be no overlap between original id and copy-paste id.
            pan_seg_gt_copy_paste = -pan_seg_gt_list_copy_paste[idx]
            pan_seg_gt = pan_seg_gt_list[idx]
            pan_seg_gt = (pan_seg_gt * (1.0 - copy_paste_masks).astype(pan_seg_gt.dtype) +
                        pan_seg_gt_copy_paste * copy_paste_masks.astype(pan_seg_gt.dtype))

            # We use 4x downsampled gt for final supervision.
            pan_seg_gt = pan_seg_gt[::4, ::4]

            sem_seg_gt = -np.ones_like(pan_seg_gt) # H x W, init with -1

            # We create sorted_classes & sorted_maskes & sorted_ids for padding instances
            sorted_classes = [-1 for _ in range(ids_length)]
            sorted_masks = [np.zeros_like(pan_seg_gt) for _ in range(ids_length)]
            sorted_ids = [-1 for _ in range(ids_length)]

            # We then process the obtained pan_seg_gt to training format.
            image_shape = dataset_dict_list[idx]["image"].shape[1:]  # h, w
            segments_info = dataset_dict_list[idx]["segments_info"]
            instances = Instances(image_shape)
            classes = []
            masks = []
            ids = []
            # As the two images may share same stuff classes, we use a dict to track existing stuff and merge them.
            stuff_class_to_idx = {}
            for segment_info in segments_info:
                class_id = segment_info["category_id"]
                if not segment_info["iscrowd"]:
                    # -1 is reserved to indicate padded pixels.
                    if segment_info["id"] in [-1, 0]:
                        print(segment_info)
                        raise ValueError("id should not be -1, 0")
                    binary_mask = (pan_seg_gt == segment_info["id"])
                    # As it is possible that some masks are removed during the copy-paste process, we need
                    # to double check if the maks exists.
                    valid_pixel_num_ = binary_mask.sum()
                    if valid_pixel_num_ > 0:
                        sem_seg_gt[binary_mask] = class_id      
                        if not segment_info["isthing"]:
                            # For original image, stuff should only appear once.
                            if class_id in stuff_class_to_idx:
                                raise ValueError('class_id should not already be in stuff_class_to_idx!')
                            else:
                                stuff_class_to_idx[class_id] = len(masks)
                        classes.append(class_id)
                        masks.append(binary_mask)
                        ids.append(segment_info["id"])

            for segment_info in segments_info_copy_paste:
                if segment_info["id"] in clip_copy_label_list:
                    class_id = segment_info["category_id"]
                    if not segment_info["iscrowd"]:
                        # -1 is reserved to indicate padded pixels.
                        if segment_info["id"] in [-1, 0]:
                            print(segment_info)
                            raise ValueError("id should not be -1, 0")
                        # Note that copy-paste id is negative.
                        binary_mask = (pan_seg_gt == -segment_info["id"])
                        valid_pixel_num_ = binary_mask.sum()
                        if valid_pixel_num_ > 0:
                            sem_seg_gt[binary_mask] = class_id
                            if not segment_info["isthing"]:
                                # The stuff in copy-paste image already appeared in original image.
                                if class_id in stuff_class_to_idx:
                                    # Merge into original stuff masks. 
                                    masks[stuff_class_to_idx[class_id]] = np.logical_or(masks[stuff_class_to_idx[class_id]], binary_mask)
                                    continue
                                else:
                                    stuff_class_to_idx[class_id] = len(masks)
                            classes.append(class_id)
                            masks.append(binary_mask)
                            ids.append(segment_info["id"])

            # We set class & mask & ids to their corresponding positions, there is one question: the copy_paste frame might have new ids? how do we handle this?
            for anno_idx, id in enumerate(ids):
                id_idx = merged_ids[id]
                sorted_classes[id_idx] = classes[anno_idx]
                sorted_masks[id_idx] = masks[anno_idx]
                sorted_ids[id_idx] = ids[anno_idx]

            sem_seg_gt = torch.tensor(sem_seg_gt, dtype=torch.int64)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored), pad to ids_length
                instances.gt_classes = torch.ones(ids_length, dtype=torch.int64) * -1
                instances.gt_masks = torch.zeros((ids_length, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
                instances.gt_boxes = Boxes(torch.zeros((ids_length, 4)))
                instances.gt_ids = torch.ones(ids_length, dtype=torch.int64) * -1
            else:
                instances.gt_classes = torch.tensor(sorted_classes, dtype=torch.int64)
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in sorted_masks])
                )
                instances.gt_masks = masks.tensor
                instances.gt_boxes = masks.get_bounding_boxes()
                instances.gt_ids = torch.tensor(sorted_ids)

            video_dataset_dict["instances"].append(instances)
            video_dataset_dict["sem_seg_gt"].append(sem_seg_gt)

        return video_dataset_dict

    def call_video_copypaste(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # read video length
        video_length = len(dataset_dict["file_names"])

        if self.is_train:
            if self.train_clip_length < video_length:
                start_idx = random.randrange(video_length - self.train_clip_length)
                index_list = [start_idx + _ for _ in range(self.train_clip_length)]
            else:
                index_list = list(range(video_length))
                index_list += [index_list[-1] for _ in range(self.train_clip_length - video_length)]
            if self.random_reverse and random.random() < 0.5:
                index_list = index_list[::-1]
        else:
            index_list = list(range(video_length))

        video_dataset_dict = []
        for idx in index_list:
            img_dataset_dict = {}
            img_dataset_dict["width"] = dataset_dict["width"]
            img_dataset_dict["height"] = dataset_dict["height"]
            img_dataset_dict["video_id"] = dataset_dict["video_id"]
            img_dataset_dict["file_name"] = dataset_dict["file_names"][idx]
            img_dataset_dict["pan_seg_file_name"] = dataset_dict["pan_seg_file_names"][idx]
            img_dataset_dict["segments_info"] = dataset_dict["segments_infos"][idx]
            video_dataset_dict.append(img_dataset_dict)

        video_dataset_dict = self.call_copypaste(video_dataset_dict)

        return video_dataset_dict

    def __call__(self, dataset_dict):
        res = self.call_video_copypaste(dataset_dict)

        return res
