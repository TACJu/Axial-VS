import copy
import lap
import itertools
import json
import logging
import numpy as np
import os
import io

from PIL import Image
from collections import OrderedDict
from panopticapi.utils import IdGenerator, rgb2id

import torch
import torch.nn.functional as F

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation import DatasetEvaluator

from .video_panoptic_metrics import vpq_compute_parallel


def generate_rgb_and_json(imgnames,img_tensors,meta,categories,save_folder,video_id):
    annotations = []

    categories = {el['id']: el for el in categories}
    color_generator = IdGenerator(categories)
    VOID = -1
    inst2color = {}

    if meta.name == 'cityscapes_vps_video_val' or meta.name =='cityscapes_vps_image_val' or meta.name == 'coco_2017_val_panoptic_with_sem_seg':
        div_ = 1000
    elif meta.name =='panoVSPW_vps_video_val':
        div_ = 125
    for imgname, img_tensor in zip(imgnames,img_tensors):
        # print('processing image :{} {}'.format(video_id,imgname))
        pan_format = np.zeros((img_tensor.shape[0], img_tensor.shape[1], 3), dtype=np.uint8)
        l = np.unique(img_tensor)
        segm_info = []
        for el in l:
            if el == VOID:
                continue
            mask = img_tensor == el

            if el < div_:
                if el in meta.thing_dataset_id_to_contiguous_id.values():
                    continue
                else:
                    sem = el
                    if el in inst2color:
                        color = inst2color[el]

                    else:
                        color = color_generator.get_color(sem)
                        inst2color[el] = color

            else:
                sem = int(el)//int(meta.label_divisor)

                if  sem not in meta.thing_dataset_id_to_contiguous_id.values():
                    print(l)
                    print('error')
                    print(sem)
                    print(el)
                    exit()
                if el in inst2color:
                    color = inst2color[el]
                else:
                    color = color_generator.get_color(sem)
                    inst2color[el] = color

            pan_format[mask] = color
            index = np.where(mask)
            x = index[1].min()
            y = index[0].min()
            width = index[1].max() - x
            height = index[0].max() - y

            dt = {"category_id": int(sem), "iscrowd": 0, "id": int(rgb2id(color)), "bbox": [x.item(), y.item(), width.item(), height.item()], "area": int(mask.sum())}
            segm_info.append(dt)
        
        with io.BytesIO() as out:
            Image.fromarray(pan_format).save(out, format="PNG")
            annotations.append({"segments_info": segm_info,"file_name":imgname,"png_string": out.getvalue()})

    return annotations


class VIPSegEvaluator(DatasetEvaluator):
    """
    Evaluate VPQ (STQ) for VIPSeg
    """

    def __init__(
        self,
        dataset_name,
        cost_limit,
        mem_weight,
        truth_dir,
        pan_gt_json_file,
        distributed=True,
        output_dir=None,
    ):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation
                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump all
                results predicted on the dataset. The dump contains two files:

                1. "instances_predictions.pth" a file in torch serialization
                   format that contains all the raw original predictions.
                2. "coco_instances_results.json" a json file in COCO's result
                   format.
        """
        self._logger = logging.getLogger(__name__)
        self._dataset = dataset_name
        self._cost_limit = cost_limit
        self._mem_weight = mem_weight
        self._truth_dir = truth_dir
        self._pan_gt_json_file = pan_gt_json_file
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        self._metadata = MetadataCatalog.get(dataset_name)

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            panoptic_imgs, segments_infos = output[0], output[1]
            final_preds = []
            reid_mem_dic = {}
            prediction = []
            video_id = input["video_id"][0]
            length = len(panoptic_imgs)
            if length == 1:
                final_preds.append(panoptic_imgs[0])
            else:
                for i in range(length):
                    pano_preds = panoptic_imgs[i]
                    dic_cat_idmask = segments_infos[i]
                    if len(dic_cat_idmask) == 0:
                        final_preds.append(pano_preds)
                        continue
                    else:
                        if len(reid_mem_dic) == 0:
                            reid_mem_dic = dic_cat_idmask
                            final_preds.append(pano_preds)
                            continue
                        else:
                            new_pano_preds = pano_preds.clone()
                            for cls_id in dic_cat_idmask:
                                if cls_id not in reid_mem_dic:
                                    reid_mem_dic[cls_id] = dic_cat_idmask[cls_id]
                                else:
                                    mem_feat = reid_mem_dic[cls_id]
                                    mem_feat = torch.stack(mem_feat,0) # NC
                                    cur_feat = dic_cat_idmask[cls_id]
                                    cur_feat =  torch.stack(cur_feat,0) # MC

                                    cos_dist = torch.matmul(cur_feat, mem_feat.t()) # MN
                                    cos_dist = 1. - (cos_dist+1.)/2
                                    _, x, _ = lap.lapjv(cos_dist.cpu().numpy(), extend_cost=True, cost_limit=self._cost_limit)
                                    matches, unmatched_a = [], []
                                    for ix, mx in enumerate(x):
                                        if mx >= 0:
                                            matches.append([ix, mx])
                                    unmatched_a = np.where(x < 0)[0]
                                    matches = np.asarray(matches)
                                    for matched in matches:
                                        cur_idx, mem_idx = matched
                                        point_id = cls_id * self._metadata.label_divisor + cur_idx
                                        ins_id = mem_idx
                                        new_id = cls_id * self._metadata.label_divisor + ins_id
                                        new_pano_preds[pano_preds==point_id] = new_id
                                        reid_mem_dic[cls_id][mem_idx] = reid_mem_dic[cls_id][mem_idx] * self._mem_weight + dic_cat_idmask[cls_id][cur_idx] * (1 - self._mem_weight)
                                        reid_mem_dic[cls_id][mem_idx] = F.normalize(reid_mem_dic[cls_id][mem_idx], p=2, dim=0)
                                    for unmatched_idx in unmatched_a:
                                        ins_id = len(reid_mem_dic[cls_id])
                                        reid_mem_dic[cls_id].append(dic_cat_idmask[cls_id][unmatched_idx])
                                        new_id = cls_id * self._metadata.label_divisor + ins_id
                                        point_id = cls_id * self._metadata.label_divisor + unmatched_idx
                                        new_pano_preds[pano_preds==point_id] = new_id
                            final_preds.append(new_pano_preds)

            final_preds = torch.cat(final_preds,0)
            final_preds_ = []                                                                                                                                                                                  
            imgnames_v_  = []                                                                                                                                                                                   
            for imgname, f_p in zip(input["file_names"], final_preds):                                                                                                                                                    
                if imgname not in imgnames_v_:                                                                                                                                                                 
                    imgnames_v_.append(imgname)                                                                                                                                                                
                    final_preds_.append(f_p)                                                                                                                                                                   
                                                                                                                                                                                                            
            final_preds_ = torch.stack(final_preds_,0)  

            categories = self._metadata.categories
            annos = generate_rgb_and_json(imgnames_v_, final_preds_.squeeze(1).cpu().numpy(), self._metadata, categories, self._output_dir, video_id)
            prediction.append({'annotations': annos, 'video_id': video_id})
        self._predictions.extend(prediction)

    def evaluate(self):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if not os.path.exists(os.path.join(self._output_dir,'pan_pred')):
            os.makedirs(os.path.join(self._output_dir,'pan_pred'))

        for p in predictions:
            anno = p['annotations']
            video_id = p['video_id']
            base_path = os.path.join(self._output_dir, 'pan_pred', video_id)
            if not os.path.exists(base_path):
                os.makedirs(base_path)    
            for img in anno:
                img_filename = img['file_name'].split('/')[-1].replace('.jpg', '.png')
                with open(os.path.join(base_path, img_filename), 'wb') as f:
                    f.write(img.pop('png_string'))

        pred_jsons = {'annotations': predictions}
        pan_gt_json_file = self._pan_gt_json_file
        with open(pan_gt_json_file, 'r') as f:
            gt_jsons = json.load(f)

        categories = gt_jsons['categories']
        categories = {el['id']: el for el in categories}

        pred_annos = pred_jsons['annotations']
        pred_j = {}
        for p_a in pred_annos:
            pred_j[p_a['video_id']] = p_a['annotations']
        gt_annos = gt_jsons['annotations']
        gt_j  ={}
        for g_a in gt_annos:
            gt_j[g_a['video_id']] = g_a['annotations']

        gt_pred_split = []

        for video_images in gt_jsons['videos']:        
            video_id = video_images['video_id']
            gt_image_jsons = video_images['images']
            gt_js = gt_j[video_id]
            pred_js = pred_j[video_id]

            assert len(gt_js) == len(pred_js)

            gt_pans =[]
            pred_pans = []
            for imgname_j in gt_image_jsons:
                imgname = imgname_j['file_name']

                pred_pans.append(os.path.join(self._output_dir, 'pan_pred', video_id, imgname))
                gt_pans.append(os.path.join(self._truth_dir, video_id, imgname))

            gt_pred_split.append(list(zip(gt_js,pred_js,gt_pans,pred_pans,gt_image_jsons)))

        res = {}
        vpq_all, vpq_thing, vpq_stuff = [], [], []

        # for k in [0,5,10,15] --> num_frames_w_gt [1,2,3,4]
        for nframes in [1, 2, 4, 6]:
            gt_pred_split_ = copy.deepcopy(gt_pred_split)
            vpq_all_, vpq_thing_, vpq_stuff_ = vpq_compute_parallel(
                    gt_pred_split_, categories, nframes, self._output_dir, 32)

            del gt_pred_split_
            self._logger.info('vpq-{}: {}, vpq-{}-thing: {}, vpq-{}-stuff: {}'.format(nframes, vpq_all_, nframes, vpq_thing_, nframes, vpq_stuff_))
            res["vpq-{}".format(nframes)] = vpq_all_
            vpq_all.append(vpq_all_)
            vpq_thing.append(vpq_thing_)
            vpq_stuff.append(vpq_stuff_)

        vpq_all_final = sum(vpq_all) / len(vpq_all)
        vpq_thing_final = sum(vpq_thing) / len(vpq_thing)
        vpq_stuff_final = sum(vpq_stuff) / len(vpq_stuff)
        self._logger.info('vpq: {}, vpq-thing: {}, vpq-stuff: {}'.format(vpq_all_final, vpq_thing_final, vpq_stuff_final))
        output_filename = os.path.join(self._output_dir, 'vpq-final.txt')
        output_file = open(output_filename, 'a+')
        output_file.write("vpq_all:%.4f\n"%(sum(vpq_all)/len(vpq_all)))
        output_file.write("vpq_thing:%.4f\n"%(sum(vpq_thing)/len(vpq_thing)))
        output_file.write("vpq_stuff:%.4f\n"%(sum(vpq_stuff)/len(vpq_stuff)))
        output_file.close()

        res["vpq"] = vpq_all_final
        res["vpq_thing"] = vpq_thing_final
        res["vpq_stuff"] = vpq_stuff_final
        res["vpq_1"] = vpq_all[0]
        res["vpq_2"] = vpq_all[1]
        res["vpq_4"] = vpq_all[2]
        res["vpq_6"] = vpq_all[3]

        results = OrderedDict({"panoptic_seg": res})
        return results