# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/train_net.py
# Modified by Qihang Yu

try:
    # ignore ShapelyDeprecationWarning from fvcore
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import os
import copy
import itertools

from typing import Any, Dict, List, Set

import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    verify_results,
)
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

# kMaXDeeplab
from kmax_deeplab import (
    add_kmax_deeplab_config,
)
from maxtron_deeplab import (
    add_maxtron_deeplab_config,
    VIPSegPanopticMaXTronDatasetMapper,
    VIPSegEvaluator,
)

import train_net_utils


class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to MaskFormer.
    """
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if cfg.INPUT.DATASET_MAPPER_NAME == "vipseg_panoptic_mapper":
            detectron2_datasets_dir = os.getenv("DETECTRON2_DATASETS", "./datasets")
            truth_dir = os.path.join(detectron2_datasets_dir, 'VIPSeg/VIPSeg_720P/panomasksRGB')
            pan_gt_json_file = os.path.join(detectron2_datasets_dir, 'VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json')
            evaluator = VIPSegEvaluator(dataset_name, cfg.MODEL.MAXTRON.TEST.COST_LIMIT, cfg.MODEL.MAXTRON.TEST.MEM_WEIGHT, truth_dir=truth_dir, pan_gt_json_file=pan_gt_json_file, output_dir=output_folder)
        else:
            evaluator = None
        return evaluator


    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "vipseg_panoptic_mapper":
            mapper = VIPSegPanopticMaXTronDatasetMapper(cfg, True)
        else:
            mapper = None
        return build_detection_train_loader(cfg, mapper=mapper)


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        # Semantic segmentation dataset mapper
        dataset_name = cfg.DATASETS.TEST[0]
        if cfg.INPUT.DATASET_MAPPER_NAME == "vipseg_panoptic_mapper":
            mapper = VIPSegPanopticMaXTronDatasetMapper(cfg, False)
        else:
            mapper = None
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)
        

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        name = cfg.SOLVER.LR_SCHEDULER_NAME
        if name == "TF2WarmupPolyLR":
            return train_net_utils.TF2WarmupPolyLR(
                optimizer,
                cfg.SOLVER.MAX_ITER,
                warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
                warmup_iters=cfg.SOLVER.WARMUP_ITERS,
                warmup_method=cfg.SOLVER.WARMUP_METHOD,
                power=cfg.SOLVER.POLY_LR_POWER,
                constant_ending=cfg.SOLVER.POLY_LR_CONSTANT_ENDING,
            )
        else:
            return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        from kmax_deeplab.modeling.backbone.convnext import LayerNorm

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
            LayerNorm
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                hyperparams["name"] = (module_name, module_param_name)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "spatial_layers" in module_name or "level_embed_2d" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.SPATIAL_MULTIPLIER
                if "temporal_layers" in module_name or "level_embed_3d" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.TEMPORAL_MULTIPLIER
                if "_class_embedding_projection" in module_name or "_mask_embedding_projection" in module_name or "_transformer_mask_head" in module_name or "_transformer_class_head" in module_name or "_pixel_space_mask_batch_norm" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.PREDICTION_HEAD_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                # Rule for kMaX.
                if "_rpe" in module_name:
                    # relative positional embedding in axial attention.
                    hyperparams["weight_decay"] = 0.0
                if "_cluster_centers" in module_name:
                    # cluster center embeddings.
                    hyperparams["weight_decay"] = 0.0
                if "bias" in module_param_name:
                    # any bias terms.
                    hyperparams["weight_decay"] = 0.0
                if "gamma" in module_param_name:
                    # gamma term in convnext
                    hyperparams["weight_decay"] = 0.0

                params.append({"params": [value], **hyperparams})
        # for param_ in params:
        #     print(param_["name"], param_["lr"], param_["weight_decay"])

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        elif optimizer_type == "ADAM":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.Adam)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_kmax_deeplab_config(cfg)
    add_maxtron_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="MaXTron")
    return cfg


def main(args):
    cfg = setup(args)
    
    torch.backends.cudnn.enabled = True
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
