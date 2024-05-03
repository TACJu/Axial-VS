## Getting Started with MaXTron w/ Video-kMaX

This document provides a brief intro of the usage of MaXTron w/ Video-kMaX.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

### Inference Demo with Pre-trained Models

1. Pick a model and its config file from [model zoo](MODEL_ZOO.md), for example, `configs/VIPSeg/panoptic-segmentation/maxtron_wc_r50.yaml`.
2. We provide `demo.py` that is able to demo builtin configs. Run it with:
```
cd demo_video/
python3 demo.py --config-file ../configs/VIPSeg/panoptic-segmentation/maxtron_wc_r50.yaml \
  --input video_folder/*.jpg --output visualization_folder \
  --opts MODEL.WEIGHTS /path/to/checkpoint_file
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.

### Training & Evaluation in Command Line

We provide script `train_net.py` and `train_net_video.py`, that are made to train all the configs provided in MaXTron_Video-kMaX.

The training of MaXTron_Video-kMaX can be divided into three steps:

> Pre-train kMaX-DeepLab + Multi-Scale Deformable Attention on COCO

To pre-train kMaX-DeepLab + MSDA, first setup the COCO dataset following kMaX-DeepLab,
then run:
```
python train_net.py --num-gpus 8 \
  --config-file configs/coco/panoptic-segmentation/kmax_wc_r50.yaml
```

> Train MaXTron w/ Video-kMaX near-online

To train a near-online MaXTron model with "train_net_video.py", first
setup the corresponding datasets following
[DATASETS.md](DATASETS.md),
then run:
```
python train_net_video.py --num-gpus 8 \
  --config-file configs/VIPSeg/panoptic-segmentation/maxtron_wc_r50.yaml \
  MODEL.WEIGHTS kmax_msda_r50.pth
```

The configs are made for 8-GPU training.
Since we use ADAMW optimizer, it is not clear how to scale learning rate with batch size.
To train on 1 GPU, you need to figure out learning rate and batch size by yourself:
```
python train_net_video.py \
  --config-file configs/VIPSeg/panoptic-segmentation/maxtron_wc_r50.yaml \
  --num-gpus 1 SOLVER.IMS_PER_BATCH SET_TO_SOME_REASONABLE_VALUE SOLVER.BASE_LR SET_TO_SOME_REASONABLE_VALUE \
  MODEL.WEIGHTS kmax_msda_r50.pth
```

> Train MaXTron w/ Video-kMaX offline

To train an offline MaXTron model with "train_net_video.py", first modify the keys of the trained near-online weights,
```
python prepare_cc_weights.py MaXTron_near-online.pth MaXTron_before_offline.pth
```

then run:
```
python train_net_video.py --num-gpus 8 \
  --config-file configs/VIPSeg/panoptic-segmentation/maxtron_cc_r50.yaml \
  MODEL.WEIGHTS kmax_before_offline.pth
```

> Evaluate MaXTron w/ Video-kMaX

To evaluate a model's performance, use
```
python train_net_video.py \
  --config-file configs/VIPSeg/panoptic-segmentation/maxtron_wc_r50.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file \
  MODEL.MAXTRON.TEST.INFERENCE_TYPE 'video-wise' \
  MODEL.MAXTRON.TEST.CLASS_THRESHOLD_THING SET_TO_SOME_REASONABLE_VALUE \
  MODEL.MAXTRON.TEST.CLASS_THRESHOLD_STUFF SET_TO_SOME_REASONABLE_VALUE \
  MODEL.MAXTRON.TEST.PIXEL_CONFIDENCE_THRESHOLD SET_TO_SOME_REASONABLE_VALUE
```
For more options, see `python train_net_video.py -h`.