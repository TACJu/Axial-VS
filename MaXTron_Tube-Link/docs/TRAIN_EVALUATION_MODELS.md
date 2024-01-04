## Pretrained Models

Please download the pre-trained Mask2Former model from MMDetection official website

or they will be downloaded automatically into your .cache folder.


## Training and Inference Scripts

### [VIS-Youtube-2021/2022]
Train MaXTron WC Swin Large model
```bash
GPUS=16 bash tools/slurm_train.sh $PARTITION job_name configs/video/ytvis21/ytvis21_swin_l_maxtron_wc_2_5k_5k_10k.py --work-dir ./your_path/ --no-validate
```

Train MaXTron CC Swin Large model
```bash
GPUS=16 bash tools/slurm_train.sh $PARTITION job_name configs/video/ytvis21/ytvis21_swin_l_maxtron_cc_2_2.5k_7.5k_10k.py --work-dir ./your_path/ --no-validate
```

Inference the model for submission.
```bash
GPUS=8 bash tools/slurm_test.py $PARTITION job_name configs/video/ytvis21/ytvis21_swin_l_maxtron_cc_2_2.5k_7.5k_10k.py ./your_path_to_trained_model.pth --format-only --eval-options resfile_path=/path/to/submission
```

### [OVIS]
Train MaXTron WC Swin Large model
```bash
GPUS=16 bash tools/slurm_train.sh $PARTITION job_name configs/video/ovis/ovis_swin_l_maxtron_wc_2_5k_10k_15k.py --work-dir ./your_path/ --no-validate
```

Train MaXTron CC Swin Large model
```bash
GPUS=16 bash tools/slurm_train.sh $PARTITION job_name configs/video/ovis/ovis_swin_l_maxtron_cc_2_2.5k_7.5k_10k.py --work-dir ./your_path/ --no-validate
```

Inference the model for submission.
```bash
GPUS=8 bash tools/slurm_test.py $PARTITION job_name configs/video/ovis/ovis_swin_l_maxtron_cc_2_2.5k_7.5k_10k.py ./your_path_to_trained_model.pth --format-only --eval-options resfile_path=/path/to/submission
```