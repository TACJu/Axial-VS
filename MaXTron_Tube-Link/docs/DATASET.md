Please prepare the data structure as the following instruction:

The final dataset folder should be like this. 
```
root 
├── data
│   ├──  youtube_vis_2021
│   ├──  youtube_vis_2022
│   ├──  ovis
```

### [VIS] Youtube-VIS-2021
We use pre-processed json file according to mmtracking codebase.
see the "tools/dataset/youtubevis2coco.py".

```
├── youtube_vis_2021
│   ├── annotations
│   │   ├── train.json
│   │   ├── valid.json
│   │   ├── youtube_vis_2021_train.json
│   │   ├── youtube_vis_2021_valid.json
│   ├── train
│   │   ├──JPEGImages
│   │   │   ├──video floders
│   ├── valid
│   │   ├──JPEGImages
│   │   │   ├──video floders
```

### [VIS] Youtube-VIS-2022
Follow the same procedure as Youtube-VIS-2021.

### [VIS] OVIS

Follow the same procedure as Youtube-VIS-2021.