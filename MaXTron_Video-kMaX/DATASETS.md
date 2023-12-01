# Prepare Datasets for MaXTron w/ Video-kMaX

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

MaXTron w/ Video-kMaX has builtin support for VIPSeg.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  # panoptic datasets
  VIPSeg/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.


## Expected dataset structure for [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset):

```
VIPSeg/
  VIPSeg_720P/
    images/
    panomasks/
    panomasksRGB/
    panoptic_gt_VIPSeg.json
    panoptic_gt_VIPSeg_{train,val}.json
    panoVIPSeg_categories.json
    {train,val,test}.txt
```