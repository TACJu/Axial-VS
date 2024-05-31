# MaXTron w/ Video-kMaX Model Zoo

### COCO Pre-trained Weights

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<td align="center">Model</th>
<td align="center">Backbone</th>
<td align="center">PQ</th>
<td align="center">Config</th>
<td align="center">ckpt</th>
<!-- TABLE BODY -->
<!-- ROW: kMaX-DeepLab R50 -->
<tr><td align="center">kMaX-DeepLab</td>
<td align="center">R50</td>
<td align="center">53.3</td>
<td align="center"><a href="https://github.com/bytedance/kmax-deeplab/blob/main/configs/coco/panoptic_segmentation/kmax_r50.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1xjYtj0X1Yt35RXSF_BbMMuN8bOcLSEAB/view?usp=drive_link">download</a>
</tr>
<!-- ROW: kMaX-DeepLab + MSDA R50 -->
<tr><td align="center">kMaX-DeepLab + MSDA</td>
<td align="center">R50</td>
<td align="center">53.6</td>
<td align="center"><a href="configs/coco/panoptic_segmentation/kmax_wc_r50.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/14VV30We1QPUf5up2fnlQADA3jc4OtT14/view?usp=drive_link">download</a>
</tr>
<!-- ROW: kMaX-DeepLab + MSDA ConvNeXt-L -->
<tr><td align="center">kMaX-DeepLab + MSDA</td>
<td align="center">ConvNeXt-L</td>
<td align="center">57.9</td>
<td align="center"><a href="configs/coco/panoptic_segmentation/kmax_wc_convnext_large.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1VbQDovawqSLELhF7u9U8j9YLrRmsUYKS/view?usp=drive_link">download</a>
</tr>
<!-- ROW: kMaX-DeepLab + MSDA ConvNeXtV2-L -->
<tr><td align="center">kMaX-DeepLab + MSDA</td>
<td align="center">ConvNeXtV2-L</td>
<td align="center">58.1</td>
<td align="center"><a href="configs/coco/panoptic_segmentation/kmax_wc_convnext_large.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1e6U1MF3NyI-f4Cd9wn2E7DX2nVNyYSP_/view?usp=drive_link">download</a>
</tr>
<table><tbody>

### VIPSeg

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<td align="center">Model</th>
<td align="center">Backbone</th>
<td align="center">VPQ</th>
<td align="center">Config</th>
<td align="center">ckpt</th>
<!-- TABLE BODY -->
<!-- ROW: MaXTron WC R50 -->
<tr><td align="center">MaXTron near-online</td>
<td align="center">R50</td>
<td align="center">46.1</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_wc_r50.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1Lfr-FBuRqgqPUmUr6hCmK7ddgWt5OngH/view?usp=drive_link">download</a>
</tr>
<!-- ROW: MaXTron CC R50 -->
<tr><td align="center">MaXTron offline</td>
<td align="center">R50</td>
<td align="center">46.7</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_cc_r50.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1MQVj1j70uE2ifJ6YREPU15r8ug2lrnUj/view?usp=drive_link">download</a>
</tr>
<!-- ROW: MaXTron WC ConvNeXt-L -->
<tr><td align="center">MaXTron near-online</td>
<td align="center">ConvNeXt-L</td>
<td align="center">56.2</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_wc_convnext_large.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1fFyTyUAPSE57fqzoy4JzlTGUkIgPf8bU/view?usp=drive_link">download</a>
</tr>
<!-- ROW: MaXTron CC ConvNeXt-L -->
<tr><td align="center">MaXTron offline</td>
<td align="center">ConvNeXt-L</td>
<td align="center">57.1</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_cc_convnext_large.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1plbJPJOi9OmOHlcQ9tp2S9mfoOVUlBiv/view?usp=sharing">download</a>
</tr>
<!-- ROW: MaXTron WC ConvNeXtV2-L -->
<tr><td align="center">MaXTron near-online</td>
<td align="center">ConvNeXtV2-L</td>
<td align="center">57.7</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_wc_convnextv2_large.yaml">yaml</a></td>
<td align="center">TBA
</tr>
<!-- ROW: MaXTron CC ConvNeXtV2-L -->
<tr><td align="center">MaXTron offline</td>
<td align="center">ConvNeXtV2-L</td>
<td align="center">58.0</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_cc_convnextv2_large.yaml">yaml</a></td>
<td align="center">TBA
</tr>
<table><tbody>