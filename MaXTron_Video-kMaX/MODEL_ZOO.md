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
<td align="center"><a href="https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/kmax_r50.pth">download</a>
</tr>
<!-- ROW: kMaX-DeepLab + MSDA R50 -->
<tr><td align="center">kMaX-DeepLab + MSDA</td>
<td align="center">R50</td>
<td align="center">53.6</td>
<td align="center"><a href="configs/coco/panoptic_segmentation/kmax_wc_r50.yaml">yaml</a></td>
<td align="center"><a href="[https://drive.google.com/file/d/14VV30We1QPUf5up2fnlQADA3jc4OtT14/view?usp=drive_link](https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/kmax_msda_r50.pth)">download</a>
</tr>
<!-- ROW: kMaX-DeepLab + MSDA ConvNeXt-L -->
<tr><td align="center">kMaX-DeepLab + MSDA</td>
<td align="center">ConvNeXt-L</td>
<td align="center">57.9</td>
<td align="center"><a href="configs/coco/panoptic_segmentation/kmax_wc_convnext_large.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/kmax_msda_convnextv1-l.pth">download</a>
</tr>
<!-- ROW: kMaX-DeepLab + MSDA ConvNeXtV2-L -->
<tr><td align="center">kMaX-DeepLab + MSDA</td>
<td align="center">ConvNeXtV2-L</td>
<td align="center">58.1</td>
<td align="center"><a href="configs/coco/panoptic_segmentation/kmax_wc_convnext_large.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/kmax_msda_convnextv2-l.pth">download</a>
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
<td align="center"><a href="https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/maxtron_wc_r50.pth">download</a>
</tr>
<!-- ROW: MaXTron CC R50 -->
<tr><td align="center">MaXTron offline</td>
<td align="center">R50</td>
<td align="center">46.7</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_cc_r50.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/maxtron_wc_convnextv1-l.pth">download</a>
</tr>
<!-- ROW: MaXTron WC ConvNeXt-L -->
<tr><td align="center">MaXTron near-online</td>
<td align="center">ConvNeXt-L</td>
<td align="center">56.2</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_wc_convnext_large.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/maxtron_wc_convnextv2-l.pth">download</a>
</tr>
<!-- ROW: MaXTron CC ConvNeXt-L -->
<tr><td align="center">MaXTron offline</td>
<td align="center">ConvNeXt-L</td>
<td align="center">57.1</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_cc_convnext_large.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/maxtron_cc_r50.pth">download</a>
</tr>
<!-- ROW: MaXTron WC ConvNeXtV2-L -->
<tr><td align="center">MaXTron near-online</td>
<td align="center">ConvNeXtV2-L</td>
<td align="center">57.6 (reproduced)</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_wc_convnextv2_large.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/maxtron_cc_convnextv1-l.pth">download</a>
</tr>
<!-- ROW: MaXTron CC ConvNeXtV2-L -->
<tr><td align="center">MaXTron offline</td>
<td align="center">ConvNeXtV2-L</td>
<td align="center">57.9 (reproduced)</td>
<td align="center"><a href="configs/VIPSeg/panoptic_segmentation/maxtron_cc_convnextv2_large.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/turkeyju/Axial-VS/blob/main/Axial-VS-VPS/maxtron_cc_convnextv2-l.pth">download</a>
</tr>
<table><tbody>
