# MaXTron: Mask Transformer with Trajectory Attention for Video Panoptic Segmentation

This repo contains the code for our paper [**MaXTron: Mask Transformer with Trajectory Attention for Video Panoptic Segmentation**](https://arxiv.org/abs/2311.18537)

<div align="center">
  <img src="imgs/teaser.png" width="100%" height="100%"/>
</div><br/>

*MaXTron* is a simple yet effective unified meta-architecture for video segmentation, which enriches existing clip-level segmenters by introducing a within-clip tracking module and a cross-clip tracking module, thus achieving better temporally consistent segmentation results.

## Getting Started

For detailed usage of MaXTron, see

[MaXTron w/ Video-kMaX](MaXTron_Video-kMaX/README.md)

[MaXTron w/ Tube-Link](MaXTron_Tube-Link/README.md)

## Demos

![demo_0](imgs/0.gif)
![demo_1](imgs/1.gif)
![demo_2](imgs/2.gif)
![demo_3](imgs/3.gif)
![demo_4](imgs/4.gif)
![demo_5](imgs/5.gif)

## <a name="Citing MaXTron"></a>Citing  MaXTron

If you use MaXTron in your research, please use the following BibTeX entry.

```BibTeX
@misc{he2023maxtron,
      title={MaXTron: Mask Transformer with Trajectory Attention for Video Panoptic Segmentation}, 
      author={Ju He and Qihang Yu and Inkyu Shin and Xueqing Deng and Xiaohui Shen and Alan Yuille and Liang-Chieh Chen},
      year={2023},
      eprint={2311.18537},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement

[kMaX-DeepLab](https://github.com/bytedance/kmax-deeplab)

[Tube-Link](https://github.com/lxtGH/Tube-Link)