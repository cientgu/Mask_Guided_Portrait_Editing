# Mask-Guided Portrait Editing with Conditional GANs
This is an official pytorch implementation of "Mask-Guided Portrait Editing with Conditional GANs"(CVPR2019). The major contributors of this repository include Shuyang Gu, Jianmin Bao, Hao Yang, Dong Chen, Fang Wen, Lu Yuan at Microsoft Research.

## Introduction

**Mask-Guided Portrait Editing** is a novel technology based on mask-guided condititonal GANs, which can synthesize diverse, high-quality and controllable facial images from given masks. With the changeable input facial mask and source image, this method allows users to do high-level portrait editing.

## Citation
If you find our code  helpful for your research, please consider citing:
```
@inproceedings{gu2019mask,
  title={Mask-Guided Portrait Editing With Conditional GANs},
  author={Gu, Shuyang and Bao, Jianmin and Yang, Hao and Chen, Dong and Wen, Fang and Yuan, Lu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3436--3445},
  year={2019}
} 
```

## Getting Started

### Prerequisite
- Linux.
- Pytorch 0.4.1.
- Nvidia GPU: K40, M40, P100.
- CUDA9.2 or 10.

### Running code
- download pretrained models [here](https://drive.google.com/open?id=1MR8xV3NSUOV0Xx6hj1FHH3bkTmTUsKFg), put it under folder checkpoints/pretrained .
- component editing:
  ./scripts/test_edit.sh
- component transfer:
  ./scripts/test_edit_free_encode.sh 
  change the corresponding component file in results/pretrained/editfree_latest, then run:
  ./scripts/test_edit_free_generate.sh
  get the component transfer results.
- training:
  ./scripts/train.sh

