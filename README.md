<h1 align="center">HyperGaussians [CVPR 2026]<br><sub><sup>High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars</sup></sub></h1>

<div align="center">

[![Hugging Face](https://img.shields.io/badge/Project_Page-14161a?logo=homepage)](https://gserifi.github.io/HyperGaussians)&#160;
[![Technical Report](https://img.shields.io/badge/Paper_(arXiv)-b5212f?logo=arxiv)](https://arxiv.org/abs/2507.02803)
</div>

<div align="center">

[Gent Serifi](https://gserifi.github.io)
&nbsp;&nbsp;&nbsp;&nbsp;
[Marcel C. Buehler](https://mcbuehler.ch)

[ETH Zurich](https://ait.ethz.ch), Switzerland
</div>

![Teaer](assets/teaser.png)

This repository contains a plug-and-play implementation of **HyperGaussians**, which can be directly installed from this
repository for a frictionless integration into existing pipelines. We kindly refer to the original paper for a detailed
description of the method and its applications.

For real-world examples that use `HyperGaussians`, please refer to the `flash_avatar` and `gaussianheadavatar` branches in this
repository. They contain our modified versions of [FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code) and [GaussianHeadAvatar](https://github.com/YuelangX/Gaussian-Head-Avatar) with setup
and usage instructions.

## Get Started

Install package using pip:
```bash
pip install git+https://github.com/gserifi/HyperGaussians.git
```

Note that the package relies on an existing PyTorch installation.

## Usage

HyperGaussians can be imported and used just like any other `torch.nn.Module`. The constructor takes the number of
Gaussians, the latent dimension, and the output dimension as arguments. The output dimension is typically 3 for
position, 4 for rotation (quaternion), and 3 for scale. Depending on the application, the output values may need to be
passed through certain activation functions to map them to the desired range or format.

```python
import torch
from hypergaussians import HyperGaussians

# Initialization
num_gaussians = 10000
latent_dim = 8

hgs_xyz = HyperGaussians(num_gaussians, latent_dim, 3) # Position
hgs_rot = HyperGaussians(num_gaussians, latent_dim, 4) # Rotation
hgs_scl = HyperGaussians(num_gaussians, latent_dim, 3) # Scale

# Optionally specify initial output values
# initial_xyz = torch.randn(num_gaussians, 3)
# hgs_xyz = HyperGaussians(num_gaussians, latent_dim, 3, initial_output=initial_output)

# ...

# Conditioning

latent = torch.randn(1, num_gaussians, latent_dim)
deforms_xyz, uncertainty_xyz = hgs_xyz(latent)
deforms_rot, uncertainty_rot = hgs_rot(latent)
deforms_scl, uncertainty_scl = hgs_scl(latent)

# ... apply activation functions to restrict value range (e.g. using tanh or exp) or to convert to quaternions ...
```

## Citation
If you find HyperGaussians useful for your research, please consider citing our paper:

```bibtex
@inproceedings{Serifi2026HyperGaussians,
  title={HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars},
  author={Gent Serifi and Marcel C. Buehler},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
