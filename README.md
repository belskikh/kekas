# Kekas

![](imgs/logo.png)

Kek it easy.

Kekas is a simple tool for training neural networks on Pytorch.

I tried to keep it as simple as possible.

# Installation

`pip install kekas`

## Mixed Precision requirements

Kekas use https://github.com/NVIDIA/apex library for mixed precision training, so follow the installation instructions from its repo.

# Features

- Awesome name
- Mixed Precision (FP16)
- Learning Rate Finder
- One Cycle policy
- Best checkpoints saving
- Early stopping
- TTA
- Freeze / unfreeze
- Easy customization
    
# Quick start guide

I don't beieve in quick start guides, I think that they create more questions 
than answers.

Instead, I've created a detailed [Tutorial notebook](Tutorial.ipynb). Read it.

# Contribution guide

Just contribute something good, and don't contribute anything bad.

# TODO

- [ ] AdamW
- [x] FP16 support 
- [ ] Discriminative learning rates
- [ ] Docs :kekeke:

# Citing
If you find this library useful for your research, please consider citing:
```
@misc{aleksandr belskikh_2019,
    author      = {Aleksandr Belskikh},
    title       = {{kekas: Just another DL library}},
    month       = feb,
    year        = 2019,
    doi         = {10.5281/zenodo.2577861},
    version     = {0.1.17},
    publisher   = {Zenodo}, 
    url         = {https://doi.org/10.5281/zenodo.2577861},
    }

```

[![DOI](https://zenodo.org/badge/144457787.svg)](https://zenodo.org/badge/latestdoi/144457787)
