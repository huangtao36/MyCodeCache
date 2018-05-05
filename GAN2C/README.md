
# GAN2C in PyTorch

This is our implementation for GAN2C: Information Completion GAN with Dual Consistency Constraints.


## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries [dominate](https://github.com/Knio/dominate).
```bash
pip install dominate
```
- Clone this repository

### GAN2C train/test
- dataset folder structure will look like this::
```dir

--dataset  
&emsp;&emsp;|--trainA  
&emsp;&emsp;|--trainB  
&emsp;&emsp;|--testA  
&emsp;&emsp;|--testB  
```

- Train a model:
```bash
python train.py
```


- Test a model:
```bash
python test.py
```


## Citation
If you use this code for your research, please cite our papers.


## Acknowledgments
Code is inspired by [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
