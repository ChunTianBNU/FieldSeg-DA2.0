# CNN-based agricultural parcels delineation Methods
## FieldSeg-DA2.0
An individual arable field(IAF) extraction framework.
![image](https://github.com/ChunTianBNU/FieldSeg-DA2.0/blob/master/imgs/FieldSeg-DA2.0.png)
### TODO
- [x] Support different convolutional neural networks for parcel delineation
- [x] GPU training
* The supported networks are as follows:
* |DeepLabv3+|[Encoder-decoder with atrous separable convolution for semantic image segmentation](https://arxiv.org/abs/1802.02611)|
* |UNet|[U-net: Convolutional networks for biomedical image segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)|
* |U-TAE|[Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks](https://github.com/VSainteuf/utae-paps)|
* |FieldSeg-DA|[A deep learning method for individual arable field (IAF) extraction with cross-domain adversarial capability](https://www.sciencedirect.com/science/article/pii/S0168169922007815)|

### Introduction
This is a PyTorch(1.12.0) implementation of **FieldSeg-DA2.0, an individual arable field (IAF) extraction network using multisource remote sensing and land cover data**. The paper aims to **enhance spatiotemporal transferability on the basis of the FieldSeg-DA framework**.Our paper has not been published yet.

### Installation
The code was tested with **Anaconda** and **Python 3.8.16**.

0. For **PyTorch** dependency, see [pytorch.org](https://pytorch.org/) for more details.
1. For **GDAL** dependency used for reading and writing raster data, use version 3.6.2.
