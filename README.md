# CNN-based agricultural parcels delineation Methods

## FieldSeg-DA2.0：An individual arable field(IAF) extraction framework.
> [**FieldSeg-DA2.0: Further enhancing the spatiotemporal transferability of an individual arable field (IAF) extraction network using multisource remote sensing and land cover data.**](https://www.sciencedirect.com/science/article/pii/S0168169924004411)
> 
> Tian, C., Chen, X., Chen, J., Cao, R., & Liu, S. (2024)
> 
> Additionally, you can follow our team's homepage for more information: [Chen Lab Homepage](http://www.chen-lab.club/).
### TODO
- [x] Support different convolutional neural networks for agricultural parcels delineation
- [x] GPU training

* The supported networks are as follows:

|Method|Reference|
|:-:|:-:|
|FieldSeg-DA2.0|[FieldSeg-DA2.0: Further enhancing the spatiotemporal transferability of an individual arable field (IAF) extraction network using multisource remote sensing and land cover data](https://www.sciencedirect.com/science/article/abs/pii/S0168169924004411?fr=RR-2&ref=pdf_download&rr=88605234e8708485)|
|DeepLabv3+|[Encoder-decoder with atrous separable convolution for semantic image segmentation](https://arxiv.org/abs/1802.02611)|
|UNet|[U-net: Convolutional networks for biomedical image segmentation](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)|
|U-TAE|[Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks](https://openaccess.thecvf.com/content/ICCV2021/html/Garnot_Panoptic_Segmentation_of_Satellite_Image_Time_Series_With_Convolutional_Temporal_ICCV_2021_paper.html)|
|FieldSeg-DA|[A deep learning method for individual arable field (IAF) extraction with cross-domain adversarial capability](https://www.sciencedirect.com/science/article/pii/S0168169922007815)|
|ProDA|[Prototypical Pseudo Label Denoising and Target Structure Learning for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2101.10979)
|DUA|[The Norm Must Go On: Dynamic Unsupervised Domain Adaptation by Normalization](https://ieeexplore.ieee.org/document/9879821)


### Introduction
This is a PyTorch(1.12.0) implementation of **FieldSeg-DA2.0, an individual arable field (IAF) extraction network using multisource remote sensing and land cover data**. The paper aims to **enhance spatiotemporal transferability on the basis of the FieldSeg-DA framework**.Our paper has not been published yet.


### Prerequisites
- Python 3.6
- Pytorch 1.12.0
- numpy 1.24.3
- torchvision from master
- matplotlib
- GDAL
- OpenCV
- CUDA >= 11.0
- [albumentations](https://pypi.org/project/albumentations/)  1.3.1
### Train
You need to train **U-LSTM** first to extract IAF extents and **UNet** to extract IAF boundaries.
```
python train_src_IAFextent.py -cfg config\U-LSTMTrain.yaml
```
```
python train_src_IAFboundary.py -cfg config\UNetforBoundaryTrain.yaml
```
Next, you can proceed with the training of **FADA-A**.

```
python train_adv_IAFextent.py -cfg config\U-LSTMFADA-A.yaml
```
```
python train_adv_IAFboundary.py -cfg config\UNetforBoundaryFADA-A.yaml
```
If you need to modify the training parameters, you can go to the [config](https://github.com/ChunTianBNU/FieldSeg-DA2.0/tree/master/config) folder and edit the '.yaml' file inside.

### Others
If you need to train on your own data, you can use the code in [TrainData_generator.py](https://github.com/ChunTianBNU/FieldSeg-DA2.0/blob/master/core/util/TrainData_generator.py) for training set generation. 
If you want to replace the model's backbone, make adjustments in [core/model](https://github.com/ChunTianBNU/FieldSeg-DA2.0/tree/master/core/model)
Other image processing-related code can be found in [core/util/UNIT.py](https://github.com/ChunTianBNU/FieldSeg-DA2.0/blob/master/core/util/UNIT.py)

### Acknowledgement
* [DeepLab-V3-Plus](https://github.com/jfzhang95/pytorch-deeplab-xception)
* [UNet](https://github.com/milesial/Pytorch-UNet)
* [U-TAE](https://github.com/VSainteuf/utae-paps)
* [ProDA](https://github.com/microsoft/ProDA?tab=readme-ov-file#paper)
* [DUA](https://github.com/jmiemirza/DUA)

## Citation

If you find our method helpful for your work, please consider citing our paper:

Tian, C., Chen, X., Chen, J., Cao, R., & Liu, S. (2024). FieldSeg-DA2.0: Further enhancing the spatiotemporal transferability of an individual arable field (IAF) extraction network using multisource remote sensing and land cover data. *Computers and Electronics in Agriculture, 222*, 109050. [DOI: 10.1016/j.compag.2024.109050](https://doi.org/10.1016/j.compag.2024.109050)

