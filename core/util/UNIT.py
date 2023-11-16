# -*- coding:utf-8 -*-
'''
@Author:

@Date:

@Description:
    一些公用函数，如图像读取，输出
    by Shu
'''
from osgeo import gdal, osr
import numpy as np

GDAL2NP_CONVERSION = {
    1: "int8",
    2: "uint16",
    3: "uint16",
    4: "uint32",
    5: "int32",
    6: "float32",
    7: "float64",
    10: "complex64",
    11: "complex128"
}

NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}
def numpy2img(file_path: str, img: np.ndarray, proj=None, geot=None, type="GTiff"):
    '''
    :param file_path: Save url.
    :param img: Save numpy.ndarray image_process
    :param proj: Projection information
    :param geot: Geographic transfer information
    :return: None
    '''
    gdal_type = NP2GDAL_CONVERSION[img.dtype.name]
    # print(img.dtype.name)
    if len(img.shape) == 2:
        im_height, im_width = img.shape
        img = img.reshape(1, im_height, im_width)
    im_band, im_height, im_width = img.shape

    driver = gdal.GetDriverByName(type)

    dataset = driver.Create(file_path, im_width, im_height, im_band, gdal_type)
    if dataset is not None and proj is not None:
        dataset.SetGeoTransform(geot)  # 写入仿射变换参数
        dataset.SetProjection(proj)  # 写入投影

    for i in range(im_band):
        dataset.GetRasterBand(i + 1).WriteArray(img[i, :, :])

    del dataset


def img2numpy(file_path, geoinfo=False):
    dts = gdal.Open(file_path)
    proj = dts.GetProjection()
    geot = dts.GetGeoTransform()
    img = dts.ReadAsArray()
    if geoinfo:
        return img, proj, geot
    else:
        return img

    