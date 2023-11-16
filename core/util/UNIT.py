# -*- coding:utf-8 -*-
'''
@Author:

@Date:

@Description:
    一些公用函数，如图像读取，输出
'''

from osgeo import gdal, osr
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


def reflectance2rgb(img, bgr=True):
    '''
    将遥感的反射率影像归一化为可以显示的RGB影像
    Returns:
    '''
    img = np.copy(img)
    #返回百分位数，取2~98％的数据
    top = np.percentile(img, 98)
    bottom = np.percentile(img, 2)
    img[img > top] = top
    img[img < bottom] = bottom
    img = (img - bottom) / (top - bottom) * 255
    if bgr:
        #：：-1代表从后往前截取
        img = img[::-1, :, :]
    return img.astype(np.uint8)


def outliner_nan(arr):
    mean, outliner = np.mean(arr), np.std(arr) * 3
    invalid_locs = np.where(np.logical_or(arr > mean + outliner, arr < mean - outliner))
    arr[invalid_locs] = np.nan
    return arr

def rmse(x, y):
    return np.sqrt(np.nanmean((x - y)**2))

def r(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    nonan_locs = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))
    return pearsonr(x[nonan_locs], y[nonan_locs])[0]

def r2(x, y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    return pearsonr(x, y)[0] ** 2

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

doy8 = np.arange(1, 366, 8)
doy1 = np.arange(1, 366, 1)

def outliner_rmse(x, y):
    m_square = (x - y) ** 2
    mean, outliner = np.mean(m_square), np.std(m_square) * 3
    invalid_locs = np.where(np.logical_or(m_square > mean + outliner, m_square < mean - outliner))
    m_square[invalid_locs] = np.nan
    return np.sqrt(np.nanmean(m_square))

def rse(x, y):
    mean_y = np.mean(y)
    upper = np.mean((x - y)**2)
    downer = np.mean((mean_y - y)**2)
    return upper / downer

def scatter_map(x, y, x_label=None, y_label=None, xy_range=None):
    x = x.reshape(-1)
    y = y.reshape(-1)
    plt.figure()
    plt.title("RMSE: {} \nR2: {}".format(np.round(rmse(x, y), 4),
                                         np.round(pearsonr(x, y)[0], 4)))
    plt.scatter(x, y)
    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)
    if xy_range is not None:
        plt.xlim(xmin=xy_range[0], xmax=xy_range[1])
        plt.ylim(ymin=xy_range[0], ymax=xy_range[1])
        plt.plot(xy_range, xy_range, ls='--', lw=1, c='r')
    plt.show()


# 通过DOY，找到离它最近的Location
def nearest_loc(arr, value):
    return np.argmin(np.abs(arr - value))


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
    #assert os.path.exists(file_path), "NO FILE!!! {}".format(file_path)
    dts = gdal.Open(file_path)
    proj = dts.GetProjection()
    geot = dts.GetGeoTransform()
    img = dts.ReadAsArray()
    if geoinfo:
        return img, proj, geot
    else:
        return img


# 像素坐标 to 地理坐标
def pl2xy(pl, geotransfer):
    x = geotransfer[0] + pl[0] * geotransfer[1] + pl[1] * geotransfer[2]
    y = geotransfer[3] + pl[0] * geotransfer[4] + pl[1] * geotransfer[5]
    return x, y  # longitude, latitude


# 地理坐标 to 像素坐标
def xy2pl(xy, geotransfer):
    x, y = xy
    p = (x - geotransfer[0]) / geotransfer[1]
    l = (y - geotransfer[3]) / geotransfer[5]
    return p, l


# 投影转换, 获取源的投影，并将目标影像的投影转为源的投影
def raster_reproject(tgp, otp, dst_srs):
    if not isinstance(dst_srs, int):
        dst_srs = osr.SpatialReference(gdal.Open(dst_srs).GetProjection())
    gdal.Warp(otp, gdal.Open(tgp), dstSRS=dst_srs)


# 按照shp文件裁剪img
def shp_clip(input_raster, input_shape, output_raster, cropToCutline=False, type='ENVI'):
    input_raster = gdal.Open(input_raster)
    gdal.Warp(output_raster,
               input_raster,
               format=type,
               cutlineDSName=input_shape,  # or any other file format
               cropToCutline=cropToCutline,
               dstNodata=0)  # select the no psmf_data value you like

def Z_ScoreNormalization(x):
    x = (x - np.mean(x)) / np.std(x)
    return x

def draw_con(ref_curve, target_curve, track, target_base=0):
    tar_loc, ref_loc = track
    locs = np.arange(0, np.max(tar_loc) + 1, 1)
    line_num = len(tar_loc)

    # plt.figure()
    plt.plot(locs, ref_curve, 'k', lw=3)
    plt.plot(locs, target_curve + target_base, c='C0', lw=3)

    for i in range(line_num):
        plt.plot((tar_loc[i], ref_loc[i]), (target_curve[tar_loc[i]] + target_base, ref_curve[ref_loc[i]]), 'k--')
    # plt.show()

# 两个栅格文件都具有相同的投影，该函数将input的栅格文件的裁剪成srs_raster相同的大小（包括nodata）。
def raster_clip(input_raster, srs_raster, output_raster, datatype='ENVI', invalid=None):
    idts = gdal.Open(input_raster)
    sdts = gdal.Open(srs_raster)
    iproj = idts.GetProjection(); sproj = sdts.GetProjection()
    assert iproj == sproj
    proj, geot = sdts.GetProjection(), sdts.GetGeoTransform()
    bands = idts.RasterCount
    sxsize, sysize = sdts.RasterXSize, sdts.RasterYSize
    ixsize, iysize = idts.RasterXSize, idts.RasterYSize
    output_img = np.zeros((bands, sysize, sxsize), GDAL2NP_CONVERSION[idts.GetRasterBand(1).DataType])
    input_img = idts.ReadAsArray()

    # input_img[np.where(input_img == 200)] = 0 4088 3252
    input_img[:, 4088:] = 0
    input_img[3252:, :] = 0

    x0, y0 = xy2pl(pl2xy((0, 0), sdts.GetGeoTransform()), idts.GetGeoTransform())
    x1, y1 = xy2pl(pl2xy((sxsize, sysize), sdts.GetGeoTransform()), idts.GetGeoTransform())
    if len(input_img.shape) == 2:
        input_img = input_img.reshape(1, iysize, ixsize)
    clip_img = input_img[: ,
               max(int(y0), 0):min(int(y1), iysize),
               max(int(x0), 0):min(int(x1), ixsize)]
    # input clip上的起始起始在srs上的位置。
    sx0, sy0 = xy2pl(pl2xy((max(int(x0), 0), max(int(y0), 0)), idts.GetGeoTransform()), sdts.GetGeoTransform())
    output_img[:, int(sy0): int(sy0) + clip_img.shape[1], int(sx0):int(sx0) + clip_img.shape[2]] = clip_img

    if invalid is not None:
        ref_img = gdal.Open(sdts).ReadAsArray()
        output_img[np.where(ref_img == invalid)] = invalid
    numpy2img(output_raster, output_img, proj=proj, geot=geot, type=datatype)

def EVI2(nir, red):
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1)

def NDVI(nir, red):
    return (nir - red) / (nir + 2.4 * red + 1)

def NDPI(nir, red, swir):
    alpha = 0.74
    return (nir - (red*alpha + swir*(1 - alpha))) / (nir + (red*alpha + swir*(1 - alpha)))

def WDRVI(nir, red):
    alpha = 0.1
    return (nir * alpha - red) / (nir * alpha + red)

def NDWI(nir, swir):
    return (nir - swir) / (nir + swir)

def EVI(nir, red, blue):
    L, C1, C2, G = 1, 6, 7.5, 2.5
    return G * (nir - red) / (nir + C1 * red - C2 * blue + L)

# 返回1个8bit的值，0：clear，1：uncertainty，2：snow；3：cloud
def MOD09_State(value):
    pass

# 通过经纬度来截取一小块面积
def xy_sub(xy, img, geot, wsize):
    p, l = xy2pl(xy, geot)
    p, l = int(round(p)), int(round(l))
    p0, p1 = p - wsize, p + wsize + 1
    l0, l1 = l - wsize, l + wsize + 1

    if len(img.shape) == 2:
        im_height, im_width = img.shape
        img = img.reshape(1, im_height, im_width)

    if p0 < 0 or l0 < 0 or l1 > img.shape[1] or p1 > img.shape[2]:
        # print("OUT EXTEND")
        return None

    return img[:, l0 : l1, p0 : p1]

# 将农气站观测数据的DOY按顺序排列(仅用于8天合成数据)
def transform_phe2doy(phe):
    zero_locs = np.where(phe==0)
    phe[:4] = np.where(phe[:4] < 100, phe[:4] + 156, phe[:4] - 209 + 1) # 209 是 DOY 1
    phe[4:] += 156  # 1月1日的含义是 ：1 + 153 为 DOY 154
    phe[zero_locs] = 0
    return phe

def transform_doy2phe(doy):
    return np.where(doy < 154, doy + 209, doy - 153)

def logistic4(t, t0, b, gc, bg):
    return gc / (1 + np.exp(b*(t - t0))) + bg

def show_histogram(data):
    plt.figure()
    plt.hist(data, bins=40, density=True, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    # 显示图标题
    plt.title("Histogram")
    plt.show()

if __name__ == "__main__":
    # shp_clip("Z:\dem\China_DEM_500m.dat",
    #          "F:\psmf_data\PhenologyMonitor\psmf_data\ReseachArea\PreciseArea.shp",
    #          "Z:\dem\\reshp_500m.dat", cropToCutline=True)
    # old_cover = "F:\psmf_data\PhenologyMonitor\psmf_data\WinterWheatCover\LiujiaCoverage\coverage.dat"
    # new_cover = "F:\psmf_data\PhenologyMonitor\psmf_data\MODIS12-13\COVER\coverage.tif"
    # src_dts = "F:\psmf_data\PhenologyMonitor\psmf_data\MODIS12-13\VI\doy.tif"
    #
    # raster_clip(old_cover, src_dts, new_cover, datatype='GTIFF')

    wrap = gdal.Open("/data/cylin/pj/CoolBoy/CropSegmentation/data/Subset0_warp.tif")
    proj = wrap.GetProjection()
    geot = wrap.GetGeoTransform()
    raw = gdal.Open("/data/cylin/pj/CoolBoy/CropSegmentation/data/GF-2/Subset0_object.dat").ReadAsArray()

    numpy2img("/data/cylin/pj/CoolBoy/CropSegmentation/data/GF-2/object0_warp.tif", raw.astype(np.int16), proj=proj, geot=geot)
    