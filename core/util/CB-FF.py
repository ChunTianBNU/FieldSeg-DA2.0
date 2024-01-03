import cv2
from skimage import morphology
import UNIT as UNIT
from osgeo import gdal
import numpy as np
from tqdm import tqdm
#--------------------------------------------CB-FF--------------------------------------------
def CB(inputBoundary):
    
    # 读取二值化图像
    # Path=inputdir
    # outname=outputdir
    # dts = gdal.Open(Path)
    # proj = dts.GetProjection()
    # geot = dts.GetGeoTransform()
    # image = UNIT.img2numpy(Path)
    image = np.where(inputBoundary>0.5,1,0)
    # 提取骨架
    skeleton = morphology.skeletonize(image)
    # UNIT.numpy2img(outname+"_Ske.tif", skeleton.astype(np.uint8), proj=proj, geot=geot)
    # 进行膨胀操作
    kernel = np.ones((5, 5), np.uint8)
    expanded = cv2.dilate(skeleton.astype(np.uint8) * 255, kernel, iterations=1)
    # UNIT.numpy2img(outname+"_Ske_Dilate.tif", expanded.astype(np.uint8), proj=proj, geot=geot)
    # 提取骨架
    skeleton = morphology.skeletonize(expanded>0)
    # 进行膨胀操作
    kernel = np.ones((3, 3), np.uint8)
    expanded = cv2.dilate(skeleton.astype(np.uint8) * 255, kernel, iterations=1)
    #merging
    merge=np.logical_or(image > 0, expanded > 0).astype(np.uint8)
    return merge
def FF(image):
    # 寻找连通域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)

    # 创建一个与输入图像相同大小的输出图像
    output = np.zeros_like(image)

    # 对于每个连通域（除了背景）
    for label in range(1, num_labels):
        # 创建一个仅包含当前连通域的掩码图像
        mask = np.uint8(labels == label)

        # 对当前连通域进行膨胀操作以消除内部孔洞
        dilated = cv2.dilate(mask, None, iterations=1)
        result = cv2.erode(dilated, None, iterations=1)

        # 将膨胀后的结果添加到输出图像中
        output = cv2.add(output, result.astype(np.uint8))

    return output
# --------------------------------------------- CB ----------------------------------------------
Path="IAFBoundary.tif"
dts = gdal.Open(Path)
proj = dts.GetProjection()
geot = dts.GetGeoTransform()
Boundary = UNIT.img2numpy(Path)
merge=CB(Boundary)

# --------------------------------------------- FF ----------------------------------------------
image = UNIT.img2numpy("IAFExtent.tif")
image = np.where(image>0.5,1,0)
union = cv2.bitwise_and(image, cv2.bitwise_not(merge))
result = FF(result.astype(np.uint8))
outname="IAF.tif"
UNIT.numpy2img(outname, result.astype(np.uint8), proj=proj, geot=geot)