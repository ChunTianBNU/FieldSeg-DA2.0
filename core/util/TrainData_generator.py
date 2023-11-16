import UNIT
import numpy as np
import os
import math
def subset_dts(img, size, start, interval):
    """
    :param img:待裁剪图像
    :param size:裁剪图像大小
    :param start:裁剪的起始点坐标
    :param interval:点间隔
    :return:裁剪后图像列表
    """
    if len(img.shape) == 2:
        xlen, ylen = img.shape
        img = img.reshape((1, xlen, ylen))
        b = 1
    else:
        b, xlen, ylen = img.shape
    half_size = int(size / 2)
    #100,3000,200,
    #16, ,20
    x_center = np.arange(start, xlen, interval, dtype=np.int32)
    y_center = np.arange(start, ylen, interval, dtype=np.int32)
    #生成网格点,x_center为二维
    x_center, y_center = np.meshgrid(x_center, y_center)
    xlen_chip, ylen_chip = x_center.shape
    img_list = []
    for i in range(xlen_chip):
        for j in range(ylen_chip):
            xloc0, xloc1 = max((x_center[i, j] - half_size, 0)), min(
                (x_center[i, j] + half_size, xlen))
            yloc0, yloc1 = max((y_center[i, j] - half_size, 0)), min(
                (y_center[i, j] + half_size, ylen))
            subset_img = np.zeros((b, size, size), dtype=img.dtype)
            # =====================================裁剪=================================
            if xloc0==0:
                if yloc0==0:
                    subset_img[:, :size, :size] = img[:, :size,:size]
                elif yloc1==ylen:
                    subset_img[:, :size, :size] = img[:, :size,ylen-size:ylen]
                else:
                    subset_img[:, :size, :size] = img[:, :size,yloc0:yloc1]
            elif xloc1==xlen:
                if yloc0==0:
                    subset_img[:, :size, :size] = img[:, xlen-size:xlen,:size]
                elif yloc1==ylen:
                    subset_img[:, :size, :size] = img[:, xlen-size:xlen,ylen-size:ylen]
                else:
                    subset_img[:, :size, :size] = img[:, xlen-size:xlen,yloc0:yloc1]      
            elif yloc0==0:
                subset_img[:, :size, :size] = img[:, xloc0:xloc1,:size]
            elif yloc1==ylen:
                subset_img[:, :size, :size] = img[:, xloc0:xloc1,ylen-size:ylen]
            else:
                subset_img[:, :size, :size] = img[:, xloc0:xloc1,yloc0:yloc1]

            img_list.append(subset_img)            
    return img_list
def GF2andLabels():
    output_path = "/FieldSeg-DA2.0/dataset/sourcedomain"
    # id = 0
    for i in range(3):
        id = i
        img_spe = UNIT.img2numpy("/home/tianchun/tcsegmentation/Unet/data/iFLYTEK/Subset_Norm/Subset{}_Norm.tif".format(id))
        img_roi = UNIT.img2numpy("/home/tianchun/tcsegmentation/Unet/data/iFLYTEK/IAFExtentpatch/IAFExtentpatch{}.tif".format(id))
        img_roi = np.where(img_roi == 0, 0, 1)
        img_bd = UNIT.img2numpy("/home/tianchun/tcsegmentation/Unet/data/iFLYTEK/IAFBoundarypatch/IAFBoundarypatch{}.tif".format(id))
        img_bd = np.where(img_bd == 0, 0, 1)

        chip_size = 256
        subset_spe_imgs = subset_dts(img_spe, chip_size, 100, 200)
        subset_semantic_imgs = subset_dts(img_roi, chip_size, 100, 200)
        subset_bd_imgs = subset_dts(img_bd, chip_size, 100, 200)

        for i in range(len(subset_spe_imgs)):
            if not os.path.exists(os.path.join(output_path,"GF2patch","S{}".format(id))):
                os.makedirs(os.path.join(output_path, "GF2patch","S{}".format(id)))
            UNIT.numpy2img(os.path.join(output_path, "GF2patch","S{}/S{}_{}.tif".format(id, id, i)),subset_spe_imgs[i])

            if not os.path.exists(os.path.join(output_path, "IAFExtentpatch", "S{}".format(id))):
                os.makedirs(os.path.join(output_path, "IAFExtentpatch", "S{}".format(id)))
            UNIT.numpy2img(os.path.join(output_path, "IAFExtentpatch", "S{}/S{}_{}.tif".format(id, id, i)), subset_semantic_imgs[i].astype('int8'))

            if not os.path.exists(os.path.join(output_path, "IAFBoundarypatch", "S{}".format(id))):
                os.makedirs(os.path.join(output_path, "IAFBoundarypatch", "S{}".format(id)))
            UNIT.numpy2img(os.path.join(output_path, "IAFBoundarypatch", "S{}/S{}_{}.tif".format(id, id, i)), subset_bd_imgs[i].astype('int8'))
def Sentinel2patch():
    def Ex2generater(img, otherimg, size, start, interval):
        """
        :param img:4波段影像
        :param otherimg:低分辨率影像
        :param size:裁剪图像大小
        :param start:裁剪的起始点坐标
        :param interval:点间隔
        :return:裁剪后图像列表
        """
        if len(img.shape) == 2:
            xlen, ylen = img.shape
            img = img.reshape((1, xlen, ylen))
            b = 1
        else:
            b, xlen, ylen = img.shape
            b2,xlen2, ylen2 = otherimg.shape
        half_size = int(size / 2)
        #100,3000,200,
        x_center = np.arange(start, xlen+interval, interval, dtype=np.int32)
        y_center = np.arange(start, ylen+interval, interval, dtype=np.int32)
        #生成网格点,x_center为二维
        x_center, y_center = np.meshgrid(x_center, y_center)
        xlen_chip, ylen_chip = x_center.shape
        img_list = []
        multiple=10
        for i in range(xlen_chip):
            for j in range(ylen_chip):
                xloc0, xloc1 = max((x_center[i, j] - half_size, 0)), min(
                    (x_center[i, j] + half_size, xlen))
                yloc0, yloc1 = max((y_center[i, j] - half_size, 0)), min(
                    (y_center[i, j] + half_size, ylen))
                lowsize=math.ceil(size/10)
                other_img = np.zeros((b2, lowsize, lowsize), dtype=np.float32)
                # =====================================裁剪=================================
                if xloc0==0:
                    if yloc0==0:
                        other_img[:, :lowsize,:lowsize] = otherimg[:, :lowsize,:lowsize]
                    elif yloc1==ylen:
                        other_img[:, :lowsize,:lowsize] = otherimg[:, :lowsize,-lowsize:]
                    else:
                        other_img[:, :lowsize,:lowsize] = otherimg[:, :lowsize,yloc0 //multiple:math.ceil(yloc1 / multiple)]
                elif xloc1==xlen:
                    if yloc0==0:
                        other_img[:, :lowsize,:lowsize] = otherimg[:, (math.ceil(xlen / multiple))-lowsize:math.ceil(xlen / multiple),:lowsize]
                    elif yloc1==ylen:
                        other_img[:, :lowsize,:lowsize] = otherimg[:, (math.ceil(xlen / multiple))-lowsize:math.ceil(xlen / multiple),-lowsize:]
                    else:
                        other_img[:, :lowsize,:lowsize] = otherimg[:, (math.ceil(xlen / multiple))-lowsize:math.ceil(xlen / multiple),yloc0 //multiple:math.ceil(yloc1 / multiple)]       
                elif yloc0==0:
                    other_img[:, :lowsize,:lowsize] = otherimg[:, xloc0 //multiple:math.ceil(xloc1 / multiple),:lowsize]
                elif yloc1==ylen:
                    other_img[:, :lowsize,:lowsize] = otherimg[:, xloc0 //multiple:math.ceil(xloc1 / multiple),-lowsize:]
                else:
                    other_img[:, :lowsize,:lowsize] = otherimg[:, xloc0 //multiple:math.ceil(xloc1 / multiple),yloc0 //multiple:math.ceil(yloc1 / multiple)]

                img_list.append(other_img)
        return img_list
    output_path = "/FieldSeg-DA2.0/dataset/sourcedomain/Sentinel2patch"
    for i in range(1):
        id = 3
        for Month in range(1,360,10):
            OutDir= os.path.join(output_path, "S{}".format(id))
            MonthDir=os.path.join(OutDir,str(Month))
            if not os.path.exists(OutDir):
                os.makedirs(OutDir)
            if not os.path.exists(MonthDir):
                os.makedirs(MonthDir)
            img_spe = UNIT.img2numpy("/home/tianchun/tcsegmentation/Unet/data/TargetData/HH/GF2/z_normHH.tif".format(id))
            Low_img = UNIT.img2numpy("/home/tianchun/tcsegmentation/Unet/data/Ex10/long/z_normYearSentinel/Sen{}/Sentinel{}_{}.tif".format(id,id,Month)) 
            chip_size = 256
            subset_spe_imgs = Ex2generater(img_spe, Low_img, chip_size, 100, 200)

            for i in range(len(subset_spe_imgs)):
                UNIT.numpy2img(
                    os.path.join(MonthDir,"Sen{}_{}_{}.tif".format(id,Month,i)),
                    subset_spe_imgs[i].astype(np.float32))
                

if __name__ == "__main__":
    GF2andLabels()
    Sentinel2patch()