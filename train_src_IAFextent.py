import torch
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import os, torch, core.util.UNIT as UNIT
from core.model.build import build_model
import torch.optim as optim
from core.losses import TanimotoLoss
from torch.utils.tensorboard import SummaryWriter
import time
import albumentations as A
import random
from torch.utils.data import random_split
import argparse
import yaml
seednum = 50
def augmentation(**images):
    '''
    通过图像左右、上下翻转进行增强
    Returns:
    '''
    band_size = [
        0,
    ]
    images_concatenate = []
    for key in images:
        temp = np.transpose(images[key], (1, 2, 0))
        band_size.append(temp.shape[2] + band_size[-1])
        images_concatenate.append(temp)
    images_concatenate = np.concatenate(images_concatenate, axis=2)


    datatransform = A.Compose([
                        A.RandomRotate90(p=0.5),
                        A.Flip(p=0.5)],p=1)
    random.seed(seednum)
    images_concatenate = datatransform(image=images_concatenate)["image"]

    A.save(datatransform, 'augmentation.json')

    for i, key in enumerate(images):
        temp = images_concatenate[:, :, band_size[i]:band_size[i + 1]]
        temp = np.transpose(temp, (2, 0, 1))
        images[key] = temp
    return images
def sentiaugmentation(**images):
    '''
    Returns:
    '''
    band_size = [
        0,
    ]
    images_concatenate = []
    for key in images:
        temp = np.transpose(images[key], (1, 2, 0))
        band_size.append(temp.shape[2] + band_size[-1])
        images_concatenate.append(temp)
    images_concatenate = np.concatenate(images_concatenate, axis=2)



    datatransform = A.load('augmentation.json')
    global seednum
    random.seed(seednum)

    seednum =int((seednum+1)%100)
    images_concatenate = datatransform(image=images_concatenate)["image"]
    for i, key in enumerate(images):
        temp = images_concatenate[:, :, band_size[i]:band_size[i + 1]]
        temp = np.transpose(temp, (2, 0, 1))
        images[key] = temp
    return images
class DatasetSegtask2(BaseDataset):
    """
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)
    """
    def __init__(
        self,
        GF_dir,
        images_dir,
        segmentation_dir,
        augmentation=None,
        sentiaugmentation=None,
    ):
        Data1="S0"             
        self.gf_fps=[]
        self.images_fps=[]
        self.segmentation_fps=[]
        
        for Month in range(1,360,10):

            self.images_names1=os.listdir(os.path.join(images_dir,Data1,str(Month)))
            self.images_names1.sort()
            images_fps1=[os.path.join(images_dir,Data1,str(Month),name) for name in self.images_names1]        
            self.images_fps.append(images_fps1)
            if Month == 351:
                segmentation_fps1 = [os.path.join(segmentation_dir,Data1,Data1+'_'+name.split('351_')[1].split('.tif')[0]+".tif") for name in self.images_names1]
                gf_fps1 = [os.path.join(GF_dir,Data1,Data1+'_'+name.split('351_')[1].split('.tif')[0]+".tif") for name in self.images_names1]
                self.segmentation_fps= segmentation_fps1
                self.gf_fps=gf_fps1
        # convert str names to class values on masks
        self.augmentation = augmentation
        self.sentiaugmentation = sentiaugmentation

    def __getitem__(self, i):
        image=[]
    
        imagedir = [X[i] for X in self.images_fps]
        manyimage=[]
        for ima in imagedir:
            imag = UNIT.img2numpy(ima)
            if np.isnan(imag).any():
                for j in range(4): 
                    exist = (~np.isnan(imag[j]))
                    temp = np.where(np.isnan(imag[j]),0,imag[j])
                    mean=temp.sum()/(exist.sum()+1e-7)
                    imag[j] = np.where(np.isnan(imag[j]),mean,imag[j])

            manyimage.append(imag)
        segmentation = UNIT.img2numpy(self.segmentation_fps[i])
        segmentation = np.expand_dims(segmentation, axis=0)
        segmentation = segmentation.astype(float)

        gf = UNIT.img2numpy(self.gf_fps[i])
        gf = gf[0:4]
        if self.augmentation:
            data = self.augmentation(gf=gf, mask=segmentation)

            Senti = self.sentiaugmentation(manyimage0=manyimage[0],manyimage1=manyimage[1],manyimage2=manyimage[2],manyimage3=manyimage[3],
                                           manyimage4=manyimage[4],manyimage5=manyimage[5],manyimage6=manyimage[6],manyimage7=manyimage[7],
                                           manyimage8=manyimage[8],manyimage9=manyimage[9],manyimage10=manyimage[10],manyimage11=manyimage[11],
                                           manyimage12=manyimage[12],manyimage13=manyimage[13],manyimage14=manyimage[14],manyimage15=manyimage[15],
                                           manyimage16=manyimage[16],manyimage17=manyimage[17],manyimage18=manyimage[18],manyimage19=manyimage[19],
                                           manyimage20=manyimage[20],manyimage21=manyimage[21],manyimage22=manyimage[22],manyimage23=manyimage[23],
                                           manyimage24=manyimage[24],manyimage25=manyimage[25],manyimage26=manyimage[26],manyimage27=manyimage[27],
                                           manyimage28=manyimage[28],manyimage29=manyimage[29],manyimage30=manyimage[30],manyimage31=manyimage[31],
                                           manyimage32=manyimage[32],manyimage33=manyimage[33],manyimage34=manyimage[34],manyimage35=manyimage[35])
            gf, segmentation = data["gf"], data["mask"]
            for i in range(36):
                manyimage[i]=Senti["manyimage"+str(i)]
            pass
        image.append(np.stack((manyimage[i] for i in range(36)),axis=0))  
        aimage = np.array(image).squeeze() #T,C,H,W  
        return gf,aimage,segmentation

    def __len__(self):
        return len(self.segmentation_fps)

def training(cfg):

    with open(cfg, 'r') as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)
    val_percent = 0.1

    gf_dir = cfg["DATASETDIR"]["GF2DIR"]
    spe_dir = cfg["DATASETDIR"]["SENTINEL2DIR"]
    segmentation_dir = cfg["DATASETDIR"]["IAFEXTENTDIR"]
    # dataset
    dataset_extent = DatasetSegtask2(gf_dir,
                                     spe_dir,
                                    segmentation_dir,
                                    augmentation=augmentation,
                                    sentiaugmentation=sentiaugmentation)

    n_val = int(len(dataset_extent) * val_percent)  
    n_train = len(dataset_extent) - n_val  
    train_dataset_extent, val_dataset_extent = random_split(
        dataset_extent, [n_train, n_val])
    model = build_model(cfg).cuda()
    
    loss_fn1 = TanimotoLoss()
    optimizer = optim.Adam([{'params':model.parameters(),'lr':cfg["SOLVER"]["BASE_LR"]}])  #GF
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    print("Data Loading...")
    trainloader = torch.utils.data.DataLoader(train_dataset_extent,
                                              batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                                              shuffle=True,
                                              num_workers=0)
    valloader = torch.utils.data.DataLoader(val_dataset_extent,
                                            batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                                            shuffle=True,
                                            num_workers=0)
    writer1 = SummaryWriter(comment=f'LR_{0.001}_BS_{255}',log_dir=cfg["SOLVER"]["LOSS_CURVEDIR"])
    global_step = 0
    early_stop = False
    print("Loading Finished.")
    with torch.autograd.set_detect_anomaly(True):
        train_epoch_best_loss1 = 100.0
        epochs_no_improve=0
        since = time.time()
        for epoch in range(cfg["SOLVER"]["MAX_ITER"]):
            if early_stop:
                break
            loss_total_crop = 0
            for step, data in enumerate(trainloader, 0):
                gf,image, seg= data
                image = image.type(torch.cuda.FloatTensor).cuda()
                gf = gf.type(torch.cuda.FloatTensor).cuda()
                seg = seg.cuda()
                optimizer.zero_grad()
                Crop = model(gf,image)
                loss_crop = loss_fn1(Crop, seg.float())
                loss_crop.backward()
                # 
                loss_total_crop += loss_crop.item()
                optimizer.step()
            loss_total_crop = loss_total_crop / len(trainloader)
            writer1.add_scalars('train', {'crop_loss': loss_total_crop},
                               global_step)
            if epoch % 1 == 0:
                print("---Validation e {}: ".format(epoch), end=", ")
                num = 0
                loss_val1 = 0
                with torch.no_grad():
                    for step, data in enumerate(valloader, 0):
                        gf,image, seg = data
                        image = image.to(torch.float32).cuda()
                        gf = gf.to(torch.float32).cuda()
                        seg = seg.cuda()
                        Crop = model(gf,image)
                        loss_crop = loss_fn1(Crop, seg.float())
                        loss_val1 += loss_crop.item()
                        num += 1
                loss_val1 = loss_val1 / len(valloader)
                print("\033[31m"
                      "LR:", optimizer.state_dict()['param_groups'][0]['lr'],";",optimizer.state_dict()['param_groups'][1]['lr'],'\n',
                      "Val_CropLoss:",loss_val1,"\n",
                      ".\033[0m")
                writer1.add_scalars('test', {'crop_loss': loss_val1},
                                   global_step)
            scheduler.step()
            train_epoch_loss1 = loss_val1
            if train_epoch_loss1 >= train_epoch_best_loss1:
                epochs_no_improve+=1
            else:
                train_epoch_best_loss1 = train_epoch_loss1
                epochs_no_improve=0
                torch.save(model.state_dict(), cfg["SOLVER"]["MODEL_SAVEDIR"])
            if epochs_no_improve == cfg["SOLVER"]["MAXEPOCH_NOIMPROVE"]:
                print(f'Early stopping after {epoch+1} epochs.')
                early_stop = True
            global_step += 1
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
    writer1.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='U-LSTM src Training')
    parser.add_argument('cfg', 
                        type=str, 
                        help='path to config file',
                        default="",
                        metavar="FILE")   
    args = parser.parse_args()
    training(args.cfg)