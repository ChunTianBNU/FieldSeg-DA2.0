import torch
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import os, torch, core.util.UNIT as UNIT
from core.model.build import build_model,build_discriminator
import torch.optim as optim
from core.losses import TanimotoLoss,soft_label_cross_entropy
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
import random
import argparse
import yaml
# os.chdir("/home/tianchun/tcsegmentation/")
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

    # compose = A.Compose([A.RandomRotate90(p=0.5), A.Flip(p=0.5)], p=1)
    datatransform = A.Compose([
                        A.RandomRotate90(p=0.5),
                        A.Flip(p=0.5)],p=1)
    random.seed(seednum)
    images_concatenate = datatransform(image=images_concatenate)["image"]
    #保存
    A.save(datatransform, 'augmentation.json')

    for i, key in enumerate(images):
        temp = images_concatenate[:, :, band_size[i]:band_size[i + 1]]
        temp = np.transpose(temp, (2, 0, 1))
        images[key] = temp
    return images
def sentiaugmentation(**images):
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

    # compose = A.Compose([A.RandomRotate90(p=0.5), A.Flip(p=0.5)], p=1)
    #加载
    datatransform = A.load('augmentation.json')
    global seednum
    random.seed(seednum)
    # rng = np.random.default_rng(seednum)
    seednum =int((seednum+1)%100)
    images_concatenate = datatransform(image=images_concatenate)["image"]
    for i, key in enumerate(images):
        temp = images_concatenate[:, :, band_size[i]:band_size[i + 1]]
        temp = np.transpose(temp, (2, 0, 1))
        images[key] = temp
    return images
# 读取全部S123
class sourceData(BaseDataset):
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
        # 月份文件夹名字
        
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
                    # imag[i] = np.where(np.isnan(imag[i]),0,imag[i]) 

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

class targetData(BaseDataset):
    """_summary_

    Args:
    GF_dir = os.path.join("/home/tianchun/tcsegmentation/Unet/data/TargetData/NewCQ/Patch/Spec")
    senti_dir = os.path.join("/home/tianchun/tcsegmentation/Unet/data/TargetData/NewCQ/Patch/PatchSenti")
    """
    def __init__(
        self,
        GF_dir,
        senti_dir,
        dw_dir=None,
        sentiaugmentation=None,
    ):
        Data1="S0"
        self.images_fps=[]
        self.dw_fps=[]
        # 月份文件夹名字
        
        for Month in range(1,360,10):
            #Senti
            self.images_names_target=os.listdir(os.path.join(senti_dir,Data1,str(Month)))
            self.images_names_target.sort()
            images_fps_source=[os.path.join(senti_dir,Data1,str(Month),name) for name in self.images_names_target]
            self.images_fps.append(images_fps_source)
            if Month == 351:
                gf_fps1 = [os.path.join(GF_dir,Data1,Data1+'_'+name.split('351_')[1].split('.tif')[0]+".tif") for name in self.images_names_target]
                self.gf_fps=gf_fps1
                if dw_dir is not None:
                    dw_fps1 = [os.path.join(dw_dir,Data1,Data1+'_'+name.split('351_')[1].split('.tif')[0]+".tif") for name in self.images_names_target]     
                    self.dw_fps= dw_fps1
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
                    # imag[i] = np.where(np.isnan(imag[i]),0,imag[i]) 

            manyimage.append(imag)
        if self.dw_fps:
            dw = UNIT.img2numpy(self.dw_fps[i])
            dw = np.expand_dims(dw, axis=0)
            dw = dw.astype(float)
        gf = UNIT.img2numpy(self.gf_fps[i])
        gf = gf[0:4]
        if self.augmentation:
            if self.segmentation_fps: 
                data = self.augmentation(gf=gf, mask=segmentation,gt=gt)
                gf, segmentation,gt = data["gf"], data["mask"],data["gt"]
            else:
                data = self.augmentation(gf=gf)
                gf= data["gf"]
            Senti = self.sentiaugmentation(manyimage0=manyimage[0],manyimage1=manyimage[1],manyimage2=manyimage[2],manyimage3=manyimage[3],
                                           manyimage4=manyimage[4],manyimage5=manyimage[5],manyimage6=manyimage[6],manyimage7=manyimage[7],
                                           manyimage8=manyimage[8],manyimage9=manyimage[9],manyimage10=manyimage[10],manyimage11=manyimage[11],
                                           manyimage12=manyimage[12],manyimage13=manyimage[13],manyimage14=manyimage[14],manyimage15=manyimage[15],
                                           manyimage16=manyimage[16],manyimage17=manyimage[17],manyimage18=manyimage[18],manyimage19=manyimage[19],
                                           manyimage20=manyimage[20],manyimage21=manyimage[21],manyimage22=manyimage[22],manyimage23=manyimage[23],
                                           manyimage24=manyimage[24],manyimage25=manyimage[25],manyimage26=manyimage[26],manyimage27=manyimage[27],
                                           manyimage28=manyimage[28],manyimage29=manyimage[29],manyimage30=manyimage[30],manyimage31=manyimage[31],
                                           manyimage32=manyimage[32],manyimage33=manyimage[33],manyimage34=manyimage[34],manyimage35=manyimage[35])
            for i in range(36):
                manyimage[i]=Senti["manyimage"+str(i)]
        else:
            pass
        image.append(np.stack((manyimage[i] for i in range(36)),axis=0))  
        aimage = np.array(image).squeeze() #T,C,H,W 
        if self.segmentation_fps:
            return gf,aimage,dw
        else:    
            return gf,aimage

    def __len__(self):
        return len(self.images_fps[0])

def training(cfg):
    def adjust_learning_rate(base_lr, iters, max_iters, power):
        lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
        return lr
    # 从配置文件加载配置
    with open(cfg, 'r') as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)
    val_percent = 0.1
    gf_dir = cfg["DATASETDIR"]["GF2DIR"]
    spe_dir = cfg["DATASETDIR"]["SENTINEL2DIR"]
    segmentation_dir = cfg["DATASETDIR"]["IAFEXTENTDIR"]
    # dataset
    source_extent = sourceData(gf_dir,
                                     spe_dir,
                                    segmentation_dir,
                                    augmentation=augmentation,
                                    sentiaugmentation=sentiaugmentation)
    targetgf_dir = cfg["DATASETDIR"]["TARGETGF2DIR"]
    targetsenti_dir = cfg["DATASETDIR"]["TARGETSENTINEL2DIR"]
    ESA_dir = cfg["DATASETDIR"]["TARGETDWDIR"]
    target_extent = targetData(targetgf_dir,
                               targetsenti_dir,
                               ESA_dir,
                                sentiaugmentation=sentiaugmentation)

    model = build_model(cfg).cuda()
    model_D = build_discriminator(cfg).cuda()
    load_name = cfg["MODEL"]["MODELSOURCEWEIGHTS"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if load_name:
        pretrained_dict = torch.load(load_name,map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        fc_params_id=[]
        for name,param in model.named_parameters():
            fc_params_id.append(id(param))
    else:
        pass
    loss_1 = TanimotoLoss()

    optimizer = optim.Adam([{'params':model.parameters(),'lr':cfg["SOLVER"]["BASE_LR"]}])
    optimizer_D = optim.SGD(model_D.parameters(), lr=cfg["SOLVER"]["BASE_LR_D"],momentum=cfg["SOLVER"]["MOMENTUM"])
    print("Data Loading...")
    src_trainloader = torch.utils.data.DataLoader(source_extent,
                                              batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                                              shuffle=True,
                                              num_workers=0)
    tar_trainloader = torch.utils.data.DataLoader(target_extent,
                                            batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                                            shuffle=True,
                                            num_workers=0)
    #comment为名称，log_dir为存放目录
    writer1 = SummaryWriter(comment=f'LR_{0.001}_BS_{255}',log_dir=cfg["SOLVER"]["LOSS_CURVEDIR"])
    global_step = 0
    print("Loading Finished.")
    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(cfg["SOLVER"]["MAX_ITER"]):
            loss_total_seg = 0
            loss_total_adv = 0
            loss_total_D = 0
            loss_total_D_tar = 0
            loss_total_D_src = 0
            for step, ((srcdata),(tardata)) in enumerate(zip(src_trainloader,tar_trainloader)):
                #dataset
                current_lr = adjust_learning_rate(cfg["SOLVER"]["BASE_LR"], epoch, cfg["SOLVER"]["MAX_ITER"], cfg["SOLVER"]["OPTIMIZERPOWER"])
                current_lr_D = adjust_learning_rate(cfg["SOLVER"]["BASE_LR_D"], epoch, cfg["SOLVER"]["MAX_ITER"], cfg["SOLVER"]["OPTIMIZERPOWER_D"])
                for index in range(len(optimizer.param_groups)):
                    optimizer.param_groups[index]['lr'] = current_lr
                for index in range(len(optimizer_D.param_groups)):
                    optimizer_D.param_groups[index]['lr'] = current_lr_D
                src_input,src_senti,src_label= srcdata
                tar_input,tar_senti,tar_dw= tardata
                src_input = src_input.type(torch.cuda.FloatTensor).cuda()
                src_label = src_label.cuda()
                tar_input = tar_input.type(torch.cuda.FloatTensor).cuda()
                tar_dw = tar_dw.cuda()
                #loss
                optimizer.zero_grad()
                optimizer_D.zero_grad()
                src_fea,src_pred = model(src_input,src_senti)
                loss_seg = loss_1(src_pred,src_label.float())
                loss_seg.backward()
                src_pred=torch.sigmoid(src_pred)
                #2.loss_adv
                src_soft_label = torch.cat((src_pred.detach(),(1-src_pred.detach())),dim=1)
                src_soft_label[src_soft_label>0.9] = 0.9
                tar_fea,_ = model(tar_input,tar_senti)
                tar_soft_label = torch.cat((tar_dw,(1-tar_dw)),dim=1)
                tar_soft_label[tar_soft_label>0.9] = 0.9

                tar_D_pred = model_D(tar_fea)
                #Class balance
                src_count_pos = torch.sum(src_pred>=0.5)*1.0
                src_count_neg = torch.sum(src_pred<0.5)*1.0
                tar_count_pos = torch.sum(tar_dw>=0.5)*1.0
                tar_count_neg = torch.sum(tar_dw<0.5)*1.0
                #
                loss_adv_tgt = cfg["LOSSWEIGHTS"]["ADV"]*soft_label_cross_entropy(tar_D_pred, torch.cat((tar_soft_label, torch.zeros_like(tar_soft_label)), dim=1),tar_count_pos,tar_count_neg)

                loss_adv_tgt.backward()
                #3.loss_D
                optimizer.step()
                optimizer_D.zero_grad()
                src_D_pred = model_D(src_fea.detach())
                loss_D_src = cfg["LOSSWEIGHTS"]["SRC"]*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1),src_count_pos,src_count_neg)
                tgt_D_pred = model_D(tar_fea.detach())
                loss_D_tgt = cfg["LOSSWEIGHTS"]["TGT"]*soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tar_soft_label), tar_soft_label), dim=1),tar_count_pos,tar_count_neg)
                loss_D=loss_D_src+loss_D_tgt
                loss_D.backward()
                optimizer_D.step()
                if step % 40 == 0:
                    print('Current epoch-step: {}-{}'
                          ' <Segloss>:{}'
                          ' <advloss>:{}'
                          ' <Dloss>:{}'
                          ' AllocMem (Mb): {}'.format(
                              epoch, step,loss_seg.item(),loss_adv_tgt.item(),loss_D.item(),
                              torch.cuda.memory_allocated() / 1024 / 1024))
                    print()
                loss_total_seg+=loss_seg.item()
                loss_total_adv+=loss_adv_tgt.item()
                loss_total_D+=loss_D
                loss_total_D_src+=loss_D_src.item()
                loss_total_D_tar+=loss_D_tgt.item()
            loss_total_seg = loss_total_seg / len(tar_trainloader)
            loss_total_adv = loss_total_adv / len(tar_trainloader)
            loss_total_D = loss_total_D / len(tar_trainloader)
            loss_total_D_src = loss_total_D_src / len(tar_trainloader)
            loss_total_D_tar = loss_total_D_tar / len(tar_trainloader)
            writer1.add_scalars('train', {'seg_loss': loss_total_seg,'adv_loss': loss_total_adv,'D_loss': loss_total_D,'D_srcloss': loss_total_D_src,'D_tgtloss': loss_total_D_tar},global_step)
            if epoch==cfg["SOLVER"]["MAX_ITER"]-1 or epoch%(cfg["SOLVER"]["MAX_ITER"]/8)==0:
                torch.save(model.state_dict(), cfg["SOLVER"]["MODEL_SAVEDIR"])
            global_step += 1
    writer1.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IAFExtent FADAA Training')
    parser.add_argument('cfg', 
                        type=str, 
                        help='path to config file',
                        default="",
                        metavar="FILE")   
    args = parser.parse_args()
    training(args.cfg)