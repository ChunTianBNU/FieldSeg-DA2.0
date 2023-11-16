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

    for i, key in enumerate(images):
        temp = images_concatenate[:, :, band_size[i]:band_size[i + 1]]
        temp = np.transpose(temp, (2, 0, 1))
        images[key] = temp
    return images

class sourceData(BaseDataset):
    """_summary_

    Args:
    GF_dir = os.path.join("/home/tianchun/tcsegmentation/Unet/data/WSpec/long")
    images_dir = os.path.join("/home/tianchun/tcsegmentation/Unet/data/Ex10/long/Spec_10")
    segmentation_dir = os.path.join("/home/tianchun/tcsegmentation/Unet/data/Semantic")
    """
    def __init__(
        self,
        GF_dir,
        segmentation_dir,
        augmentation=None,
    ):
        self.gf_fps=[]
        self.segmentation_fps=[]
        
        for i in range(3):
            if i != 99:
                for file in os.listdir(os.path.join(segmentation_dir,"S{}".format(i))):
                    self.segmentation_fps.append(os.path.join(segmentation_dir,"S{}".format(i),file))
                    self.gf_fps.append(os.path.join(GF_dir,"S{}".format(i),file))
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        segmentation = UNIT.img2numpy(self.segmentation_fps[i])
        segmentation = np.expand_dims(segmentation, axis=0)
        segmentation = segmentation.astype(float)

        gf = UNIT.img2numpy(self.gf_fps[i])
        gf = gf[0:4]
        if self.augmentation:
            data = self.augmentation(gf=gf, mask=segmentation)
            gf, segmentation = data["gf"], data["mask"]
            pass
        return gf,segmentation

    def __len__(self):
        return len(self.segmentation_fps)

class targetData(BaseDataset):
    """_summary_

    Args:
    GF_dir = os.path.join("/home/tianchun/tcsegmentation/Unet/data/TargetData/HH/Patch/Spec")
    senti_dir = os.path.join("/home/tianchun/tcsegmentation/Unet/data/TargetData/HH/Patch/PatchSenti")
    """
    def __init__(
        self,
        GF_dir,
        esa_dir=None
    ):
        self.gf_fps=[]
        self.segmentation_fps=[]
        
        for i in range(1):
            if i != 99:
                for file in os.listdir(os.path.join(GF_dir,"S{}".format(i))):
                    self.gf_fps.append(os.path.join(GF_dir,"S{}".format(i),file))
                    if esa_dir:
                        self.segmentation_fps.append(os.path.join(esa_dir,"S{}".format(i),file))
        self.augmentation = augmentation

    def __getitem__(self, i):
        # read data
        if self.segmentation_fps:
            segmentation = UNIT.img2numpy(self.segmentation_fps[i])
            segmentation = np.expand_dims(segmentation, axis=0)
            segmentation = segmentation.astype(float)
        gf = UNIT.img2numpy(self.gf_fps[i])
        gf = gf[0:4]
        if self.augmentation:
            if self.segmentation_fps:
                data = self.augmentation(gf=gf, mask=segmentation)
                gf, segmentation = data["gf"], data["mask"]
            else:
                data = self.augmentation(gf=gf)
                gf = data["gf"]
        if self.segmentation_fps:
            return gf,segmentation
        else:
            return gf

    def __len__(self):
        return len(self.gf_fps)
def soft_label_cross_entropy(pred, soft_label,count_pos,count_neg,pixel_weights=None):
    # soft_label (b,4,h,w)

    beta_1 = count_pos / (count_pos + count_neg)
    beta_2 = count_neg / (count_pos + count_neg)
    soft_label[:,0]=soft_label[:,0]*beta_2
    soft_label[:,1]=soft_label[:,1]*beta_1
    soft_label[:,2]=soft_label[:,2]*beta_2
    soft_label[:,3]=soft_label[:,3]*beta_1
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def training(cfg):
    def adjust_learning_rate(base_lr, iters, max_iters, power):
        lr = base_lr * ((1 - float(iters) / max_iters) ** (power))
        return lr
    with open(cfg, 'r') as config_file:
        cfg = yaml.load(config_file, Loader=yaml.FullLoader)
    sourcegf_dir = cfg["DATASETDIR"]["GF2DIR"]
    segmentation_dir = cfg["DATASETDIR"]["IAFBOUNDARYDIR"]
    targetgf_dir = cfg["DATASETDIR"]["TARGETGF2DIR"]
    Prior_dir = cfg["DATASETDIR"]["TARGETPRIORDIR"]
    # dataset
    source_extent = sourceData(sourcegf_dir,
                                     segmentation_dir,
                                     augmentation=augmentation,)
    target_extent = targetData(targetgf_dir,
                               Prior_dir,)
    model = build_model(cfg).cuda()
    model_D = build_discriminator(cfg).cuda()

    load_name = cfg["MODEL"]["MODELSOURCEWEIGHTS"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if load_name:
        pretrained_dict = torch.load(load_name,map_location=device)
        model_dict = model.state_dict()
        # model.load_state_dict(torch.load(load_name, map_location=device))
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict) 
        fc_params_id=[]
        for name,param in model.named_parameters():
            if 'segmentation' not in name:
                print(name)
                fc_params_id.append(id(param))

    loss_Bce = TanimotoLoss()
    optimizer = optim.Adam([{'params':filter(lambda p: id(p) not in fc_params_id and p.requires_grad, model.parameters()),'lr':cfg["SOLVER"]["BASE_LR"]}, 
        {'params':filter(lambda p: id(p) in fc_params_id and p.requires_grad, model.parameters()),'lr':cfg["SOLVER"]["BASE_LR"]}])
    optimizer_D = optim.SGD(model_D.parameters(), lr=cfg["SOLVER"]["BASE_LR_D"],momentum=cfg["SOLVER"]["MOMENTUM"])

    print("Data Loading...")
    src_trainloader = torch.utils.data.DataLoader(source_extent,
                                              batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                                              shuffle=True,
                                              num_workers=0,)
    tar_trainloader = torch.utils.data.DataLoader(target_extent,
                                            batch_size=cfg["SOLVER"]["BATCH_SIZE"],
                                            shuffle=True,
                                            num_workers=0,)
    writer1 = SummaryWriter(comment=f'LR_{0.001}_BS_{255}',log_dir=cfg["SOLVER"]["LOSS_CURVEDIR"])
    global_step = 0
    print("Loading Finished.")

    with torch.autograd.set_detect_anomaly(True):
        for epoch in range(cfg["SOLVER"]["MAX_ITER"]):
            loss_total_seg = 0
            loss_total_adv_tgt = 0
            loss_total_D = 0
            loss_total_D_src = 0
            loss_total_D_tgt = 0
            for step, ((srcdata),(tardata)) in enumerate(zip(src_trainloader,tar_trainloader)):
                current_lr = adjust_learning_rate(cfg["SOLVER"]["BASE_LR"], epoch, cfg["SOLVER"]["MAX_ITER"], cfg["SOLVER"]["OPTIMIZERPOWER"])
                current_lr_D = adjust_learning_rate(cfg["SOLVER"]["BASE_LR_D"], epoch, cfg["SOLVER"]["MAX_ITER"], cfg["SOLVER"]["OPTIMIZERPOWER_D"])
                for index in range(len(optimizer.param_groups)):
                    if index==1: #F
                        optimizer.param_groups[index]['lr'] = current_lr
                    if index==0: #C
                        optimizer.param_groups[index]['lr'] = current_lr
                for index in range(len(optimizer_D.param_groups)):
                    optimizer_D.param_groups[index]['lr'] = current_lr_D
                #dataset
                src_input,src_label= srcdata
                tar_input,tar_esa,tar_label= tardata
                src_input = src_input.type(torch.cuda.FloatTensor).cuda()
                src_label = src_label.cuda()
                tar_input = tar_input.type(torch.cuda.FloatTensor).cuda()
                tar_esa = tar_esa.cuda()
                tar_label = tar_label.cuda()
                #loss
                optimizer.zero_grad()
                optimizer_D.zero_grad()
                src_fea,src_pred = model(src_input)
                loss_seg1 = loss_Bce(src_pred,src_label.float())
                src_pred = torch.sigmoid(src_pred)
                tar_fea,tar_pred = model(tar_input)
                tar_pred = torch.sigmoid(tar_pred)

                #2.loss_adv
                src_soft_label = torch.cat((src_pred.detach(),(1-src_pred.detach())),dim=1)
                src_soft_label[src_soft_label>0.9] = 0.9

                tar_gan = tar_label.detach()
                tar_soft_label = torch.cat((tar_gan,(1-tar_gan)),dim=1)

                src_count_pos = torch.sum(src_pred>=0.5)*1.0
                src_count_neg = torch.sum(src_pred<0.5)*1.0
                tar_count_pos = torch.sum(tar_gan>=0.5)*1.0
                tar_count_neg = torch.sum(tar_gan<0.5)*1.0

                tar_D_pred = model_D(tar_fea)
                src_D_pred = model_D(src_fea)
                loss_adv_tgt = cfg["LOSSWEIGHTS"]["ADV"]*soft_label_cross_entropy(tar_D_pred, torch.cat((tar_soft_label, torch.zeros_like(tar_soft_label)), dim=1),tar_count_pos,tar_count_neg)
                loss_adv = loss_adv_tgt+loss_seg1

                loss_adv.backward()
                #3.loss_D
                optimizer.step()
                optimizer_D.zero_grad()
                src_D_pred = model_D(src_fea.detach())
                loss_D_src =cfg["LOSSWEIGHTS"]["SRC"]*soft_label_cross_entropy(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1),src_count_pos,src_count_neg)
                
                tgt_D_pred = model_D(tar_fea.detach())
                loss_D_tgt = cfg["LOSSWEIGHTS"]["TGT"]*soft_label_cross_entropy(tgt_D_pred, torch.cat((torch.zeros_like(tar_soft_label), tar_soft_label), dim=1),tar_count_pos,tar_count_neg)
                loss_D=loss_D_src+loss_D_tgt
                loss_D.backward()
                optimizer_D.step()

                


                if step % 20 == 0:
                    print('Current epoch-step: {}-{}'
                          ' <Segloss>:{}'
                          ' <advloss>:{}'
                          ' <Dloss>:{}'
                          ' AllocMem (Mb): {}'.format(
                              epoch, step,loss_seg1.item(),loss_adv_tgt.item(),loss_D.item(),
                              torch.cuda.memory_allocated() / 1024 / 1024))
                    print()
                loss_total_seg+=loss_seg1.item()
                loss_total_adv_tgt+=loss_adv_tgt.item()
                loss_total_D+=loss_D
                loss_total_D_src+=loss_D_src
                loss_total_D_tgt+=loss_D_tgt
            loss_total_seg = loss_total_seg / len(tar_trainloader)
            loss_total_adv_tgt = loss_total_adv_tgt / len(tar_trainloader)
            loss_total_D = loss_total_D / len(tar_trainloader)
            loss_total_D_src = loss_total_D_src / len(tar_trainloader)
            loss_total_D_tgt = loss_total_D_tgt / len(tar_trainloader)
        
            writer1.add_scalars('train', {'seg_loss': loss_total_seg,'adv_tgt_loss': loss_total_adv_tgt,\
                                          'D_loss': loss_total_D,'Dsrc_loss': loss_total_D_src,'Dtgt_loss': loss_total_D_tgt},global_step)
            if epoch==["SOLVER"]["MAX_ITER"]-1 or epoch%(cfg["SOLVER"]["MAX_ITER"]/8)==0:
                torch.save(model.state_dict(), cfg["SOLVER"]["MODEL_SAVEDIR"]+str(epoch))
            global_step += 1
    writer1.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IAFBoundary FADAA Training')
    parser.add_argument('cfg', 
                        type=str, 
                        help='path to config file',
                        default="",
                        metavar="FILE")   
    args = parser.parse_args()
    training(args.cfg)