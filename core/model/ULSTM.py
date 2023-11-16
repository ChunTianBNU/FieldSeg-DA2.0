import torch
from torch import nn
import torch.nn.functional as F
from .CNNSobel import Sobel
from .Resnet34 import Resnet34
from .convlstm import BConvLSTM
from .CBAM import CBAMBlock

class SegmentationHead(nn.Sequential):
    def __init__(self,
                 in_channels=32,
                 out_channels=1,
                 kernel_size=3,
                 upsampling=1):
        conv2d = nn.Conv2d(in_channels,
                           out_channels,
                           kernel_size=kernel_size,
                           padding=kernel_size // 2,
                           padding_mode='replicate')
        upsampling = nn.UpsamplingBilinear2d(
            scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1,
                      bias=False),  # 要采样的话在这里改变stride
            nn.BatchNorm2d(outchannel),  # 批处理正则化
            nn.ReLU(inplace=True),  # 激活
            nn.Conv2d(outchannel, outchannel, 3, 1, 1,
                      bias=False),  # 采样之后注意保持feature map的大小不变
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)  # 计算残差
        out += residual
        return nn.ReLU(inplace=True)(out)  # 注意激活
class feature_Fusion(nn.Module):
    def __init__(self):
        super(feature_Fusion, self ).__init__()
        self.GFpre = ResidualBlock(16,16,1)
        self.Sentipre= nn.Sequential(nn.Conv2d(32,16,kernel_size=1,stride=1,padding=0),ResidualBlock(16,16,1))
        self.CBAM=CBAMBlock(channel=16,reduction=4,kernel_size=7)
    def forward(self, feagf2,feasenti):
        fuse = self.Sentipre(feasenti)+self.GFpre(feagf2)
        out = self.CBAM(fuse)
        return out
class ULSTM(nn.Module):
    def __init__(self,inchannels,lstm_inputsize,mode):
        super(ULSTM, self ).__init__()
        self.Sobel =Sobel(inchannels)
        self.fFus =feature_Fusion()
        self.GFNet = Resnet34(1)
        self.SentiNet=BConvLSTM(input_size=(lstm_inputsize,lstm_inputsize),input_dim=4,hidden_dim=16,kernel_size=(3,3))
        self.fuseseg=SegmentationHead(in_channels=16)
        self.dropout = nn.Dropout(p=0.2) 
        self.mode=mode
        self.lstm_inputsize=lstm_inputsize
    def forward(self, Gf2,Senti):
        GF2sobel=self.Sobel(Gf2)
        _,_,h,w=Gf2.size()
        NewSenti=torch.zeros(Senti.shape[0],Senti.shape[1],Senti.shape[2],self.lstm_inputsize,self.lstm_inputsize).cuda()
        for i in range(36):
            NewSenti[:,i]=F.interpolate(Senti[:,i], size=(self.lstm_inputsize, self.lstm_inputsize), mode='bilinear')
        gf2fea=self.GFNet(GF2sobel)
        lowsentifea=self.SentiNet(NewSenti)
        sentifea=F.interpolate(lowsentifea,size=(h,w),mode='bilinear')
        if self.mode=='train':
            fusionfea=self.fFus(self.dropout(gf2fea),sentifea)
        else:
            fusionfea=self.fFus(gf2fea,sentifea)
        fusioncrop = self.fuseseg(fusionfea)
        pcrop = torch.sigmoid(fusioncrop)
        if self.mode=='train':
            return pcrop
        if self.mode=='GAN':
            return fusionfea,fusioncrop
        if self.mode=='test':
            return pcrop