import torch
import numpy as np
from torch import nn
from torch.autograd import Variable

class Laplace(nn.Module):
    def __init__(self):
        super(Laplace, self ).__init__()

    def nn_conv2d(self,im):
        conv_op = nn.Conv2d(4, 1, 3, bias=False,padding=1)
        sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]*4, dtype='float32')
        sobel_kernel = sobel_kernel.reshape((1, 4, 3, 3))
        conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
        for param in conv_op.parameters():
            param.requires_grad = False  
        edge_detect = conv_op(Variable(im))
        return edge_detect
    def forward(self, GF2):
        GF2=self.nn_conv2d(GF2)
        return GF2

class Sobel(nn.Module):
    def __init__(self,inchannels,mode=None):
        super(Sobel, self ).__init__()
        self.inchannels=inchannels
        self.mode=mode
    def Gx(self,im):
        conv_op = nn.Conv2d(self.inchannels, 1, 3, bias=False,padding=1,padding_mode='replicate')
        sobel_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]*self.inchannels, dtype='float32')
        sobel_kernel = sobel_kernel.reshape((1, self.inchannels, 3, 3))
        conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
        for param in conv_op.parameters():
            param.requires_grad = False  
        edge_detect = conv_op(Variable(im))
        return edge_detect

    def Gy(self,im):
        conv_op = nn.Conv2d(self.inchannels, 1, 3, bias=False,padding=1,padding_mode='replicate')
        sobel_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]*self.inchannels, dtype='float32')
        sobel_kernel = sobel_kernel.reshape((1, self.inchannels, 3, 3))
        conv_op.weight.data = torch.from_numpy(sobel_kernel).cuda()
        for param in conv_op.parameters():
            param.requires_grad = False  
        edge_detect = conv_op(Variable(im))
        return edge_detect

    def forward(self, GF2):
        gx=self.Gx(GF2)
        gy=self.Gy(GF2)
        sobel=abs(gx)+abs(gy)
        # sobel=gx**2+gy**2
        if self.mode:
            grad_angle = torch.arctan2(gy, gx)
            return sobel,grad_angle
        return sobel