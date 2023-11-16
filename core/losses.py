import torch.nn as nn
import torch
import torch.nn.functional as F

class TanimotoLoss(nn.Module):
    def __init__(self):
        super(TanimotoLoss, self).__init__()

    def tanimoto(self, x, y):
        epo=1e-10
        # print((x * x).sum(1) + (y * y).sum(1) - (x * y).sum(1))
        result=(x * y).sum(1)/ ((x * x).sum(1) + (y * y).sum(1) - (x * y).sum(1)+epo)
        return result
    def forward(self, pre, tar):
        '''
        pre and tar must have same shape. (N, C, H, W)
        var(N,H,W)
        '''
        pre=torch.sigmoid(pre)
        N = tar.size()[0]
        # 将宽高 reshape 到同一纬度
        input_flat = pre.reshape(N, -1)
        targets_flat = tar.reshape(N, -1)
        # var_flat = var.view(N,-1)

        t1 = self.tanimoto(input_flat, targets_flat)
        # t2 = self.tanimoto(1.0-input_flat,1.0-targets_flat)
        
        loss = 1-t1
        # t_ = self.tanimoto(1 - input_flat, 1 - targets_flat)
        # loss = t + t_
        loss = loss.mean()
        return loss
    

def soft_label_cross_entropy(pred, soft_label,count_pos,count_neg):
    beta_1 = count_pos / (count_pos + count_neg)
    beta_2 = count_neg / (count_pos + count_neg)
    soft_label[:,0]=soft_label[:,0]*beta_2
    soft_label[:,1]=soft_label[:,1]*beta_1
    soft_label[:,2]=soft_label[:,2]*beta_2
    soft_label[:,3]=soft_label[:,3]*beta_1
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)

    return torch.mean(torch.sum(loss, dim=1))