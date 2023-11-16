from torch import nn
import torch.nn.functional as F
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=128, num_classes=4):
        super(PixelDiscriminator, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)
        self.cls = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size=None):
        out = self.D(x)
        s_t_out = self.cls(out)
        if size is not None:
            s_t_out = F.interpolate(s_t_out, size=size, mode='bilinear', align_corners=True)
        return s_t_out