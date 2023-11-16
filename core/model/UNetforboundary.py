'''
该模型是最基础的，benchmark的模型。
'''

import torch.nn as nn, torch
from torchvision.models.resnet import ResNet
from torchvision.models.resnet import BasicBlock
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class ResnetEncoder(ResNet):
    in_channels = 4  # R, G, B, Nir

    def __init__(self, block, layers, url, **kwargs):
        super(ResnetEncoder, self).__init__(block, layers, **kwargs)

        zoo_load = model_zoo.load_url(url)
        self.load_state_dict(zoo_load)

        del self.fc
        del self.avgpool

        module = None
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                break
        module.in_channels = 4

        new_weight = torch.Tensor(
            module.out_channels,
            self.in_channels,
            *module.kernel_size
        )

        weight = module.weight.detach()
        for i in range(self.in_channels):
            new_weight[:, i] = weight[:, i % 3]
        new_weight = new_weight * (3 / self.in_channels)
        module.weight = nn.parameter.Parameter(new_weight)

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.relu),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()
        features = []
        for i in range(len(stages)):
            x = stages[i](x)
            features.append(x)
        return features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
    ):

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResnetDecoder(nn.Module):
    def __init__(
        self,
        encoder_channels,
        decoder_channels,
        use_batchnorm=True,
    ):
        super().__init__()

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        # combine decoder keyword arguments
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        x = features[0]
        skips = features[1:]

        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
        return x


def get_encoder():
    params = {
        # "out_channels": (3, 64, 64, 128, 256, 512),
        "block": BasicBlock,
        "layers": [3, 4, 6, 3],
        "url": "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
    }
    encoder = ResnetEncoder(**params)
    return encoder


def get_decoder():
    encoder_channels = (4, 64, 64, 128, 256, 512)
    decoder_channels = (256, 128, 64, 32, 16)
    decoder = ResnetDecoder(
        encoder_channels=encoder_channels,
        decoder_channels=decoder_channels,
    )
    return decoder


def get_segmentation_head():
    kernel_size = 3
    in_channels = 16
    out_channels = 1
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, kernel_size), padding=kernel_size // 2)
    return nn.Sequential(conv2d)


class UNet(nn.Module):

    @staticmethod
    def initialize_decoder(module):
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def initialize_head(module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def __init__(self,mode=None):
        super().__init__()
        self.encoder = get_encoder()
        self.decoder = get_decoder()
        self.segmentation_head = get_segmentation_head()

        # 初始化decoder和segmentation head中的Parameters
        self.initialize_decoder(self.decoder)
        self.initialize_head(self.segmentation_head)
        self.mode=mode

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        masks = self.segmentation_head(decoder_output)
        if self.mode=='GAN':
            return decoder_output,masks
        elif self.mode=='Train':
            return masks
        else:
            return torch.sigmoid(masks)
        

    @torch.no_grad()
    def predict(self, x):
        if self.training:
            self.eval()
        x = self.forward(x)
        return x
