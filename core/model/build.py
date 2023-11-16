from model.ULSTM import ULSTM
from model.UNetforboundary import UNet
from model.Discriminator import PixelDiscriminator
def build_model(cfg):
    model_name = cfg["MODEL"]["NAME"]
    if model_name=='ULSTM':
        model = ULSTM(inchannels=cfg["MODEL"]["INPUTCHANNELS"],lstm_inputsize=cfg["MODEL"]["LSTMSIZE"],mode=cfg["MODEL"]["MODE"])
    elif model_name=='UNetforboundary':
        model = UNet(mode=cfg["MODEL"]["MODE"])
    else:
        raise NotImplementedError
    return model

def build_discriminator(cfg):
    discriminator = PixelDiscriminator(input_nc=cfg["DISCRIMINATOR"]["INPUTCHANNELS"], ndf=cfg["DISCRIMINATOR"]["HIDDENDIMS"], num_classes=cfg["DISCRIMINATOR"]["OUTDIMS"])
    return discriminator