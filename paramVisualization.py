import torch
from torchsummary import summary

from nets.pspnet import MyNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = MyNet(num_classes=2, backbone="resnet101conv3x3stem", down_rate=8,
              aux_branch=False, pretrained=True).train().to(device)

summary(model, (3, 256, 256))
