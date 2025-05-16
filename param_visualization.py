import torch
from torchsummary import summary

from nets.fcn import MyNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# from nets.hrnet_seg import MyNet
# model = MyNet(num_classes=2, backbone="hrnetv2_w48", down_rate=8, pretrained=False, aux_branch=False).train().to(device)
# summary(model, (3, 256, 256))

model = MyNet(num_classes=2, backbone="resnet50", down_rate=8, pretrained=False, aux_branch=False).train().to(device)
summary(model, (3, 256, 256))
