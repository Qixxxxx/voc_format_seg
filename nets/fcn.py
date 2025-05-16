import torch.nn.functional as F
import torch
import torch.nn as nn
from .backbone.resnet import get_backbone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 网络结构
class MyNet(nn.Module):
    def __init__(self, num_classes, down_rate, backbone='resnet50', pretrained=True, aux_branch=True):
        '''num_classes: 分类数, down_rate： 经过特征提取后的下采样倍数, backbone：特征提取网络的使用, pretrained：是否加载预训练权重,
         aux_branch：是否使用辅助loss'''
        super(MyNet, self).__init__()
        # 根据特征提取网络的输出维度确定psp_module的输入维度
        out_channel = 2048
        self.backbone = get_backbone(backbone, pretrained, outstride=down_rate)

        self.final_seg = nn.Sequential(
            nn.Conv2d(out_channel, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        self.aux_branch = aux_branch

        self.auxiliary_branch = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

        self.initialize_weights(self.final_seg)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        (x0, x1, x2, x3, x4) = self.backbone(x)  # backbone输出

        seg_final = self.final_seg(x4)
        output = F.interpolate(seg_final, size=input_size, mode='bilinear', align_corners=True)

        # 使用辅助loss
        if self.aux_branch:
            output_aux = self.auxiliary_branch(x3)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()
