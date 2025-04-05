import torch.nn.functional as F
import torch
import torch.nn as nn
from backbone.resnet import get_backbone

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SPPMoudle(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes):
        ''' in_channels: 输入spp模块的通道维度， pool_sizes: 池化核大小组成的list, out_channels:输出通道数'''
        super(SPPMoudle, self).__init__()

        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size)
                                     for pool_size in pool_sizes])

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)  # 用全局平均池化将特征图缩放
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 每个stage输出通道一致
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]  # 创建一个list里面已经存在原输入
        # 使用extend将pspmodule产生的特征图添加入列表
        pyramids.extend(
            [F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


# 网络结构
class MyNet(nn.Module):
    def __init__(self, num_classes, down_rate, backbone='resnet50', pretrained=True, aux_branch=True):
        '''num_classes: 分类数, down_rate： 经过特征提取后的下采样倍数, backbone：特征提取网络的使用, pretrained：是否加载预训练权重,
         aux_branch：是否使用辅助loss'''
        super(MyNet, self).__init__()
        # 根据特征提取网络的输出维度确定psp_module的输入维度
        out_channel = 2048
        self.backbone = get_backbone(backbone, pretrained, outstride=down_rate)
        self.spp = SPPMoudle(out_channel, 512, pool_sizes=[1, 2, 3, 6])

        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        self.aux_branch = aux_branch

        if self.aux_branch:
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
        (x1, x2, x3, x4) = self.backbone(x)  # backbone输出

        spp_out = self.spp(x4)
        seg_final = self.final_seg(spp_out)
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
