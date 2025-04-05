import math
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# 默认权重地址
DEFAULT_MODEL_URLS = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet18conv3x3stem': 'https://download.openmmlab.com/pretrain/third_party/resnet18_v1c-b5776b93.pth',
    'resnet50conv3x3stem': 'https://download.openmmlab.com/pretrain/third_party/resnet50_v1c-2cccc1ad.pth',
    'resnet101conv3x3stem': 'https://download.openmmlab.com/pretrain/third_party/resnet101_v1c-e67eebb6.pth',
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None, act_cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = act_cfg
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None, act_cfg=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = act_cfg
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, use_conv3x3_stem=False, outstride=32, contract_dilation=True,
                 norm_layer=nn.BatchNorm2d, act_cfg=nn.ReLU(inplace=True)):

        self.inplanes = 128 if use_conv3x3_stem else 64
        super(ResNet, self).__init__()
        # 使用空洞卷积时的步长设置和空洞率设置
        outstride_to_strides_and_dilations = {
            8: ((1, 2, 1, 1), (1, 1, 2, 4)),
            16: ((1, 2, 2, 1), (1, 1, 1, 2)),
            32: ((1, 2, 2, 2), (1, 1, 1, 1)),
        }
        assert outstride in outstride_to_strides_and_dilations, 'unsupport outstride %s' % outstride
        stride_list, dilation_list = outstride_to_strides_and_dilations[outstride]

        if use_conv3x3_stem:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            )
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = act_cfg
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=layers[0], stride=stride_list[0], dilation=dilation_list[0], contract_dilation=contract_dilation, norm_layer=norm_layer, act_cfg=act_cfg)
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=layers[1], stride=stride_list[1], dilation=dilation_list[1], contract_dilation=contract_dilation, norm_layer=norm_layer, act_cfg=act_cfg)
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=layers[2], stride=stride_list[2], dilation=dilation_list[2], contract_dilation=contract_dilation, norm_layer=norm_layer, act_cfg=act_cfg)
        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=layers[3], stride=stride_list[3], dilation=dilation_list[3], contract_dilation=contract_dilation, norm_layer=norm_layer, act_cfg=act_cfg)

    def _make_layer(self, block, planes, num_blocks, stride=1, dilation=1, contract_dilation=True, norm_layer=None, act_cfg=None):
        downsample = None
        dilations = [dilation] * num_blocks
        if contract_dilation and dilation > 1:
            dilations[0] = dilation // 2

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilations[0], downsample=downsample, norm_layer=norm_layer, act_cfg=act_cfg))
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, stride=1, dilation=dilations[i], norm_layer=norm_layer, act_cfg=act_cfg))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return tuple([x1, x2, x3, x4])



def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(DEFAULT_MODEL_URLS['resnet18']), strict=False)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(DEFAULT_MODEL_URLS['resnet34']), strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(DEFAULT_MODEL_URLS['resnet50']), strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(load_url(DEFAULT_MODEL_URLS['resnet101']), strict=False)
    return model

def resnet18_conv3x3stem(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], use_conv3x3_stem=True, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(DEFAULT_MODEL_URLS['resnet18conv3x3stem']), strict=False)
    return model

def resnet50_conv3x3stem(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], use_conv3x3_stem=True, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(DEFAULT_MODEL_URLS['resnet50conv3x3stem']), strict=False)
    return model


def resnet101_conv3x3stem(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], use_conv3x3_stem=True, **kwargs)
    if pretrained:
        model.load_state_dict(load_url(DEFAULT_MODEL_URLS['resnet101conv3x3stem']), strict=False)
    return model


def get_backbone(backbone_name, pretrained=False, **kwargs):

    DEFAULT_MODEL = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet18conv3x3stem': resnet18_conv3x3stem,
        'resnet50conv3x3stem': resnet50_conv3x3stem,
        'resnet101conv3x3stem': resnet101_conv3x3stem,
    }

    return DEFAULT_MODEL[backbone_name](pretrained, **kwargs)
