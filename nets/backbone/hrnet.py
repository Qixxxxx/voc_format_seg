import torch.nn as nn

ARCH_SETTINGS = {
    'hrnetv2_w18_small': {
        'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (2,),
                   'num_channels': (64,), },
        'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (2, 2),
                   'num_channels': (18, 36), },
        'stage3': {'num_modules': 3, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (2, 2, 2),
                   'num_channels': (18, 36, 72), },
        'stage4': {'num_modules': 2, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (2, 2, 2, 2),
                   'num_channels': (18, 36, 72, 144), },
    },
    'hrnetv2_w18': {
        'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,),
                   'num_channels': (64,), },
        'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4),
                   'num_channels': (18, 36), },
        'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4),
                   'num_channels': (18, 36, 72), },
        'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4),
                   'num_channels': (18, 36, 72, 144), },
    },
    'hrnetv2_w32': {
        'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,),
                   'num_channels': (64,), },
        'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4),
                   'num_channels': (32, 64), },
        'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4),
                   'num_channels': (32, 64, 128), },
        'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4),
                   'num_channels': (32, 64, 128, 256), },
    },
    'hrnetv2_w40': {
        'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,),
                   'num_channels': (64,), },
        'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4),
                   'num_channels': (40, 80), },
        'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4),
                   'num_channels': (40, 80, 160), },
        'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4),
                   'num_channels': (40, 80, 160, 320), },
    },
    'hrnetv2_w48': {
        'stage1': {'num_modules': 1, 'num_branches': 1, 'block': 'BOTTLENECK', 'num_blocks': (4,),
                   'num_channels': (64,), },
        'stage2': {'num_modules': 1, 'num_branches': 2, 'block': 'BASIC', 'num_blocks': (4, 4),
                   'num_channels': (48, 96), },
        'stage3': {'num_modules': 4, 'num_branches': 3, 'block': 'BASIC', 'num_blocks': (4, 4, 4),
                   'num_channels': (48, 96, 192), },
        'stage4': {'num_modules': 3, 'num_branches': 4, 'block': 'BASIC', 'num_blocks': (4, 4, 4, 4),
                   'num_channels': (48, 96, 192, 384), },
    },
}


# ---------------------------------------#
#   hrnet的模块也是残差模块
# ---------------------------------------#
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, norm_layer=None, act_cfg=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
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


class HRNet(nn.Module):
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(self, arch='hrnetv2_w48', norm_layer=nn.SyncBatchNorm, act_cfg=nn.ReLU(inplace=True)):
        super(HRNet, self).__init__()
        self.arch = arch
        self.act_cfg = act_cfg
        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = norm_layer(64)
        self.relu = act_cfg
        self.stages_cfg = ARCH_SETTINGS[arch]

        # stage1
        self.stage1_cfg = self.stages_cfg['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]
        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * block.expansion
        self.layer1 = self.makelayer(block, 64, num_channels, num_blocks, norm_layer=norm_layer, act_cfg=act_cfg)

    def makelayer(self, block, inplanes, planes, num_blocks, stride=1, norm_layer=None, act_cfg=None):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample, norm_layer=norm_layer, act_cfg=act_cfg))
        inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(
                block(inplanes, planes, norm_layer=norm_layer, act_cfg=act_cfg)
            )
        return nn.Sequential(*layers)
