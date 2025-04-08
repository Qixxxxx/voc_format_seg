import colorsys
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from nets.net import MyNet
from utils.common_util import cvtColor, divide_255, resize_image, show_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PredictModel(object):
    _defaults = {
        # -------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        # -------------------------------------------------------------------#
        "model_path": 'model_data/net_81.12.pth',
        "num_classes": 21,
        # ----------------------------------------#
        #   所使用的的主干网络：
        # ----------------------------------------#
        "backbone": "resnet50conv3x3stem",
        # ----------------------------------------#
        #   输入图片的大小
        # ----------------------------------------#
        "input_shape": [512, 512],
        # ----------------------------------------#
        #   下采样的倍数，与训练时设置的一样即可
        # ----------------------------------------#
        "downsample_factor": 8,
        # -------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #   mix_type = 0的时候代表仅保留生成的图
        #   mix_type = 1的时候代表原图与生成的图进行混合
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        # -------------------------------------------------#
        "mix_type": 0,
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   画框设置不同的颜色
        # ---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                           (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
                           (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
                           (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

        show_config(**self._defaults)

    def generate(self):
        # -------------------------------#
        #   载入模型与权值
        # -------------------------------#
        self.net = MyNet(num_classes=self.num_classes, backbone=self.backbone, down_rate=self.downsample_factor,
                         pretrained=False, aux_branch=False)

        self.net = self.net.eval()
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        print('{} model, and classes loaded.'.format(self.model_path))

    """
    获得可视化图
    """

    def get_image(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------#
        #   对输入图像进行一个备份，后面用于绘图
        # ---------------------------------------------------#
        old_img = copy.deepcopy(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   不失真的resize(给图像增加灰条)
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   处理维度为 b, c, h, w
        # ---------------------------------------------------------#
        image_data = [np.array(image_data, np.float32) / 255]
        image_data = np.transpose(image_data, (0, 3, 1, 2))

        with torch.no_grad():
            images = torch.from_numpy(image_data).type(torch.FloatTensor)
            if self.cuda:
                images = images.cuda()

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        if self.mix_type == 0:
            seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            for c in range(self.num_classes):
                seg_img[:, :, 0] += ((pr[:, :] == c) * self.colors[c][0]).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * self.colors[c][1]).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * self.colors[c][2]).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))

        elif self.mix_type == 1:
            seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            for c in range(self.num_classes):
                seg_img[:, :, 0] += ((pr[:, :] == c) * self.colors[c][0]).astype('uint8')
                seg_img[:, :, 1] += ((pr[:, :] == c) * self.colors[c][1]).astype('uint8')
                seg_img[:, :, 2] += ((pr[:, :] == c) * self.colors[c][2]).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h))
            #  blend混合
            image = Image.blend(old_img, image, 0.7)
        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            image = Image.fromarray(np.uint8(seg_img))

        return image

    """
    获得预测的得分图
    """

    def get_predict_score_image(self, image):
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        image_data = [np.array(image_data, np.float32) / 255]
        image_data = np.transpose(image_data, (0, 3, 1, 2))

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            pr = self.net(images)[0]
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh),
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]

        image = Image.fromarray(np.uint8(pr)).resize((orininal_w, orininal_h), Image.NEAREST)
        return image

    def get_FPS(self, image, test_interval):
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                self.net(image)

        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time
