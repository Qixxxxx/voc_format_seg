import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.common_util import cvtColor, divide_255

# gt子文件夹
GT_DIR = "images_gt"
# image子文件夹
IMG_DIR = "images"

class CustomDataset(Dataset):
    """
    annotation_lines: 图片名称列表
    input_shape: 输入的HW列表，ex：[256, 256]
    num_classes: 分割结果类别数量
    type_is_train: 当前运行模式是否为Train
    dataset_path: 数据集路径
    """
    def __init__(self, annotation_lines, input_shape, num_classes, type_is_train, dataset_path):
        super(CustomDataset, self).__init__()
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.type_is_train = type_is_train
        self.dataset_path = dataset_path

    def __len__(self):
        return self.length

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        image = cvtColor(image)
        label = Image.fromarray(np.array(label))
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape

        if not random:
            iw, ih = image.size
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

            label = label.resize((nw, nh), Image.NEAREST)
            new_label = Image.new('L', (w, h), (0))
            new_label.paste(label, ((w - nw) // 2, (h - nh) // 2))
            return new_image, new_label

        # ------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        # ------------------------------------------#
        new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)

        image = image.resize((nw, nh), Image.BICUBIC)
        label = label.resize((nw, nh), Image.NEAREST)
        label = label.convert("L")

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        # ------------------------------------------#
        #   将图像多余的部分加上灰条
        # ------------------------------------------#
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_label = Image.new('L', (w, h), (0))
        new_image.paste(image, (dx, dy))
        new_label.paste(label, (dx, dy))
        image = new_image
        label = new_label

        # ------------------------------------------#
        #   颜色抖动  RGB->HVS->RGB
        # ------------------------------------------#
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        return image_data, label

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name = annotation_line.split()[0]

        # -------------------------------#
        #   从文件中读取图像
        # -------------------------------#
        image = Image.open(os.path.join(os.path.join(self.dataset_path, IMG_DIR), name + ".jpg"))
        label = Image.open(os.path.join(os.path.join(self.dataset_path, GT_DIR), name + ".png"))
        # -------------------------------#
        #   数据增强
        # -------------------------------#
        image, label = self.get_random_data(image, label, self.input_shape, random=self.type_is_train)
        label = np.array(label)
        # 未注明的类视为背景
        label[label >= self.num_classes] = self.num_classes
        # -------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        # -------------------------------------------------------#
        seg_labels = np.eye(self.num_classes + 1)[label.reshape([-1])]
        seg_labels = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes + 1))

        # pytorch中tensor默认是CHW,所以需要进行维度转换
        image = np.transpose(np.array(image), [2, 0, 1]) / 255
        """
        image: 输入图
        label: 标签图gt
        seg_labels: 分割标签数组，就是label的one-hot格式（one-hot长度为num_class+1）
        """
        return image, label, seg_labels




# DataLoader中collate_fn使用
def net_dataset_collate(batch):
    images = []
    pngs = []
    seg_labels = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images = np.array(images)
    pngs = np.array(pngs)
    seg_labels = np.array(seg_labels)
    return images, pngs, seg_labels
