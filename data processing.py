#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import cv2
import os
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

# 对图片image和label进行弹性形变
def elastic_transform(image, label, alpha, sigma,
                      alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # 随机仿射变换矩阵
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # 三点法仿射变换
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
    labelB = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # 产生服从[0,1]均匀分布的随机迁移场
    # alpha控制变形强度
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # 迁移
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # 双线性插值
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    labelC = map_coordinates(labelB, indices, order=1, mode='constant').reshape(shape)

    return imageC, labelC

#对图像img进行扩充，大小变为原图九倍
def extend(img):
    img22 = img
    img12 = img32 = img[::-1,:]
    img21 = img23 = img[:,::-1]
    img11 = img13 = img12[:,::-1]
    img31 = img33 = img32[:,::-1]
    img1 = np.c_[img11,img12,img13]
    img2 = np.c_[img21,img22,img23]
    img3 = np.c_[img31,img32,img33]
    img = np.r_[img1,img2,img3]
    return img

#数据存储位置
data_path = 'D:\大学常用\机器学习\大作业\A\dataset' + '\\'
save_path = 'D:\大学常用\机器学习\大作业\A\dataset\\new1' + '\\'

Data = 'train_img' + '\\'
Label = 'train_label' + '\\'
cnt = 0
for file in os.listdir(data_path + Data):
    #读入图片和标签
    img = cv2.imdecode(np.fromfile(data_path + Data + file, dtype=np.uint8), cv2.IMREAD_COLOR)
    label = cv2.imdecode(np.fromfile(data_path + Label + file, dtype=np.uint8), cv2.IMREAD_COLOR)

    #转为灰度图
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

    w,h = img.shape
    data = []
    lab = []
    data.append(img)
    lab.append(label)

    # 对图像img进行扩充，大小变为原图九倍
    img = extend(img)
    label = extend(label)
    img, label = elastic_transform(img, label,  img.shape[1] * 2,
                                   img.shape[1] * 0.08,
                                   img.shape[1] * 0.08)
    # 截取图片中心部分
    img = img[w:w * 2, h:h * 2]
    label = label[w:w * 2, h:h * 2]

    data.append(img)
    lab.append(label)

    # 对每张图片和标签分别旋转90°，180°，270°
    for i in range(2):
        img = data[i]
        label = lab[i]

        img90 = np.rot90(img)
        img180 = np.rot90(img90)
        img270 = np.rot90(img180)
        data.append(img90)
        data.append(img180)
        data.append(img270)

        lab90 = np.rot90(label)
        lab180 = np.rot90(lab90)
        lab270 = np.rot90(lab180)
        lab.append(lab90)
        lab.append(lab180)
        lab.append(lab270)
    # 将200张图片进行保存
    for i in range(len(data)):
        img = data[i]
        label = lab[i]
        cv2.imencode('.jpg', img)[1].tofile(save_path + Data + str(cnt) + '.jpg')
        cv2.imencode('.jpg', label)[1].tofile(save_path + Label + str(cnt) + '.jpg')
        cnt = cnt + 1

    # 以下代码为分割图片，最终模型训练过程未使用下述代码生成的数据
    # 对每张图片进行分割，步长为1 / 6 *512，图片大小为512 / 24 *5，每张图片分割为25份，共得5000张图片
    # for i in range(len(data)):
    #     img = data[i]
    #     label = lab[i]
    #     d = int(512 / 24 * 5) #图片大小为512 / 24 *5
    #     x = 0
    #     y = 0
    #     while x + d < 513:
    #         y = 0
    #         while y + d < 513:
    #             cv2.imencode('.jpg', img[x:x + d, y:y + d])[1].tofile(save_path + Label + str(cnt) + '.jpg')
    #             cv2.imencode('.jpg', label[x:x + d, y:y + d])[1].tofile(save_path + Label + str(cnt) + '.jpg')
    #             cnt = cnt + 1
    #             y = y + int(1 / 6 * 512) #步长为1 / 6 *512
    #         x = x + int(1 / 6 * 512)