import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.dataset import BasicDataset

#预测图片输入目录
predict_input = 'data/test/test_img/'
#输入模型
input_model='checkpoints/CP_epoch.pth'
#预测图片输出目录
predict_output='result/masks_predict/'
#图片放缩，设置为1
IMG_SCALE=1

#定义预测函数
def predict_img(net,
                full_img,
                device,
                scale_factor=0.5,
                out_threshold=0.5):
    #评估模式
    net.eval()

    #图片读入
    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        #获取预测输出
        output = net(img)

        # 使用sigmoid函数，将输出结果映射到[0,1]范围内
        probs = torch.sigmoid(output)
        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    #输出预测mask,超过阈值部分为1，否则为0
    return full_mask > out_threshold


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    #建立网络
    net = UNet(n_channels=1, n_classes=1)

    logging.info("Loading model {}".format(input_model))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)

    #加载网络参数
    checkpoint = torch.load(input_model)
    net.load_state_dict(checkpoint['net'])

    logging.info("Model loaded !")


    for root,_,files in os.walk(predict_input):
        for fn in files:
            logging.info("\nPredicting image {} ...".format(fn))
            img = Image.open(os.path.join(root, fn))

            mask = predict_img(net=net,
                                full_img=img,
                                scale_factor=1,
                                out_threshold=0.5,
                                device=device)


            result = mask_to_image(mask)
            result.save(predict_output+'_'+fn)
            result.save(predict_output+'_'+fn)
            logging.info("Mask saved to "+predict_output+fn)


