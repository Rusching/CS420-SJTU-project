import argparse
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
torch.set_printoptions(profile="full")


#训练图片目录
dir_train_img='data/processed/train_img/'
#训练标签目录
dir_train_label='data/processed/train_label/'
#测试图片目录
dir_test_img='data/test/test_img/'
#测试标签目录
dir_test_label='data/test/test_label/'
#训练中保存模型目录
dir_checkpoint = 'checkpoints/'

#超参
EPOCHS=50
BATCH_SIZE=1
LR=0.001
#验证集占比，取值0-100，指0%至100%
VAL_PERCENT=10
#是否在训练中保存模型checkpoint
SAVE_CP= True
#读入图片时的放缩，设置为1即可
IMG_SCALE=1


#RESUME=True表示由训练结果继续训练，为False表示重新训练
# RESUME=True
RESUME=False
#输入模型，仅当由训练结果继续训练时使用
input_model='checkpoints/CP_epoch3.pth'


#定义训练过程
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=1):

    #加载训练数据
    train=BasicDataset(dir_train_img, dir_train_label, img_scale)
    n_train=len(train)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    #加载测试数据
    val=BasicDataset(dir_test_img, dir_test_label, img_scale)
    n_val=len(val)
    val_loader = DataLoader(val, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)


    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')

    #记录完成的Batch个数
    global_step = 0

    #显示模型的参数信息
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    #优化器
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    #调度器，训练15代和25代后将学习率调整为0.1倍
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 25], gamma=0.1)
    #损失函数
    criterion = nn.BCEWithLogitsLoss()

    #从哪一epoch开始训练
    start_epoch=-1

    #如果从已有模型开始训练
    if(RESUME):
        #加载模型
        checkpoint = torch.load(input_model)
        #加载模型参数
        net.load_state_dict(checkpoint['net'])
        #加载优化器
        optimizer.load_state_dict(checkpoint['optimizer'])
        #加载之前训练结束时的epoch，从此开始训练
        start_epoch=checkpoint['epoch']
        #加载调度器
        scheduler.load_state_dict(checkpoint['scheduler'])

    #记录最好得分
    best_val_score=0
    #记录最好得分的epoch数
    best_epoch=0
    for epoch in range(start_epoch+1,epochs):
        #训练模式
        net.train()
        #记录本epoch的loss值
        epoch_loss = 0

        #加载tqdm库，显示进度条
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                #原数据
                imgs = batch['image']
                #原标签
                true_masks = batch['mask']
                #断言指令，保证图片通道数和输入通道数一致
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                #获得预测标签
                masks_pred = net(imgs)

                #损失函数
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                #更新参数
                optimizer.zero_grad()
                loss.backward()
                #裁剪梯度，最大绝对值为0.1
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()


                pbar.update(imgs.shape[0])
                #更新完成的batch数
                global_step += 1

                #每完成10个batch，显示一次信息
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)

                    val_score = eval_net(net, val_loader, device)

                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)
                    writer.add_images('images', imgs, global_step)
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        #创建模型保存路径
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass

            #将网络，优化器，epoch数，调度器进行打包
            checkpoint={
                "net":net.state_dict(),
                "optimizer":optimizer.state_dict(),
                "epoch":epoch,
                "scheduler":scheduler.state_dict()
            }
            #进行保存
            torch.save(checkpoint,
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

        #更新最高得分和对应epoch数
        if (val_score>best_val_score):
            best_val_score=val_score
            best_epoch=epoch
        #每个epoch后，将相关信息写入score.txt文件中
        record = open('result/score.txt', mode='a')
        record.write("score:{}, loss:{}, epoch:{}, batchsize:{}, origin learning rate:{}, data: origin data \n".format(
            format(val_score,'.4f'),format(epoch_loss,'.4f'),epoch+1, batch_size, lr))
        record.close()
    #将最优的评分和对应epoch数写入score.txt文件中
    record = open('result/score.txt', mode='a')
    record.write(
        "Best score:{}, in epoch:{} \n".format(
            format(best_val_score, '.4f'),  best_epoch + 1,))
    record.close()
    torch.save(net, 'latest_model/latest.pth')
    logging.info('Latest model saved !')
    writer.close()


if __name__ == '__main__':
    #设置输出日志等级，以便观察
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    #设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    #定义网络
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    #print(net)
    #显示网络信息
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    net.to(device=device)

    #进行训练
    try:
        train_net(net=net,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE,
                  lr=LR,
                  device=device,
                  img_scale=IMG_SCALE,
                  val_percent=VAL_PERCENT / 100)

    #人为中断时，保存模型
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
