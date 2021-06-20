""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
import  numpy as np

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #U型网络下降部分
        self.inc = DoubleConv(n_channels, 45)
        self.down1 = Down(45, 90)
        self.down2 = Down(90, 180)
        self.down3 = Down(180, 360)
        factor = 2 if bilinear else 1
        self.down4 = Down(360, 720 // factor)

        #U型网络上升部分
        self.up1 = Up(720, 360 // factor, bilinear)
        self.up2 = Up(360, 180 // factor, bilinear)
        self.up3 = Up(180, 90 // factor, bilinear)
        self.up4 = Up(90, 45, bilinear)
        self.outc = OutConv(45, n_classes)

        #获取各个跳跃层的gate
        self.get_gate1=Get_gate(512*512*45)
        self.get_gate2=Get_gate(256*256*90)
        self.get_gate3=Get_gate(128*128*180)
        self.get_gate4=Get_gate(64*64*360)

        self.batch=torch.nn.Sequential(
            torch.nn.LayerNorm(4)
        )



    def forward(self, x):
        # U型网络下降部分
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        #展成一维数据
        x1_1dim = x1.view(x1.size(0), -1)
        x2_1dim = x2.view(x1.size(0), -1)
        x3_1dim = x3.view(x1.size(0), -1)
        x4_1dim = x4.view(x1.size(0), -1)



        print(x1_1dim.size())
        print(x2_1dim.size())
        print(x3_1dim.size())
        print(x4_1dim.size())

        #获取gate值
        gate_1=self.get_gate1(x1_1dim)
        gate_2=self.get_gate2(x2_1dim)
        gate_3=self.get_gate3(x3_1dim)
        gate_4=self.get_gate4(x4_1dim)


        #gate值拼接，归一化
        gate_1234=torch.cat([gate_1,gate_2,gate_3,gate_4], dim=1)
        gate_all=self.batch(gate_1234)
        gate_all=nn.functional.softmax(gate_all,dim=1)*2

        #输出gate值
        # print(str(gate_all[0][0])+'\n')
        # print(str(gate_all[0][1])+'\n')
        # print(str(gate_all[0][2])+'\n')
        # print(str(gate_all[0][3])+'\n')

        #U型网络上升部分
        x = self.up1(x5, x4,gate_all[0][0])
        x = self.up2(x, x3,gate_all[0][1])
        x = self.up3(x, x2,gate_all[0][2])
        x = self.up4(x, x1,gate_all[0][3])
        logits = self.outc(x)
        return logits
