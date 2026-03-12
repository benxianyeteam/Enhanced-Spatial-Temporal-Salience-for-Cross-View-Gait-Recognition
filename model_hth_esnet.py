import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import os
# from BERT import BERT5
# from resnet_feature import resnet10
from torch.nn.parameter import Parameter

class B3DSTA1Block(nn.Module):
#  '''对应Pi-Conv'''
    def __init__(self, in_planes, out_planes, STABlock):
        super(B3DSTA1Block, self).__init__()
        self.attn = STABlock
        self.conv1 = nn.Conv3d(in_planes, out_planes, kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.conv2 = nn.Conv3d(in_planes, out_planes, kernel_size=(1,3,3), stride=(1,1,1), padding=(0,1,1), bias=False)
        self.conv3 = nn.Conv3d(in_planes, out_planes, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False)
        self.adjust = nn.Conv3d(in_planes, out_planes, kernel_size=(1,1,1),stride=(1,1,1), padding=(0,0,0), bias=False)
    def forward(self, x_raw):
        #a = self.attn(x_raw)[1]
        #b = self.attn(x_raw)[2]
        #x = x_raw + x_raw*self.attn(x_raw)[0]
        x_raw = self.adjust(x_raw)
        x = x_raw + x_raw * self.attn(x_raw)
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = x_raw + x  ###残差链接
        x = F.leaky_relu(x,inplace=True)
        return x #, a, b


class STE2(nn.Module):
    '''对应STDA,3d卷积的时空注意力,空间注意力使用mean，这个好'''
    def __init__(self):
        super(STE2, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3,3), stride=(1,1), padding=1, bias=False)
        self.conv2 = nn.Conv3d(1, 1, kernel_size=(3,1,1), stride=(1,1,1), padding=(1,0,0), bias=False)
    def forward(self, x):
        #x [b,c,t,h,w]
        b,c,t,h,w = x.shape
        x_s = x.mean(1, keepdim=True)
        x_s = x_s.mean(2)
        x_score1 = self.conv1(x_s)
        x_score1 = x_score1.unsqueeze(2)

        x_t = x.mean(1)
        x_t = F.avg_pool2d(x_t, x_t.size()[2:])
        x_t = x_t.unsqueeze(1)
        x_score2 = self.conv2(x_t)
        x_score = x_score1*x_score2
        x_score = torch.sigmoid(x_score)

        del x_s,x_t
        return x_score


class MultiSpan(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,  **kwargs):
        super(MultiSpan, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=1, dilation=1, bias=False, **kwargs)
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=2, dilation=2, bias=False, **kwargs)
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=4, dilation=4, bias=False, **kwargs)
        #self.bn1 = nn.BatchNorm1d(out_channels)
        #self.bn2 = nn.BatchNorm1d(out_channels)
        #self.bn3 = nn.BatchNorm1d(out_channels)
    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(-1, c, h, w)
        n, c, h, w = x.size()
        x = x.view(n, c, -1)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat((x1,x2,x3), 1)
        n, c, _ = x.size()
        #x = F.leaky_relu(x, inplace=True)
        x = x.view(n, c, h, w)
        x = x.view(b, -1, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        del x1,x2
        return x


class ESNet(nn.Module):
    '''
    一支全局+STA+MS, 用了B3DBlock，,v1指B3DBlock, res指用了残差结构,softmax,是自己提出的ESTSE网络！！！！
    '''
    def __init__(self, planes_list=[1, 32, 64, 128], bin_num=32, hidden_dim=128):
        super(ESNet, self).__init__()
        self.planes_list = planes_list
        self.bin_num = bin_num

        '''注意力'''
        self.hidden_dim = hidden_dim
        self.attn1 = STE2()
        self.attn2 = STE2()
        self.attn3 = STE2()
        self.layer1 = nn.Conv3d(self.planes_list[0], self.planes_list[1], kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1), bias=False)
        self.layer2 = nn.Conv3d(self.planes_list[1], self.planes_list[1], kernel_size=(3,1,1), stride=(3,1,1), padding=(1,0,0), bias=False)
        self.layer3 = B3DSTA1Block(self.planes_list[1], self.planes_list[2], self.attn1)
        self.layer4 = B3DSTA1Block(self.planes_list[2], self.planes_list[3], self.attn2)
        self.layer5 = B3DSTA1Block(self.planes_list[3], self.planes_list[3], self.attn3)
        #self.layer1 = LOCALBlock1(self.planes_list[1], self.planes_list[2])
        #self.layer2 = LOCALBlock1(self.planes_list[2], self.planes_list[3])
        #self.layer3 = LOCALBlock1(self.planes_list[3], self.planes_list[3])


        self.ms = MultiSpan(self.planes_list[3], self.planes_list[3])  #MultiSpan()对应MSSD
        self.fc_bin = nn.ParameterList([nn.Parameter(nn.init.xavier_uniform_(torch.zeros(self.bin_num, 3*self.planes_list[3], self.hidden_dim)))])

        #self.Gem = GeM()
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)

    def forward(self, x):
    #---------------------------------------------------------
        n, t, h, w = x.size()
        if t == 1:
            x = x.repeat(1, 3, 1, 1)
        elif t == 2:
            x = x.repeat(1, 2, 1, 1)
        elif t == 3:
            x = torch.cat((x, x[:, 0:1, :, :]), dim=1)
        #x [b ,c, t, h, w]
        x = x.unsqueeze(2).permute(0,2,1,3,4).contiguous()
    #---------------------------------------------------------
        x = F.leaky_relu(self.layer1(x), inplace=True)
        x = F.leaky_relu(self.layer2(x), inplace=True)
        #x, a1, b1 = self.layer3(x)
        x = self.layer3(x)
        x = F.max_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        x = self.layer4(x)
        x = self.layer5(x)
        #multispan
        x = self.ms(x)
        x = torch.max(x, 2)[0]  #frame_pooling

        #Feature mapping
        b, c, h, w = x.size()
        f = x.view(b, c, self.bin_num, -1)
        
        '''MSSA'''
        #---------------------------------
        f_score = torch.softmax(f, dim=3)
        #print(f_score[0,0,0,:])
        #print(torch.sum(f_score[0,0,0,:]))
        f = f*f_score
        f = torch.sum(f, dim=3)
        #f = f.mean(3) + f.max(3)[0]
        #f = self.Gem(f).squeeze(-1)
        #---------------------------------
        
        f = f.permute(2, 0, 1).contiguous()
        f = f.matmul(self.fc_bin[0])
        f = f.permute(1, 0, 2).contiguous()   #[b, 16, 256]
        #a = []
        #a.append(a1)
        #a.append(a2)
        #a.append(a3)
        del x, f_score
        return f


es=ESNet()
x=torch.randn([8,30,64,44])
out=es(x)

