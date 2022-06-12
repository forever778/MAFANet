
import numbers
from einops import rearrange
import numpy as np
import torch
from torch import nn
from torch.nn import init
import math
import torch.nn.functional as F

import torch.nn as nn


from My_model.backbone.backbone import Backbone_resnet

from L7_code.etc.flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
from My_model.model.BFF import BFF1, BFF


class ChannelAttention(nn.Module):

    def __init__(self, in_chan, out_chan, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)  
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chan, out_chan//2, 1, 1, 0, groups=out_chan//2),       
            nn.Conv2d(out_chan//2, in_chan, 1, 1, 0, groups=out_chan//2)

        )
        self.conv2 = nn.Conv2d(in_chan, out_chan, (1,2), (1,1), 0, groups=in_chan)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_chan,out_chan,1,groups=in_chan),
            nn.Sigmoid()
        )


    def forward(self,x):
        


        y_avg = self.avg_pool(x)
        y_avg = self.conv1(y_avg)
        y_max = self.max_pool(x)
        y_max = self.conv1(y_max)

        y1 = torch.cat([y_avg, y_max], dim=3)
        y2 = self.conv2(y1)
        y =self.conv3(y2)

        out = y*x

        return out

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)        
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )    

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x_compress = self.compress(x)      
        
        x_out = self.spatial(x_compress)
        scale = self.sigmoid(x_out) 

        return scale



class MCSA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(MCSA, self).__init__()
        pad0 = int((kernel_size[0] - 1) // 2)
        pad1 = int((kernel_size[1] - 1) // 2)
        super(MCSA, self).__init__()
        self.conv_l1 = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.conv_l2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size[1]),
                                 padding=(0, pad1))
        self.conv_r2 = nn.Conv2d(out_channels, out_channels, kernel_size=(kernel_size[0], 1),
                                 padding=(pad0, 0))
        self.bn = nn.BatchNorm2d(out_channels+1)
        self.relu = nn.ReLU(inplace=True)
        self.CA = ChannelAttention(in_channels, out_channels)
        self.SA = SpatialAttention()
        self.convout = nn.Conv2d(out_channels+1, out_channels, 1, 1, 0)

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        x_l = self.CA(x_l)
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        x_r = self.SA(x_r)


        x = torch.cat((x_l, x_r), dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.convout(x)

        return x


class SPBlock1(nn.Module):
    def __init__(self,  kernel=1):
        super(SPBlock1, self).__init__()
        self.avp1=nn.AvgPool2d(kernel_size=(1, kernel), stride=(1, 1), padding=(0, kernel // 2))
        self.avp2=nn.AvgPool2d(kernel_size=(kernel, 1), stride=(1, 1), padding=(kernel//2, 0))


    def forward(self, x):
        x1 = self.avp1(x)
        x2 = self.avp2(x)
        out1= x1+x2

        return out1

class SPBlock2(nn.Module):
    def __init__(self, kernel=3):
        super(SPBlock2, self).__init__()
        self.avp1 = nn.AvgPool2d(kernel_size=(1, kernel), stride=(1, 1), padding=(0, kernel // 2))
        self.avp2 = nn.AvgPool2d(kernel_size=(kernel, 1), stride=(1, 1), padding=(kernel // 2, 0))

    def forward(self, x):
        x1 = self.avp1(x)
        x2 = self.avp2(x)

        out2 = x1 + x2

        return out2

class SPBlock3(nn.Module):
    def __init__(self, kernel=5):
        super(SPBlock3, self).__init__()
        self.avp1 = nn.AvgPool2d(kernel_size=(1, kernel), stride=(1, 1), padding=(0, kernel // 2))
        self.avp2 = nn.AvgPool2d(kernel_size=(kernel, 1), stride=(1, 1), padding=(kernel // 2, 0))

    def forward(self, x):
        x1 = self.avp1(x)
        x2 = self.avp2(x)
        out3 = x1 + x2

        return out3

class SPBlock4(nn.Module):
    def __init__(self, kernel=6):
        super(SPBlock4, self).__init__()
        self.avp1 = nn.AvgPool2d(kernel_size=(1, kernel), stride=(1, 1), padding=(0, 2))
        self.avp2 = nn.AvgPool2d(kernel_size=(kernel, 1), stride=(1, 1), padding=(2, 0))
        self.up = nn.Upsample()

    def forward(self, x):
        x_size = x.size()

        x1 = self.avp1(x)
        x2 = self.avp2(x)

        x1 = F.interpolate(x1, x_size[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, x_size[2:], mode='bilinear', align_corners=True)


        out4 = x1 + x2
        return out4

class SPBlock5(nn.Module):
    def __init__(self):
        super(SPBlock5, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x_size = x.size()
        out5 = self.avg_pool(x)

        out6 = F.interpolate(out5, x_size[2:], mode='bilinear', align_corners=True)
        return out6



class SPB(nn.Module):
    def __init__(self):
        super(SPB, self).__init__()

        self.spb1 = SPBlock1()
        self.spb2 = SPBlock2()
        self.spb3 = SPBlock3()
        self.spb4 = SPBlock4()
        self.spb5 = SPBlock5()



    def forward(self, x):
        dsize = x.shape[2:]
        s1 = self.spb1(x)
        s2 = self.spb2(x)
        s3 = self.spb3(x)
        s4 = self.spb4(x)
        s5 = self.spb5(x)


        eout = torch.cat([x, s1, s2, s3, s4, s5],dim=2)
        eout =torch.nn.functional.interpolate(eout,size=dsize,mode='bilinear', align_corners=True)


        return(eout)

def to_3D(x):
    return rearrange(x,'b c h w -> b (h w) c')

def to_4D(x,h,w):
    return rearrange(x,'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self,normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()

        if isinstance(normalized_shape,numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self,x):
        sigma = x.var(-1,keepdim=True,unbiased=False)
        return x/torch.sqrt(sigma + 1e-5) * self.weight



class LayerNorm(nn.Module):
    def __init__(self,dim,LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self,x):
        h,w = x.shape[-2:]
        return to_4D(self.body(to_3D(x)),h,w)

class Feed_Forward(nn.Module):
    def __init__(self,in_ch,ratio,bias):
        super(Feed_Forward, self).__init__()

        hidden_features = int(in_ch * ratio)
        self.project_in = nn.Conv2d(in_ch,hidden_features*2,kernel_size=1,bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2,hidden_features*2,kernel_size=3,stride=1,
                                padding=1,groups=hidden_features*2,bias=bias)
        self.project_out = nn.Conv2d(hidden_features,in_ch,kernel_size=1,bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self,dim,num_heads,bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads,1,1))

        self.kv = nn.Conv2d(dim,dim*2,kernel_size=1,bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2,dim*2,kernel_size=3,stride=1,padding=1,
                                    groups=dim*2,bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim , dim , kernel_size=3, stride=1, padding=1,
                                    groups=dim , bias=bias)
        self.project_out = nn.Conv2d(dim,dim,kernel_size=1,bias=bias)

    def forward(self, x, q):
        b,c,h,w = x.shape
        kv = self.kv(x)
        kv = self.kv_dwconv(kv)
        k,v = kv.chunk(2,dim=1)
        q = self.q(q)
        q = self.q_dwconv(q)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head = self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head = self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head = self.num_heads)

        q = F.normalize(q, dim = -1)
        k = F.normalize(k, dim = -1)

        attn = (q @ k.transpose(-2,-1)) * self.temperature

        attn = attn.softmax(dim=-1)

        out = attn @ v

        out = rearrange(out,'b head c (h w) -> b (head c) h w',head=self.num_heads,h=h,w=w)
        out = self.project_out(out)
        return out


class DMFA(nn.Module):
    def __init__(self, dim, num_heads=4, ratio=2.66, bias=False, LayerNorm_type='BiasFree'):
        super(DMFA, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.atten = Attention(dim, num_heads,bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = Feed_Forward(dim, ratio, bias)

    def forward(self, x, y):
        out1 = y + x + self.atten(self.norm1(x), self.norm2(y))
        out = out1 + self.ffn(self.norm2(out1))
        return out


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat

class BRB(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes):
        super(BRB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chan, in_chan, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chan, in_chan, 3, 1, 1),
        )
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, stride=1,padding=0, bias=True)

    def forward(self, x, size=(224, 224)):
        feat = self.conv1(x)
        feat = x+feat
        feat = self.conv(feat)
        feat = self.drop(feat)
        feat = self.conv_out(feat)

        if not size is None:
            feat = F.interpolate(feat, size=size,mode='bilinear', align_corners=True)
        return feat


class conv(nn.Module):
    def __init__(self,in_chan, out_chan):
        super(conv, self).__init__()
        self.conv2d = nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=1, padding=0, bias=False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self,x):
        feat = self.conv2d(x)
        feat = self.up(feat)
        return(feat)



class MAFANet(nn.Module):
    def __init__(self, n_classes):
        super(MAFANet, self).__init__()
        self.backbone = Backbone_resnet(backbone='resnet18')

        filters = [64, 64, 128, 256, 512]

        self.mcsa5 = MCSA(filters[4], filters[4], kernel_size=(7,7))
        self.conv1 = conv(64,64)
        self.conv2 = conv(128,64)
        self.conv3 = conv(256,128)
        self.conv4 = conv(512,256)

        self.spb = SPB()

        self.conv22 = nn.Conv2d(512, 256, 1, 1, 0)

        self.bff1 = BFF1()
        self.bff2 = BFF(128, 128)
        self.bff3 = BFF(64, 64)
        self.bff4 = BFF(64, 64)


        self.DMFA1 = DMFA(64)
        self.DMFA2 = DMFA(64)
        self.DMFA3 = DMFA(128)
        self.DMFA4 = DMFA(256)

        self.head = BRB(64, 64, n_classes)


    def forward(self, x):
        [x1, x2, x3, x4, x5] = self.backbone(x)
        x2c = self.conv1(x2)
        x3c = self.conv2(x3)
        x4c = self.conv3(x4)
        x5c = self.conv4(x5)


        x_1r = self.DMFA1(x2c, x1)
        x_2r = self.DMFA2(x3c, x2)
        x_3r = self.DMFA3(x4c, x3)
        x_4r = self.DMFA4(x5c, x4)
        x_5s = self.spb(x5)
        x_6s = self.mcsa5(x_5s)
        x_7s = self.conv22(x_6s)




        x8 = self.bff1(x_4r,x_7s)
        x9 = self.bff2(x_3r, x8)
        x10 = self.bff3(x_2r,x9)
        x11 = self.bff4(x_1r,x10)



        out = self.head(x11)
        return out





if __name__ == '__main__':
    model =MAFANet(3)
    x = torch.rand(1, 3, 224, 224)
    out = model(x)
    print('out:', out.size())
    # print(model)
    model = model.cuda()
    batch = torch.cuda.FloatTensor(1, 3, 224, 224)
    model_eval = add_flops_counting_methods(model)
    model_eval.eval().start_flops_count()
    out = model_eval(batch)  # ,only_encode=True)
    print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
    print('Params: ' + get_model_parameters_number(model))
    print('Output shape: {}'.format(list(out.shape)))
    total_paramters = sum(p.numel() for p in model.parameters())
    print('Total paramters: {}'.format(total_paramters))












