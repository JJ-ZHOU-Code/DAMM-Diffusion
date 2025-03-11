import torch.nn.functional as F
import torch.nn as nn 
import torch 

from monai.networks.blocks import TransformerBlock
from monai.networks.layers.utils import get_norm_layer, get_dropout_layer
from monai.networks.layers.factories import Conv
from einops import rearrange

import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from monai.networks.layers.utils import get_act_layer

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        self.gate_c = nn.Sequential()
        self.gate_c.add_module( 'flatten', Flatten() )
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
        self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
    def forward(self, in_tensor):
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)

class SpatialGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module( 'gate_s_conv_reduce0', nn.Conv2d(gate_channel, gate_channel//reduction_ratio, kernel_size=1))
        self.gate_s.add_module( 'gate_s_bn_reduce0',	nn.BatchNorm2d(gate_channel//reduction_ratio) )
        self.gate_s.add_module( 'gate_s_relu_reduce0',nn.ReLU() )
        for i in range( dilation_conv_num ):
            self.gate_s.add_module( 'gate_s_conv_di_%d'%i, nn.Conv2d(gate_channel//reduction_ratio, gate_channel//reduction_ratio, kernel_size=3, \
						padding=dilation_val, dilation=dilation_val) )
            self.gate_s.add_module( 'gate_s_bn_di_%d'%i, nn.BatchNorm2d(gate_channel//reduction_ratio) )
            self.gate_s.add_module( 'gate_s_relu_di_%d'%i, nn.ReLU() )
        self.gate_s.add_module( 'gate_s_conv_final', nn.Conv2d(gate_channel//reduction_ratio, 1, kernel_size=1) )
    def forward(self, in_tensor):
        return self.gate_s( in_tensor ).expand_as(in_tensor)
class BAM(nn.Module):
    def __init__(self, gate_channel):
        super(BAM, self).__init__()
        self.channel_att = ChannelGate(gate_channel)
        self.spatial_att = SpatialGate(gate_channel)
    def forward(self,in_tensor):
        att = 1 + F.sigmoid( self.channel_att(in_tensor) * self.spatial_att(in_tensor) )
        return att * in_tensor


class SE_Layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE_Layer, self).__init__()
        self.channel = channel 
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1) #对应Excitation操作
        return x * y.expand_as(x)

class MaskLayer(nn.Module):
    def __init__(self, channel):
        super(MaskLayer, self).__init__()
        mid_channel = 2*channel
        self.mask_block = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=mid_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels=channel*2, out_channels=channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        mask = self.mask_block(x)
        out = mask * x
        return out    

class SCAF(nn.Module):
    # spatial-channel attention fusion
    def __init__(self, channels=64):
        super(SCAF, self).__init__()

        self.channel_att = SE_Layer(channels)

        self.in_channels = channels

        self.norm = Normalize(channels)
        self.q1 = torch.nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)
        self.k1 = torch.nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)
        self.q2 = torch.nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)
        self.k2 = torch.nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)

        self.v = torch.nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)

        self.proj_out = torch.nn.Conv2d(channels,channels,kernel_size=1,stride=1,padding=0)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        x = x1 + x2 
        
        ch_w = self.sigmoid(self.channel_att(x))

        x = self.norm(x)
        v = self.v(x)

        h1 = self.norm(x1)
        q1 = self.q1(h1)
        k1 = self.k1(h1)
        b,c,h,w = q1.shape
        q1 = rearrange(q1, 'b c h w -> b (h w) c')
        k1 = rearrange(k1, 'b c h w -> b c (h w)')
        w1 = torch.einsum('bij,bjk->bik', q1, k1)
        w1 = w1 * (int(c)**(-0.5))
        w1 = torch.nn.functional.softmax(w1, dim=2)

        h2 = self.norm(x2)
        q2 = self.q1(h2)
        k2 = self.k1(h2)
        b,c,h,w = q2.shape
        q2 = rearrange(q2, 'b c h w -> b (h w) c')
        k2 = rearrange(k2, 'b c h w -> b c (h w)')
        w2 = torch.einsum('bij,bjk->bik', q2, k2)
        w2 = w2 * (int(c)**(-0.5))
        w2 = torch.nn.functional.softmax(w2, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w1 = rearrange(w1, 'b i j -> b j i')
        h1 = torch.einsum('bij,bjk->bik', v, w1)
        h1 = rearrange(h1, 'b c (h w) -> b c h w', h=h)
        w2 = rearrange(w2, 'b i j -> b j i')
        h2 = torch.einsum('bij,bjk->bik', v, w2)
        h2 = rearrange(h2, 'b c (h w) -> b c h w', h=h)

        xo = 2 * h1 * ch_w + 2 * h2 * (1-ch_w)
        # xo = torch.cat([x1+h1*ch_w , x2+h2*(1-ch_w)], dim=1)

        return xo



    def forward_v1(self, x1, x2):
        x = x1 + x2 
        
        ch_w = self.sigmoid(self.channel_att(x))

        x = self.norm(x)
        v = self.v(x)

        h1 = self.norm(x1)
        q1 = self.q1(h1)
        k1 = self.k1(h1)
        b,c,h,w = q1.shape
        q1 = rearrange(q1, 'b c h w -> b (h w) c')
        k1 = rearrange(k1, 'b c h w -> b c (h w)')
        w1 = torch.einsum('bij,bjk->bik', q1, k1)
        # w1 = w1 * (int(c)**(-0.5))
        # w1 = torch.nn.functional.softmax(w1, dim=2)

        h2 = self.norm(x2)
        q2 = self.q1(h2)
        k2 = self.k1(h2)
        b,c,h,w = q2.shape
        q2 = rearrange(q2, 'b c h w -> b (h w) c')
        k2 = rearrange(k2, 'b c h w -> b c (h w)')
        w2 = torch.einsum('bij,bjk->bik', q2, k2)
        # w2 = w2 * (int(c)**(-0.5))
        # w2 = torch.nn.functional.softmax(w2, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w1 = rearrange(w1, 'b i j -> b j i')
        h1 = torch.einsum('bij,bjk->bik', v, w1)
        h1 = rearrange(h1, 'b c (h w) -> b c h w', h=h)
        w2 = rearrange(w2, 'b i j -> b j i')
        h2 = torch.einsum('bij,bjk->bik', v, w2)
        h2 = rearrange(h2, 'b c (h w) -> b c h w', h=h)

        # xo = h1 * ch_w + h2 * (1-ch_w)
        xo = torch.cat([h1 * ch_w , h2 * (1-ch_w)], dim=1)

        return xo


class Att_Fusion(nn.Module):

    def __init__(self, channels=64, r=16):
        super(Att_Fusion, self).__init__()
        inter_channels = int(channels // r)
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # self.spatial_att = SpatialSelfAttention(channels)
        # self.channel_att = SELayer(channels)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        x_sp = self.spatial_att(xa)
        x_ch = self.channel_att(xa)
        
        wei = self.sigmoid(x_sp * x_ch)

        # xo = 2 * x * wei + 2 * residual * (1 - wei)
        xo = torch.cat([2 * x * wei, 2 * residual * (1 - wei)], dim=1)
        return xo