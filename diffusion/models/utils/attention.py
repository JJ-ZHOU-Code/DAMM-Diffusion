import torch 
import torch.nn.functional as F
from torch import nn, einsum

from monai.networks.blocks import TransformerBlock
from monai.networks.layers.utils import get_norm_layer, get_dropout_layer
from monai.networks.layers.factories import Conv
from einops import rearrange, repeat

from inspect import isfunction

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SpatialSelfAttention_Uncertain(nn.Module):
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

        self.u = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 1),
            torch.nn.Sigmoid()
        )
        
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x, context=None, uncertain=True):
        h_ = x
        h_ = self.norm(h_)

        q = self.q(h_)
        context = default(context, x)
        context = self.norm(context)

        k = self.k(context)
        v = self.v(context)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        if uncertain:
            u = rearrange(h_, 'b c h w -> b (h w) c')
            u_map = self.u(u) # b,hw,1
            w_ = w_ * (int(c)**(-0.5)) * u_map.transpose(-1, -2) # compute attention weight with uncertain(column)
        else: 
            w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        if uncertain:
            # u_score = torch.mean(u_map.to(torch.float32), dim=(-2, -1)) # uncertainty score: (b,)
            return x+h_, u_map
        return x+h_, None
