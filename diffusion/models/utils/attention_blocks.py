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

class GEGLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels) 
        self.proj = nn.Linear(in_channels, out_channels*2, bias=True)

    def forward(self, x):
        # x expected to be [B, C, *] 
        # Workaround as layer norm can't currently be applied on arbitrary dimension: https://github.com/pytorch/pytorch/issues/71465
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1).transpose(1, 2) # -> [B, C, N] -> [B, N, C]
        x = self.norm(x)
        x, gate = self.proj(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        return x.transpose(1, 2).reshape(b, -1, *spatial) # -> [B, C, N] -> [B, C, *]

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def compute_attention(q,k,v , num_heads, scale):
    q, k, v = map(lambda t: rearrange(t, 'b (h d) n -> (b h) d n', h=num_heads), (q, k, v)) # [(BxHeads), Dim_per_head, N]

    attn = (torch.einsum('b d i, b d j -> b i j', q*scale, k*scale)).softmax(dim=-1) # Matrix product = [(BxHeads), Dim_per_head, N] * [(BxHeads), Dim_per_head, N'] =[(BxHeads), N, N']

    out = torch.einsum('b i j, b d j-> b d i', attn, v)  # Matrix product: [(BxHeads), N, N'] * [(BxHeads), Dim_per_head, N'] = [(BxHeads), Dim_per_head, N] 
    out = rearrange(out, '(b h) d n-> b (h d) n', h=num_heads) # ->  [B, (Heads x Dim_per_head), N] 

    return out 


class LinearTransformerNd(nn.Module):
    """ Combines multi-head self-attention and multi-head cross-attention.

        Multi-Head Self-Attention: 
        Similar to  multi-head self-attention (https://arxiv.org/abs/1706.03762) without Norm+MLP (compare Monai TransformerBlock)
        Proposed here: https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
        Similar to: https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/diffusionmodules/openaimodel.py#L278
        Similar to: https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/attention.py#L80
        Similar to: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/dfbafee555bdae80b55d63a989073836bbfc257e/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L209 
        Similar to: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/diffusionmodules/model.py#L150 

        CrossAttention:
            Proposed here: https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/attention.py#L152
        
    """
    def __init__(
        self,
        spatial_dims, 
        in_channels, 
        out_channels, # WARNING: if out_channels != in_channels, skip connection is disabled 
        num_heads=8, 
        ch_per_head=32, # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs)
        norm_name=("GROUP", {'num_groups':32, "affine": True}), # Or use LayerNorm but be aware of https://github.com/pytorch/pytorch/issues/71465 (=> GroupNorm with num_groups=1) 
        dropout=None,
        emb_dim=None,
    ):
        super().__init__()
        hid_channels = num_heads*ch_per_head
        self.num_heads = num_heads
        self.scale = ch_per_head**-0.25 # Should be 1/sqrt("queries and keys of dimension"), Note: additional sqrt needed as it follows OpenAI:  (q * scale) * (k * scale) instead of (q *k) * scale 
        
        self.norm_x = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=in_channels)
        emb_dim = in_channels if emb_dim is None else emb_dim

        Convolution = Conv["conv", spatial_dims]
        self.to_q = Convolution(in_channels, hid_channels, 1) 
        self.to_k = Convolution(emb_dim, hid_channels, 1) 
        self.to_v = Convolution(emb_dim, hid_channels, 1)

        self.to_out = nn.Sequential(
            zero_module(Convolution(hid_channels, out_channels, 1)),
            nn.Identity() if dropout is None else get_dropout_layer(name=dropout, dropout_dim=spatial_dims)
        )

    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        # if no embedding is given, cross-attention defaults to self-attention 
        
        # Normalize  
        b, c, *spatial = x.shape
        x_n = self.norm_x(x)

        # Attention:  embedding (cross-attention) or x (self-attention)
        if embedding is None:
            embedding = x_n  # WARNING: This assumes that emb_dim==in_channels 
        else:
            if embedding.ndim == 2:
                embedding = embedding.reshape(*embedding.shape[:2], *[1]*(x.ndim-2)) # [B, C*] -> [B, C*, *] 
            # Why no normalization for embedding here?

        # Convolution 
        q = self.to_q(x_n)       # -> [B, (Heads x Dim_per_head), *] 
        k = self.to_k(embedding) # -> [B, (Heads x Dim_per_head), *] 
        v = self.to_v(embedding) # -> [B, (Heads x Dim_per_head), *] 

        # Flatten
        q = q.reshape(b, c, -1)                  # -> [B, (Heads x Dim_per_head), N] 
        k = k.reshape(*embedding.shape[:2], -1)  # -> [B, (Heads x Dim_per_head), N'] 
        v = v.reshape(*embedding.shape[:2], -1)  # -> [B, (Heads x Dim_per_head), N'] 

        # Apply attention
        out = compute_attention(q, k, v, self.num_heads, self.scale)
        
        out = out.reshape(*out.shape[:2], *spatial) # -> [B, (Heads x Dim_per_head), *]
        out = self.to_out(out)                      # -> [B, C', *]
        

        if x.shape == out.shape:
            out = x + out 
        return out # [B, C', *]


class LinearTransformer(nn.Module):
    """ See LinearTransformer, however this implementation is fixed to Conv1d/Linear"""
    def __init__(
        self,
        spatial_dims,
        in_channels, 
        out_channels, # WARNING: if out_channels != in_channels, skip connection is disabled 
        num_heads, 
        ch_per_head=32, # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs)
        norm_name=("GROUP", {'num_groups':32, "affine": True}),
        dropout=None,
        emb_dim=None
    ):
        super().__init__()
        hid_channels = num_heads*ch_per_head
        self.num_heads = num_heads
        self.scale = ch_per_head**-0.25 # Should be 1/sqrt("queries and keys of dimension"), Note: additional sqrt needed as it follows OpenAI:  (q * scale) * (k * scale) instead of (q *k) * scale 
        
        self.norm_x = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=in_channels)
        emb_dim = in_channels if emb_dim is None else emb_dim

        # Note: Conv1d and Linear are interchangeable but order of input changes [B, C, N] <-> [B, N, C]
        self.to_q = nn.Conv1d(in_channels, hid_channels, 1) 
        self.to_k = nn.Conv1d(emb_dim, hid_channels, 1) 
        self.to_v = nn.Conv1d(emb_dim, hid_channels, 1)
        # self.to_qkv = nn.Conv1d(emb_dim, hid_channels*3, 1)

        self.to_out = nn.Sequential(
            zero_module(nn.Conv1d(hid_channels, out_channels, 1)),
            nn.Identity() if dropout is None else get_dropout_layer(name=dropout, dropout_dim=spatial_dims)
        )

    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        # if no embedding is given, cross-attention defaults to self-attention 
        
        # Normalize  
        b, c, *spatial = x.shape
        x_n = self.norm_x(x)

        # Attention:  embedding (cross-attention) or x (self-attention)
        if embedding is None:
            embedding = x_n  # WARNING: This assumes that emb_dim==in_channels 
        else:
            if embedding.ndim == 2:
                embedding = embedding.reshape(*embedding.shape[:2], *[1]*(x.ndim-2)) # [B, C*] -> [B, C*, *] 
            # Why no normalization for embedding here?

        # Flatten 
        x_n = x_n.reshape(b, c, -1)                              # [B, C,  *] -> [B, C,  N] 
        embedding = embedding.reshape(*embedding.shape[:2], -1)  # [B, C*, *] -> [B, C*, N'] 

        # Convolution 
        q = self.to_q(x_n)       # -> [B, (Heads x Dim_per_head), N] 
        k = self.to_k(embedding) # -> [B, (Heads x Dim_per_head), N'] 
        v = self.to_v(embedding) # -> [B, (Heads x Dim_per_head), N'] 
        # qkv = self.to_qkv(x_n)
        # q,k,v = qkv.split(qkv.shape[1]//3, dim=1)

        # Apply attention
        out = compute_attention(q, k, v, self.num_heads, self.scale)
        
        out = self.to_out(out)                      # -> [B, C', N]
        out = out.reshape(*out.shape[:2], *spatial) # -> [B, C', *]

        if x.shape == out.shape:
            out = x + out 
        return out # [B, C', *]


class BasicTransformerBlock(nn.Module):
    def __init__(
        self, 
        spatial_dims,
        in_channels, 
        out_channels, # WARNING: if out_channels != in_channels, skip connection is disabled 
        num_heads, 
        ch_per_head=32,
        norm_name=("GROUP", {'num_groups':32, "affine": True}),
        dropout=None, 
        emb_dim=None
    ):
        super().__init__()
        self.self_atn = LinearTransformer(spatial_dims, in_channels, in_channels, num_heads, ch_per_head, norm_name, dropout, None)
        if emb_dim is not None:  
            self.cros_atn = LinearTransformer(spatial_dims, in_channels, in_channels, num_heads, ch_per_head, norm_name, dropout, emb_dim)
        self.proj_out = nn.Sequential(
            GEGLU(in_channels, in_channels*4),
            nn.Identity() if dropout is None else get_dropout_layer(name=dropout, dropout_dim=spatial_dims),
            Conv["conv", spatial_dims](in_channels*4, out_channels, 1, bias=True)
        )
        
 
    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        x = self.self_atn(x)
        if embedding is not None:
            x = self.cros_atn(x, embedding=embedding)
        out = self.proj_out(x)
        if out.shape[1] == x.shape[1]:
            return out + x
        return x

class SpatialTransformer(nn.Module):
    """ Proposed here: https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/attention.py#L218 
        Unrelated to: https://arxiv.org/abs/1506.02025  
    """
    def __init__(
        self, 
        spatial_dims,
        in_channels, 
        out_channels, # WARNING: if out_channels != in_channels, skip connection is disabled 
        num_heads, 
        ch_per_head=32, # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs)
        norm_name = ("GROUP", {'num_groups':32, "affine": True}),
        dropout=None, 
        emb_dim=None,
        depth=1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.norm = get_norm_layer(norm_name, spatial_dims=spatial_dims, channels=in_channels)
        conv_class = Conv["conv", spatial_dims]
        hid_channels = num_heads*ch_per_head

        self.proj_in = conv_class(
            in_channels,
            hid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(spatial_dims, hid_channels, hid_channels, num_heads, ch_per_head, norm_name, dropout=dropout, emb_dim=emb_dim) 
            for _ in range(depth)]
        )

        self.proj_out = conv_class( # Note: zero_module is used in original code  
            hid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x, embedding=None):
        # x expected to be [B, C, *]  and embedding is None or [B, C*] or [B, C*, *]
        # Note: if no embedding is given, cross-attention is disabled
        h = self.norm(x)
        h = self.proj_in(h) 

        for block in self.transformer_blocks:
            h = block(h, embedding=embedding)

        h = self.proj_out(h) # -> [B, C'', *] 
        if h.shape == x.shape:
            return h + x
        return h 


class Attention(nn.Module):
    def __init__(
        self, 
        spatial_dims,
        in_channels, 
        out_channels,  
        num_heads=8, 
        ch_per_head=32, # rule of thumb: 32 or 64 channels per head (see stable-diffusion / diffusion models beat GANs) 
        norm_name = ("GROUP", {'num_groups':32, "affine": True}),
        dropout=0, 
        emb_dim=None,
        depth=1,
        attention_type='linear'
    ) -> None:
        super().__init__()
        if attention_type == 'spatial':
            self.attention = SpatialTransformer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                ch_per_head=ch_per_head,
                depth=depth,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=emb_dim 
            )
        elif attention_type == 'linear':
            self.attention = LinearTransformer(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                num_heads=num_heads,
                ch_per_head=ch_per_head,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=emb_dim
            )
       
    
    def forward(self, x, emb=None):
        if hasattr(self, 'attention'):
            return self.attention(x, emb)
        else:
            return x 


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


class Spatial_CrossAttention(nn.Module):
    def __init__(self, in_channels, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = heads * dim_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_in_context = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        context_dim = context_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_channels),
            nn.Dropout(dropout)
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None, mask=None):
        if context is None:
            context = x
        b, c, h, w = x.shape
        x_in = x
        # img-like -> seq-like
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')

        context = self.norm(context)
        context = self.proj_in_context(context)
        context = rearrange(context, 'b c h w -> b (h w) c')

        # cross atten
        head = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=head), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)

        out = rearrange(out, '(b h) n d -> b n (h d)', h=head)
        out = self.to_out(out)
        
        # seq-like -> img-like
        x = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


class Atn_Fusion(nn.Module):
    def __init__(self, in_channel, out_channel, heads=8, dim_head=64):
        super().__init__()
        self.crs_attn = Spatial_CrossAttention(in_channels=in_channel, context_dim=in_channel, heads=heads, dim_head=dim_head)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channel, in_channel, 1, 1)
        self.fusion_conv = nn.Conv2d(in_channel*3, out_channel, 1, 1)

    def forward(self, x, context):

        # cross attn
        attention_feat = self.crs_attn(x, context)

        # depth wise attn
        depth_feat = F.conv2d(x.reshape(1,-1,x.shape[2],x.shape[3]), self.avg(context).reshape(1,-1,self.avg(context).shape[2],self.avg(context).shape[3]).permute(1,0,2,3), groups=x.shape[0]*x.shape[1])
        depth_feat = depth_feat.reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
        depth_feat = self.conv1(depth_feat)

        # concat
        # output = torch.cat([depth_feat, x], 1)
        output = torch.cat([attention_feat, depth_feat, x], 1)
        
        # fusion
        output = self.fusion_conv(output)

        return output 


class Relation_Attention(nn.Module):
    def __init__(self, in_channel, out_channel, heads=8, dim_head=64, emb_channels: int = None, act_name=None):
        super().__init__()
        self.mid_Atn = Atn_Fusion(in_channel, in_channel, heads=heads, dim_head=dim_head)
        # self.mid_Atn2 = Atn_Fusion(in_channel, in_channel, heads=heads, dim_head=dim_head)

        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 1)

    def forward(self, x, context):

        attn = self.mid_Atn(x, context)
        out = self.conv1(attn)

        return out

class DeepwiseAtn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 1)

    def forward(self, x, context):

        # depth wise attn
        depth_feat = F.conv2d(x.reshape(1,-1,x.shape[2],x.shape[3]), self.avg(context).reshape(1,-1,self.avg(context).shape[2],self.avg(context).shape[3]).permute(1,0,2,3), groups=x.shape[0]*x.shape[1])
        depth_feat = depth_feat.reshape(x.shape[0],x.shape[1],x.shape[2],x.shape[3])
        depth_feat = self.conv1(depth_feat)

        return depth_feat