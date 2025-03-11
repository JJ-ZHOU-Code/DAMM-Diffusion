
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from monai.networks.blocks import UnetOutBlock

from diffusion.models.utils.conv_blocks import BasicBlock, UpBlock, DownBlock, UnetBasicBlock, UnetResBlock, save_add, BasicDown, BasicUp, SequentialEmb
from diffusion.models.embedders import TimeEmbbeding
from diffusion.models.utils.attention_blocks import Attention, zero_module
from diffusion.models.utils.attention import SpatialSelfAttention_Uncertain
from einops import rearrange, repeat

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

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
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


class Conv1_MM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1_MM, self).__init__()
        self.channel_attn = SELayer(in_channels)
        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1)),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.conv1_block(x)
        return x    

# spatial =-> channel attention --> 1x1 conv
class UNet(nn.Module):

    def __init__(self, 
            in_ch=1, 
            out_ch=1, 
            spatial_dims = 3,
            hid_chs =    [256, 256, 512,  1024],
            kernel_sizes=[ 3,  3,   3,   3],
            strides =    [ 1,  2,   2,   2], # WARNING, last stride is ignored (follows OpenAI)
            act_name=("SWISH", {}),
            norm_name = ("GROUP", {'num_groups':32, "affine": True}),
            time_embedder=TimeEmbbeding,
            time_embedder_kwargs={},
            cond_embedder=None,
            cond_embedder_kwargs={},
            deep_supervision=True, # True = all but last layer, 0/False=disable, 1=only first layer, ... 
            use_res_block=True,
            estimate_variance=False ,
            use_self_conditioning = False, 
            use_img_conditioning = False,
            img_condition_num = 0,
            dropout=0.0, 
            learnable_interpolation=True,
            use_attention='none',
            num_res_blocks=2,
        ):
        super().__init__()
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.use_self_conditioning = use_self_conditioning
        self.use_img_conditioning = use_img_conditioning
        self.use_res_block = use_res_block
        self.depth = len(strides)
        self.num_res_blocks = num_res_blocks

        # ------------- Time-Embedder-----------
        if time_embedder is not None:
            self.time_embedder=time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None 
            time_emb_dim = None 

        # ------------- Condition-Embedder-----------

        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock

        # ----------- In-Convolution ------------
        self.in_conv = BasicBlock(spatial_dims, in_ch, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0])
        
        
        # ----------- Encoder ------------
        in_blocks = [] 
        mask_blocks = [MaskLayer(channel=hid_chs[0])]
        for i in range(1, self.depth):
            for k in range(num_res_blocks):
                seq_list = [] 
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i-1 if k==0 else i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )
                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        num_heads=8,
                        ch_per_head=hid_chs[i]//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )
                in_blocks.append(SequentialEmb(*seq_list))
                mask_blocks.append(
                        MaskLayer(channel=hid_chs[i])
                    )

            if i < self.depth-1:
                in_blocks.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation 
                    )
                )
                mask_blocks.append(
                        MaskLayer(channel=hid_chs[i])
                    )
 

        self.in_blocks = nn.ModuleList(in_blocks)
        self.mask_blocks = nn.ModuleList(mask_blocks)
        
        # ----------- Middle ------------
        self.middle_block = SequentialEmb(
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            ),
            Attention(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                num_heads=8,
                ch_per_head=hid_chs[-1]//8,
                depth=1,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=time_emb_dim,
                attention_type=use_attention[-1]
            ),
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            )
        )

 
     
        # ------------ Decoder ----------
        out_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks+1):
                seq_list = [] 
                out_channels=hid_chs[i-1 if k==0 else i]
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i]+hid_chs[i-1 if k==0 else i],
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )
            
                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        num_heads=8,
                        ch_per_head=out_channels//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )

                if (i >1) and k==0:
                    seq_list.append(
                        BasicUp(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=strides[i],
                            stride=strides[i],
                            learnable_interpolation=learnable_interpolation 
                        )
                    )
        
                out_blocks.append(SequentialEmb(*seq_list))
        self.out_blocks = nn.ModuleList(out_blocks)
        
        
        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch*2 if estimate_variance else out_ch
        self.outc = zero_module(UnetOutBlock(spatial_dims, hid_chs[0], out_ch_hor, dropout=None))
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-2 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            zero_module(UnetOutBlock(spatial_dims, hid_chs[i]+hid_chs[i-1], out_ch, dropout=None) )
            for i in range(2, deep_supervision+2)
        ])
 

    def forward(self, x_t, t=None, img_cond=None):
        # x_t [B, C, *]
        # t [B,]
        # condition [B,]
        # self_cond [B, C, *]
        

        # -------- Time Embedding (Gloabl) -----------
        time_emb = self.time_embedder(t) # [B, C]

        emb = time_emb
       
        # ---------- Self-conditioning-----------
        if self.use_img_conditioning:
            x_t = torch.cat([x_t] + img_cond, dim=1) 
    
        # --------- Encoder --------------
        x = [self.in_conv(x_t)]
        for i in range(len(self.in_blocks)):
            x.append(self.in_blocks[i](x[i], emb))

        # ---------- Middle --------------
        h = self.middle_block(x[-1], emb)
        
        # -------- Decoder -----------
        y_ver = []
        for i in range(len(self.out_blocks), 0, -1):
            h = torch.cat([h, x.pop()], dim=1)

            depth, j = i//(self.num_res_blocks+1), i%(self.num_res_blocks+1)-1
            y_ver.append(self.outc_ver[depth-1](h)) if (len(self.outc_ver)>=depth>0) and (j==0) else None 

            h = self.out_blocks[i-1](h, emb)

        # ---------Out-Convolution ------------
        y = self.outc(h)

        return y, y_ver[::-1]



class UNet_DAMM(nn.Module):

    def __init__(self, 
            in_ch=2, 
            out_ch=1, 
            spatial_dims = 3,
            hid_chs =    [256, 256, 512,  1024],
            kernel_sizes=[ 3,  3,   3,   3],
            strides =    [ 1,  2,   2,   2], # WARNING, last stride is ignored (follows OpenAI)
            act_name=("SWISH", {}),
            norm_name = ("GROUP", {'num_groups':32, "affine": True}),
            time_embedder=TimeEmbbeding,
            time_embedder_kwargs={},
            cond_embedder=None,
            cond_embedder_kwargs={},
            deep_supervision=True, # True = all but last layer, 0/False=disable, 1=only first layer, ... 
            use_res_block=True,
            estimate_variance=False ,
            use_self_conditioning = False, 
            use_img_conditioning = False,
            img_condition_num = 0,
            dropout=0.0, 
            learnable_interpolation=True,
            use_attention='none',
            num_res_blocks=2,
        ):
        super().__init__() 
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention]*len(strides) 
        self.use_self_conditioning = use_self_conditioning
        self.use_img_conditioning = use_img_conditioning
        self.use_res_block = use_res_block
        self.depth = len(strides)
        self.num_res_blocks = num_res_blocks

        # ------------- Time-Embedder-----------
        if time_embedder is not None:
            self.time_embedder=time_embedder(**time_embedder_kwargs)
            time_emb_dim = self.time_embedder.emb_dim
        else:
            self.time_embedder = None 
            time_emb_dim = None 

        # ------------- Condition-Embedder-----------

        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock

        # ----------- In-Convolution ------------
        in_ch = in_ch * 2
        self.in_conv = BasicBlock(spatial_dims, in_ch, hid_chs[0], kernel_size=kernel_sizes[0], stride=strides[0])
        
        # ----------- Encoder ------------
        in_blocks = [] 
        mask_blocks = [MaskLayer(channel=hid_chs[0])]
        conv1_blocks = [Conv1_MM(in_channels=hid_chs[0]*2, out_channels=hid_chs[0])]
        for i in range(1, self.depth):
            for k in range(num_res_blocks):
                seq_list = [] 
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i-1 if k==0 else i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )
                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        num_heads=8,
                        ch_per_head=hid_chs[i]//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )
                in_blocks.append(SequentialEmb(*seq_list))
                mask_blocks.append(MaskLayer(channel=hid_chs[i]))
                conv1_blocks.append(Conv1_MM(in_channels=hid_chs[i]*2, out_channels=hid_chs[i]))

            if i < self.depth-1:
                in_blocks.append(
                    BasicDown(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i],
                        out_channels=hid_chs[i],
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        learnable_interpolation=learnable_interpolation 
                    )
                )
                mask_blocks.append(MaskLayer(channel=hid_chs[i]))
                conv1_blocks.append(Conv1_MM(in_channels=hid_chs[i]*2, out_channels=hid_chs[i]))
 

        self.in_blocks = nn.ModuleList(in_blocks)
        self.mask_blocks = nn.ModuleList(mask_blocks)
        self.conv1_blocks = nn.ModuleList(conv1_blocks)
        
        # ----------- Middle ------------
        self.un_crs_attn = SpatialSelfAttention_Uncertain(in_channels=hid_chs[-1])
        
        self.middle_block = SequentialEmb(
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            ),
            Attention(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                num_heads=8,
                ch_per_head=hid_chs[-1]//8,
                depth=1,
                norm_name=norm_name,
                dropout=dropout,
                emb_dim=time_emb_dim,
                attention_type=use_attention[-1]
            ),
            ConvBlock(
                spatial_dims=spatial_dims,
                in_channels=hid_chs[-1],
                out_channels=hid_chs[-1],
                kernel_size=kernel_sizes[-1],
                stride=1,
                norm_name=norm_name,
                act_name=act_name,
                dropout=dropout,
                emb_channels=time_emb_dim
            )
        )

 
        # ------------ Decoder ----------
        out_blocks = [] 
        for i in range(1, self.depth):
            for k in range(num_res_blocks+1):
                seq_list = [] 
                out_channels=hid_chs[i-1 if k==0 else i]
                seq_list.append(
                    ConvBlock(
                        spatial_dims=spatial_dims,
                        in_channels=hid_chs[i]+hid_chs[i-1 if k==0 else i],
                        out_channels=out_channels,
                        kernel_size=kernel_sizes[i],
                        stride=1,
                        norm_name=norm_name,
                        act_name=act_name,
                        dropout=dropout,
                        emb_channels=time_emb_dim
                    )
                )
            
                seq_list.append(
                    Attention(
                        spatial_dims=spatial_dims,
                        in_channels=out_channels,
                        out_channels=out_channels,
                        num_heads=8,
                        ch_per_head=out_channels//8,
                        depth=1,
                        norm_name=norm_name,
                        dropout=dropout,
                        emb_dim=time_emb_dim,
                        attention_type=use_attention[i]
                    )
                )

                if (i >1) and k==0:
                    seq_list.append(
                        BasicUp(
                            spatial_dims=spatial_dims,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=strides[i],
                            stride=strides[i],
                            learnable_interpolation=learnable_interpolation 
                        )
                    )
        
                out_blocks.append(SequentialEmb(*seq_list))
        self.out_blocks = nn.ModuleList(out_blocks)
        
        
        # --------------- Out-Convolution ----------------
        out_ch_hor = out_ch*2 if estimate_variance else out_ch
        self.outc = zero_module(UnetOutBlock(spatial_dims, hid_chs[0], out_ch_hor, dropout=None))
        if isinstance(deep_supervision, bool):
            deep_supervision = self.depth-2 if deep_supervision else 0 
        self.outc_ver = nn.ModuleList([
            zero_module(UnetOutBlock(spatial_dims, hid_chs[i]+hid_chs[i-1], out_ch, dropout=None) )
            for i in range(2, deep_supervision+2)
        ])
 

        self.unet_S = UNet(
            in_ch=in_ch, 
            out_ch=out_ch, 
            spatial_dims = spatial_dims,
            hid_chs =hid_chs,
            kernel_sizes=kernel_sizes,
            strides =strides, # WARNING, last stride is ignored (follows OpenAI)
            act_name=act_name,
            norm_name =norm_name,
            time_embedder=time_embedder,
            time_embedder_kwargs=time_embedder_kwargs,
            cond_embedder=cond_embedder,
            cond_embedder_kwargs=cond_embedder_kwargs,
            deep_supervision=deep_supervision, # True = all but last layer, 0/False=disable, 1=only first layer, ... 
            use_res_block=use_res_block,
            estimate_variance=estimate_variance ,
            use_self_conditioning = use_self_conditioning, 
            use_img_conditioning = use_img_conditioning,
            img_condition_num = img_condition_num,
            dropout=dropout, 
            learnable_interpolation=learnable_interpolation,
            use_attention=use_attention,
            num_res_blocks=num_res_blocks,)


    def forward(self, x_t, t=None, img_cond=None, use_uncertain=True):
        # x_t [B, C, *]
        # t [B,]
        # img_cond [B, C, *]
        
        # -------- Time Embedding (Gloabl) -----------
        time_emb = self.time_embedder(t) # [B, C]

        emb = time_emb
       
        # ---------- Self-conditioning-----------
        x_CD31 = torch.cat([x_t] + [img_cond[0]], dim=1)
        x_DAPI = torch.cat([x_t] + [img_cond[1]], dim=1)
        x_CD31_in = self.unet_S.in_conv(x_CD31)
        x_DAPI_in = self.in_conv(x_DAPI)
        x_s_list = [x_CD31_in]

        x_s_middle = self.unet_S.mask_blocks[0](x_CD31_in)
        x_d = self.mask_blocks[0](x_DAPI_in)
        x_mm = torch.cat([x_s_middle, x_d], dim=1)
        x_mm = self.conv1_blocks[0](x_mm)
        x_mm_list = [x_mm]
    
        # --------- Encoder --------------
        for i in range(len(self.unet_S.in_blocks)):
            x_s = self.unet_S.in_blocks[i](x_s_list[i], emb)
            x_mm = self.in_blocks[i](x_mm_list[i], emb)
            x_s_list.append(x_s)

            x_s_middle = self.unet_S.mask_blocks[i+1](x_s)
            x_mm = self.mask_blocks[i+1](x_mm)
            x_mm = torch.cat([x_s_middle, x_mm], dim=1)
            x_mm = self.conv1_blocks[i+1](x_mm)
            x_mm_list.append(x_mm)


        # ---------- Middle --------------
        h_s = self.unet_S.middle_block(x_s_list[-1], emb)

        x_mm, w_i, u_map = self.un_crs_attn(x_s, x_mm, use_uncertain)
        h_mm = self.middle_block(x_mm, emb)
        
        # -------- Decoder -----------
        for i in range(len(self.unet_S.out_blocks), 0, -1):
            h_s = torch.cat([h_s, x_s_list.pop()], dim=1)
            h_s = self.unet_S.out_blocks[i-1](h_s, emb)

            h_mm = torch.cat([h_mm, x_mm_list.pop()], dim=1)
            h_mm = self.out_blocks[i-1](h_mm, emb)

        # ---------Out-Convolution ------------
        y_s = self.unet_S.outc(h_s)
        y_mm = self.outc(h_mm)

        return y_s, y_mm, None, u_map