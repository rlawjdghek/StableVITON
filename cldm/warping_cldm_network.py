import os
from os.path import join as opj
from typing import Any, Optional
import omegaconf
from glob import glob

import cv2
import einops
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch as th
import torch.nn as nn
from cleanfid import fid
from pytorch_lightning.utilities.distributed import rank_zero_only
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from eval_models import PerceptualLoss

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    normalization
)

from einops import rearrange
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel, TimestepEmbedSequential, ResBlock, Downsample, AttentionBlock
from ldm.util import exists
from ldm.modules.attention import Normalize, CrossAttention, MemoryEfficientCrossAttention, XFORMERS_IS_AVAILBLE, FeedForward
class CustomBasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False,use_loss=True):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint
        self.use_loss = use_loss

    def forward(
            self, 
            x, 
            context=None, 
            mask=None, 
            mask1=None, 
            mask2=None, 
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
        ):
        if not (use_attention_tv_loss or use_attention_mask):
            x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None, mask=mask) + x
            x = self.attn2(self.norm2(x), context=context, mask=mask) + x
            x = self.ff(self.norm3(x)) + x
            return x
        elif use_attention_mask:
            x1 = self.attn1(
                self.norm1(x), 
                context=context if self.disable_self_attn else None, 
                mask=mask, 
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=False,
            )
            x = x1 + x
            x2 = self.attn2(  # cross attention
                self.norm2(x), 
                context=context,
                mask=mask,
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=False,
            )
            x = x2 + x
            x = self.ff(self.norm3(x)) + x
            return x
        else:
            x1, loss1 = self.attn1(
                self.norm1(x), 
                context=context if self.disable_self_attn else None, 
                mask=mask, 
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=use_attention_tv_loss,
                tv_loss_type=tv_loss_type,
            )
            x = x1 + x
            x2, loss2 = self.attn2(
                self.norm2(x), 
                context=context,
                mask=mask,
                mask1=mask1, 
                mask2=mask2, 
                use_attention_tv_loss=use_attention_tv_loss,
                use_loss=self.use_loss,
                tv_loss_type=tv_loss_type,
            )
            x = x2 + x
            x = self.ff(self.norm3(x)) + x
            loss = loss1 + loss2
            return x, loss
        
class CustomSpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    NEW: use_linear for more efficiency instead of the 1x1 convs
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None,
                 disable_self_attn=False, use_linear=False,
                 use_checkpoint=True,use_loss=True):
        super().__init__()
        if exists(context_dim) and not isinstance(context_dim, list):
            context_dim = [context_dim]
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(in_channels,
                                     inner_dim,
                                     kernel_size=1,
                                     stride=1,
                                     padding=0)
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                CustomBasicTransformerBlock(
                inner_dim, 
                n_heads, 
                d_head, 
                dropout=dropout, 
                context_dim=context_dim[d],
                disable_self_attn=disable_self_attn, 
                checkpoint=use_checkpoint, use_loss=use_loss) for d in range(depth)
            ]
        )
        if not use_linear:
            self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                                  in_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0))
        else:
            self.proj_out = zero_module(nn.Linear(in_channels, inner_dim))
        self.use_linear = use_linear
        self.use_loss = use_loss
    def forward(
            self, 
            x, 
            context=None, 
            mask=None, 
            mask1=None, 
            mask2=None, 
            use_attention_mask=False,
            use_attention_tv_loss=False,
            tv_loss_type=None,
    ):
        # note: if no context is given, cross-attention defaults to self-attention
        loss = 0
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)
        for i, block in enumerate(self.transformer_blocks):
            if not (use_attention_tv_loss or use_attention_mask):
                x = block(x, context=context[i], mask=mask)
            elif use_attention_mask:
                x = block(
                    x,
                    context=context[i],
                    mask=mask, 
                    mask1=mask1, 
                    mask2=mask2, 
                    use_attention_mask=True,
                    use_attention_tv_loss=False,
                    use_center_loss=False,
                )
            else:
                x, attn_loss = block(
                    x,
                    context=context[i],
                    mask=mask, 
                    mask1=mask1, 
                    mask2=mask2, 
                    use_attention_mask=use_attention_mask,
                    use_attention_tv_loss=use_attention_tv_loss,
                    tv_loss_type=tv_loss_type,
                )
                loss += attn_loss
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        if not (use_attention_tv_loss):
            return x + x_in
        else:
            return x + x_in, loss
class StableVITON(UNetModel):
    def __init__(
        self,
        dim_head_denorm=1,
        use_atv_loss=False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        warp_flow_blks = []
        warp_zero_convs = []

        self.encode_output_chs = [
            320,
            320,
            640,
            640,
            640,
            1280, 
            1280, 
            1280, 
            1280
        ]

        self.encode_output_chs2 = [
            320,
            320,
            320,
            320,
            640, 
            640, 
            640,
            1280, 
            1280
        ]

        
        for idx, (in_ch, cont_ch) in enumerate(zip(self.encode_output_chs, self.encode_output_chs2)):
            dim_head = in_ch // self.num_heads
            dim_head = dim_head // dim_head_denorm
            warp_flow_blks.append(CustomSpatialTransformer(
                in_channels=in_ch,
                n_heads=self.num_heads,
                d_head=dim_head,
                depth=self.transformer_depth,
                context_dim=cont_ch,
                use_linear=self.use_linear_in_transformer,
                use_checkpoint=self.use_checkpoint,
                use_loss=idx%3 == 1,
            ))
            warp_zero_convs.append(self.make_zero_conv(in_ch))
        self.warp_flow_blks = nn.ModuleList(reversed(warp_flow_blks))
        self.warp_zero_convs = nn.ModuleList(reversed(warp_zero_convs))
        self.use_atv_loss = use_atv_loss
    def make_zero_conv(self, channels):
        return zero_module(conv_nd(2, channels, channels, 1, padding=0))
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        mask1 = kwargs.get("mask1", None)
        mask2 = kwargs.get("mask2", None)
        loss = 0
        with torch.no_grad():
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)
            h = x.type(self.dtype)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
            h = self.middle_block(h, emb, context)

        if control is not None:                 
            hint = control.pop()
        # resolution 8 is skipped
        for module in self.output_blocks[:3]:
            control.pop()
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        n_warp = len(self.encode_output_chs)
        for i, (module, warp_blk, warp_zc) in enumerate(zip(self.output_blocks[3:n_warp+3], self.warp_flow_blks, self.warp_zero_convs)):
            if control is None or (h.shape[-2] == 8 and h.shape[-1] == 6):
                assert 0, f"shape is wrong : {h.shape}"
            else:
                hint = control.pop()
                h, attn_loss = self.warp(h, hint, warp_blk, warp_zc, mask1=mask1, mask2=mask2)
                loss += attn_loss
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        for module in self.output_blocks[n_warp+3:]:
            if control is None:
                h = torch.cat([h, hs.pop()], dim=1)                                          
            else:
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
        h = h.type(x.dtype)
        if self.use_atv_loss:
            return self.out(h), loss
        else:
            return self.out(h)
    def warp(self, x, hint, crossattn_layer, zero_conv, mask1=None, mask2=None):
        hint = rearrange(hint, "b c h w -> b (h w) c").contiguous()
        if self.use_atv_loss:
            output, attn_loss = crossattn_layer(x, hint, mask1=mask1, mask2=mask2, use_attention_tv_loss=True)
            output = zero_conv(output)
            return output + x, attn_loss
        else:
            output = crossattn_layer(x, hint)
            output = zero_conv(output)
            return output + x, 0

class NoZeroConvControlNet(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  
            legacy=True,
            disable_self_attentions=None,
            num_attention_blocks=None,
            disable_middle_self_attn=False,
            use_linear_in_transformer=False,
            use_VAEdownsample=False,
            cond_first_ch=8,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.dims = dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received um_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.use_VAEdownsample = use_VAEdownsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )

        self.cond_first_block = TimestepEmbedSequential(
            zero_module(conv_nd(dims, cond_first_ch, model_channels, 3, padding=1))
        )


        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

    def forward(self, x, hint, timesteps, context, only_mid_control=False, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if not self.use_VAEdownsample:
            guided_hint = self.input_hint_block(hint, emb, context)
        else:
            guided_hint = self.cond_first_block(hint, emb, context)

        outs = []
        hs = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            if guided_hint is not None:
                h = module(h, emb, context)
                h += guided_hint
                hs.append(h)
                guided_hint = None
            else:                                                
                h = module(h, emb, context)
                hs.append(h)
            outs.append(h)

        h = self.middle_block(h, emb, context)
        outs.append(h)
        return outs, None