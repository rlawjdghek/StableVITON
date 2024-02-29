from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from typing import Optional, Any
import os

from ldm.modules.diffusionmodules.util import checkpoint

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

# CrossAttn precision handling
import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max
def sim_minmax(x):
    assert x.ndim == 3, f"sim matrix shape : {x.shape} mush be [b HW hw]"
    return (x - x.min(dim=-1, keepdim=True)[0]) / (x.max(dim=-1, keepdim=True)[0] - x.min(dim=-1, keepdim=True)[0])

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

def get_tvloss(coords, mask, ch, cw):
    b, n, _ = coords.shape
    coords = coords.reshape(b,ch,cw,2)
    mask = mask.unsqueeze(-1)
    y_mask = mask[:,1:] * mask[:,:-1]
    x_mask = mask[:,:,1:] * mask[:,:,:-1]
    
    y_tvloss = torch.abs(coords[:,1:] - coords[:,:-1]) * y_mask
    x_tvloss = torch.abs(coords[:,:,1:] - coords[:,:,:-1]) * x_mask
    tv_loss = y_tvloss.sum() / y_mask.sum() + x_tvloss.sum() / x_mask.sum()
    return tv_loss

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


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
@torch.no_grad()
def attn_mask_resize(m,h,w):
    """
    m : [BS x 1 x mask_h x mask_w] => downsample, reshape and bool, [BS x h x w]
    """  
    m = F.interpolate(m, (h, w)).squeeze(1).contiguous()
    m = torch.where(m>=0.5, True, False)
    return m

class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0., **kwargs):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
        

    def forward(self, x, context=None, mask=None, mask1=None, mask2=None, use_attention_tv_loss=False):
        h = self.heads
        is_self_attn = context is None
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        key_token_length = k.shape[1]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k
        attn_mask = None
        if exists(mask1) or exists(mask2):  # [BS x 1 x H x W] float
            if mask1.ndim == 4 and mask2.ndim == 4:
                _, HW, hw = sim.shape
                bs = mask1.shape[0]
                dx = int((HW//12) ** 0.5)
                mH = int(4*dx)
                mW = int(3*dx)
                dx = int((hw//12) ** 0.5)
                mh = int(4*dx)
                mw = int(3*dx)
                if mH != 8:
                    mask1 = attn_mask_resize(mask1, mH, mW)  # [BS x H x W]
                    mask2 = attn_mask_resize(mask2, mh, mw)  # [BS x h x w]
                    
                    attn_mask = mask1.reshape(bs, -1).unsqueeze(-1) * mask2.reshape(bs, -1).unsqueeze(1)  # [BS x HW x hw]               
                    attn_mask = repeat(attn_mask, "b HW hw -> (b h) HW hw", h=h)
        
                    assert attn_mask.shape == sim.shape, f"mask : {attn_mask.shape}, attn map : {sim.shape}"   
                             
                    if not use_attention_tv_loss:
                        max_neg_value = -torch.finfo(sim.dtype).max
                        sim.masked_fill_(attn_mask, max_neg_value)
                        
            else:
                raise NotImplementedError
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)
               
        
        sim = sim.softmax(dim=-1)  # [(BSxh) x HW x hw]

        attn_loss = torch.tensor(0, dtype=x.dtype, device=x.device)
        out = einsum('b i j, b j d -> b i d', sim, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)

        if not use_attention_tv_loss:
            return self.to_out(out)
        else:
            return self.to_out(out), attn_loss

class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0, zero_init=False, **kwargs):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head
        if not zero_init:
            self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
            self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
            self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        else:
            self.to_q = zero_module(nn.Linear(query_dim, inner_dim, bias=False))
            self.to_k = zero_module(nn.Linear(context_dim, inner_dim, bias=False))
            self.to_v = zero_module(nn.Linear(context_dim, inner_dim, bias=False))

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))
        self.attention_op: Optional[Any] = None

    def forward(
            self, 
            x,
            context=None, 
            mask=None, 
            hint=None, 
            mask1=None, 
            mask2=None, 
            use_attention_tv_loss=False, 
            use_loss=True, 
            **kwargs
        ):
        q = self.to_q(x)
        is_self_attn = context is None
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)
        key_token_length = k.shape[1]
        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        attn_loss = torch.tensor(0, dtype=x.dtype, device=x.device)
        if use_attention_tv_loss and key_token_length > 700 and (not is_self_attn) and key_token_length < 3000 and use_loss:
            sim = einsum('b i d, b j d -> b i j', q, k) * (self.dim_head ** -0.5)
            sim = sim.softmax(dim=-1)
            h = self.heads
            _, HW, hw = sim.shape
            dx = int((HW//12) ** 0.5)
            mH = int(4*dx)
            mW = int(3*dx)
            dx = int((hw//12) ** 0.5)
            mh = int(4*dx)
            mw = int(3*dx)
            
            mask1 = attn_mask_resize(mask1, mH, mW)  # [BS x H x W]
            reshaped_sim = sim.reshape(-1, h, mH*mW, mh, mw).mean(dim=1) 
            mask1_repeat = mask1
            h_linspace = torch.linspace(0,mh-1,mh, device=sim.device)
            w_linspace = torch.linspace(0,mw-1,mw, device=sim.device)
            grid_h, grid_w = torch.meshgrid(h_linspace, w_linspace)
            grid_hw = torch.stack([grid_h, grid_w])
            
            weighted_grid_hw = reshaped_sim.unsqueeze(2) * grid_hw.unsqueeze(0).unsqueeze(0)  # [b HW 2 h w]
            weighted_centered_grid_hw = weighted_grid_hw.sum((-2,-1))  # [b HW 2]

            tv_loss = get_tvloss(weighted_centered_grid_hw, ~mask1_repeat, ch=mh, cw=mw)
            attn_loss = tv_loss * 0.001
        
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        # if not (use_attention_tv_loss or use_center_loss):
        if not use_attention_tv_loss:
            return self.to_out(out)
        else:
            return self.to_out(out), attn_loss
    
class BasicTransformerBlock(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttention,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False, attn_drop=0.0,attn_res=None,use_learnable_temperature=False,is_lora=False,lora_context_dim=None):
        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None, attn_drop=attn_drop,attn_res=attn_res,use_learnable_temperature=use_learnable_temperature,is_lora=is_lora,lora_context_dim=lora_context_dim)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout, attn_drop=attn_drop,attn_res=attn_res,use_learnable_temperature=use_learnable_temperature)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None,hint=None):
        if hint is None:
            return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)
        else:
            return checkpoint(self._forward, (x, context, hint), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None,hint=None):
        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None,hint=hint) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x

class SpatialTransformer(nn.Module):
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
                 use_checkpoint=True, attn_drop=0.0,attn_res=None,use_learnable_temperature=False, is_lora=False,lora_context_dim=None):
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
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim[d],
                                   disable_self_attn=disable_self_attn, checkpoint=use_checkpoint, attn_drop=attn_drop,attn_res=attn_res,use_learnable_temperature=use_learnable_temperature,is_lora=is_lora,lora_context_dim=lora_context_dim)
                for d in range(depth)]
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
        self.is_lora = is_lora

    def forward(self, x, context=None,hint=None):
        # note: if no context is given, cross-attention defaults to self-attention
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
            x = block(x, context=context[i],hint=hint)
        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in