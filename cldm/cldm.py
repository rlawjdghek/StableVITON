import os
from os.path import join as opj
import omegaconf

import cv2
import einops
import torch
import torch as th
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config

class ControlLDM(LatentDiffusion):
    def __init__(
            self, 
            control_stage_config, 
            validation_config, 
            control_key, 
            only_mid_control, 
            use_VAEdownsample=False,
            config_name="",
            control_scales=None,
            use_pbe_weight=False,
            u_cond_percent=0.0,
            img_H=512,
            img_W=384,
            always_learnable_param=False,
            *args, 
            **kwargs
        ):
        self.control_stage_config = control_stage_config
        self.use_pbe_weight = use_pbe_weight
        self.u_cond_percent = u_cond_percent
        self.img_H = img_H
        self.img_W = img_W
        self.config_name = config_name
        self.always_learnable_param = always_learnable_param
        super().__init__(*args, **kwargs)
        control_stage_config.params["use_VAEdownsample"] = use_VAEdownsample
        self.control_model = instantiate_from_config(control_stage_config)
        self.control_key = control_key
        self.only_mid_control = only_mid_control
        if control_scales is None:
            self.control_scales = [1.0] * 13
        else:
            self.control_scales = control_scales
        self.first_stage_key_cond = kwargs.get("first_stage_key_cond", None)
        self.valid_config = validation_config
        self.use_VAEDownsample = use_VAEdownsample
    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)
        if isinstance(self.control_key, omegaconf.listconfig.ListConfig):
            control_lst = []
            for key in self.control_key:
                control = batch[key]
                if bs is not None:
                    control = control[:bs]
                control = control.to(self.device)
                control = einops.rearrange(control, 'b h w c -> b c h w')
                control = control.to(memory_format=torch.contiguous_format).float()
                control_lst.append(control)
            control = control_lst
        else:
            control = batch[self.control_key]
            if bs is not None:
                control = control[:bs]
            control = control.to(self.device)
            control = einops.rearrange(control, 'b h w c -> b c h w')
            control = control.to(memory_format=torch.contiguous_format).float()
            control = [control]
        cond_dict = dict(c_crossattn=[c], c_concat=control) 
        if self.first_stage_key_cond is not None:
            first_stage_cond = []
            for key in self.first_stage_key_cond:
                if not "mask" in key:
                    cond, _ = super().get_input(batch, key, *args, **kwargs)
                else:
                    cond, _ = super().get_input(batch, key, no_latent=True, *args, **kwargs)      
                first_stage_cond.append(cond)
            first_stage_cond = torch.cat(first_stage_cond, dim=1)
            cond_dict["first_stage_cond"] = first_stage_cond
        return x, cond_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)       
        
        diffusion_model = self.model.diffusion_model
        cond_txt = torch.cat(cond["c_crossattn"], 1)
        if self.proj_out is not None:
            if cond_txt.shape[-1] == 1024:
                cond_txt = self.proj_out(cond_txt)  # [BS x 1 x 768]
        if self.always_learnable_param:
            cond_txt = self.get_unconditional_conditioning(cond_txt.shape[0])
        
        if cond['c_concat'] is None:
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control)
        else:
            if "first_stage_cond" in cond:
                x_noisy = torch.cat([x_noisy, cond["first_stage_cond"]], dim=1)
            if not self.use_VAEDownsample:
                hint = cond["c_concat"]
            else:
                hint = []
                for h in cond["c_concat"]:
                    if h.shape[2] == self.img_H and h.shape[3] == self.img_W:
                        h = self.encode_first_stage(h)
                        h = self.get_first_stage_encoding(h).detach()
                    hint.append(h)
            hint = torch.cat(hint, dim=1)
            control, _ = self.control_model(x=x_noisy, hint=hint, timesteps=t, context=cond_txt, only_mid_control=self.only_mid_control)
            if len(control) == len(self.control_scales):
                control = [c * scale for c, scale in zip(control, self.control_scales)]
                
            eps = diffusion_model(x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control)
        return eps, None
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        if not self.kwargs["use_imageCLIP"]:
            return self.get_learned_conditioning([""] * N)
        else:
            return self.learnable_vector.repeat(N,1,1)
    def low_vram_shift(self, is_diffusing):
        if is_diffusing:
            self.model = self.model.cuda()
            self.control_model = self.control_model.cuda()
            self.first_stage_model = self.first_stage_model.cpu()
            self.cond_stage_model = self.cond_stage_model.cpu()
        else:
            self.model = self.model.cpu()
            self.control_model = self.control_model.cpu()
            self.first_stage_model = self.first_stage_model.cuda()
            self.cond_stage_model = self.cond_stage_model.cuda()