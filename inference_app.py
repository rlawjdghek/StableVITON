import os
from os.path import join as opj
from omegaconf import OmegaConf

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from cldm.plms_hacked import PLMSSampler
from cldm.model import create_model
from dataset import AppDataset
from inference import build_args
from utils import tensor2img


@torch.no_grad()
def predict(im_name, cloth):
    global args
    global model_config
    global model
    global sampler

    batch_size = args.batch_size
    img_H = args.img_H
    img_W = args.img_W

    dataset = AppDataset(
        data_root_dir=args.data_root_dir,
        img_H=img_H,
        img_W=img_W,
        im_name=im_name,
        cloth=cloth,
        is_paired=not args.unpair,
        is_test=True,
    )
    dataloader = DataLoader(dataset, num_workers=4, shuffle=False, batch_size=batch_size, pin_memory=True)

    shape = (4, img_H // 8, img_W // 8)

    #################################################################
    for batch_idx, batch in enumerate(dataloader):
        print(f"{batch_idx}/{len(dataloader)}")
        z, c = model.get_input(batch, model_config.model.params.first_stage_key)
        bs = z.shape[0]
        c_crossattn = c["c_crossattn"][0][:bs]
        if c_crossattn.ndim == 4:
            c_crossattn = model.get_learned_conditioning(c_crossattn)
            c["c_crossattn"] = [c_crossattn]
        uc_cross = model.get_unconditional_conditioning(bs)
        uc_full = {"c_concat": c["c_concat"], "c_crossattn": [uc_cross]}
        uc_full["first_stage_cond"] = c["first_stage_cond"]
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        sampler.model.batch = batch

        ts = torch.full((1,), 999, device=z.device, dtype=torch.long)
        start_code = model.q_sample(z, ts)

        samples, _, _ = sampler.sample(
            args.denoise_steps,
            bs,
            shape,
            c,
            x_T=start_code,
            verbose=False,
            eta=args.eta,
            unconditional_conditioning=uc_full,
        )

        x_samples = model.decode_first_stage(samples)
        for sample_idx, (x_sample, fn, cloth_fn) in enumerate(zip(x_samples, batch['img_fn'], batch["cloth_fn"])):
            x_sample_img = tensor2img(x_sample)  # [0, 255]
            if args.repaint:
                repaint_agn_img = np.uint8((batch["image"][sample_idx].cpu().numpy() + 1) / 2 * 255)  # [0,255]
                repaint_agn_mask_img = batch["agn_mask"][sample_idx].cpu().numpy()  # 0 or 1
                x_sample_img = repaint_agn_img * repaint_agn_mask_img + x_sample_img * (1 - repaint_agn_mask_img)
                x_sample_img = np.uint8(x_sample_img)

            to_path = opj(opj(args.save_dir, "unpair" if args.unpair else "pair"),
                          f"{fn.split('.')[0]}_{cloth_fn.split('.')[0]}.jpg")
            cv2.imwrite(to_path, x_sample_img[:, :, ::-1])

    return x_sample_img


@torch.no_grad()
def init():
    print("init stable viton...")
    global args
    global model_config
    global model
    global sampler

    model_config = OmegaConf.load(args.config_path)
    model_config.model.params.img_H = args.img_H
    model_config.model.params.img_W = args.img_W

    model = create_model(config_path=None, config=model_config)
    model.load_state_dict(torch.load(args.model_load_path, map_location="cpu"))
    model = model.cuda()
    model.eval()

    sampler = PLMSSampler(model)

    save_dir = opj(args.save_dir, "unpair" if args.unpair else "pair")
    os.makedirs(save_dir, exist_ok=True)
    print("init stable viton end...")
    return sampler


args = build_args()
model_config = None
model = None
sampler = None
