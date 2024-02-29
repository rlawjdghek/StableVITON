import os
import torch

from omegaconf import OmegaConf
import transformers
from ldm.util import instantiate_from_config


def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    if transformers.__version__ != "4.19.2" and "cond_stage_model.transformer.vision_model.embeddings.position_ids" in state_dict.keys():
        del state_dict["cond_stage_model.transformer.vision_model.embeddings.position_ids"]    
        print(f"delete cond_stage_model.transformer.vision_model.embeddings.position_ids from loaded state dict (transformers version : {transformers.__version__})")
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict

def create_model(config_path, config=None, **kwargs):
    if config is None:
        config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
