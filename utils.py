import numpy as np

def tensor2img(x):
    '''
    x : [BS x c x H x W] or [c x H x W]
    '''
    if x.ndim == 3:
        x = x.unsqueeze(0)
    BS, C, H, W = x.shape
    x = x.permute(0,2,3,1).reshape(-1, W, C).detach().cpu().numpy()
    x = np.clip(x, -1, 1)
    x = (x+1)/2
    x = np.uint8(x*255.0)
    if x.shape[-1] == 1:
        x = np.concatenate([x,x,x], axis=-1)
    return x