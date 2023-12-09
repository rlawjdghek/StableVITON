# StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On
This repository is the official implementation of [StableVITON](https://arxiv.org/abs/2312.01725)

> **StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On**<br>
> [Jeongho Kim](https://scholar.google.co.kr/citations?user=ucoiLHQAAAAJ&hl=ko), [Gyojung Gu](https://www.linkedin.com/in/gyojung-gu-29033118b/), [Minho Park](https://pmh9960.github.io/), [Sunghyun Park](https://psh01087.github.io/), [Jaegul Choo](https://sites.google.com/site/jaegulchoo/) 

[[Arxiv Paper](https://arxiv.org/abs/2312.01725)]&nbsp;
[[Website Page](https://rlawjdghek.github.io/StableVITON/)]&nbsp;

![teaser](assets/teaser.png)&nbsp;

## TODO List
- [x] ~~Inference code~~
- [x] ~~Release model weights~~
- [ ] Training code

## Environments
```bash
git clone https://github.com/rlawjdghek/StableVITON
cd StableVITON

conda create --name StableVITON python=3.10 -y
conda activate StableVITON

# install packages
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install pytorch-lightning==1.5.0
pip install einops
pip install opencv-python==4.7.0.72
pip install matplotlib
pip install omegaconf
pip install transformers==4.33.2
pip install xformers==0.0.19
pip install triton==2.0.0
pip install open-clip-torch==2.19.0
pip install diffusers==0.20.2
pip install scipy==1.10.1
conda install -c anaconda ipython -y
```

## Weights and Data
You can download the VITON-HD dataset from [here](https://github.com/shadow2496/VITON-HD).<br>
To download the model weights, please fill the [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdmAZL7J9_CBNhVuRqaL25UQuOYzL5zVJM0Q0cWi54rJFR1Vg/viewform?usp=sf_link) related to the consent.<br>
The input data should include (1) agnostic-map (2) agnostic-mask (3) cloth (4) densepose. For testing VITONHD, the test dataset should be organized as follows:

```
test
|-- image
|-- image-densepose
|-- agnostic
|-- agnostic-mask
|-- cloth
```

## Preprocessing
The VITON-HD dataset serves as a benchmark and provides an agnostic mask. However, you can attempt virtual try-on on **arbitrary images** using segmentation tools like [SAM](https://github.com/facebookresearch/segment-anything). Please note that for densepose, you should use the same densepose model as used in VITON-HD.

## Inference
```bash
# paired setting
python inference.py --config_path ./configs/VITON512.yaml --batch_size 4 --model_load_path <model weight path> --save_dir <save directory>

# unpaired setting
python inference.py --config_path ./configs/VITON512.yaml --batch_size 4 --model_load_path <model weight path> --unpair --save_dir <save directory>
```

You can also preserve the unmasked region by '--repaint' option. 


## Citation
If you find our work useful for your research, please cite us:
```
@artical{kim2023stableviton,
    title={StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On},
    author={Kim, Jeongho and Gu, Gyojung and Park, Minho and Park, Sunghyun and Choo, Jaegul},
    booktitle={arXiv preprint arxiv:2312.01725},
    year={2023}
}
```

**Acknowledgements** Sunghyun Park is the corresponding author.

## License
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).