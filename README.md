# StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On
This repository is the official implementation of [StableVITON]()

> **StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On**<br>
> Jeongho Kim, Gyojung Gu, Minho Park, Sunghyun Park, Jaegul Choo,

[[Arxiv Paper]()]&nbsp;
[[Website Page](https://rlawjdghek.github.io/StableVITON/)]&nbsp;

![teaser](assets/teaser.png)&nbsp;

## TODO List
- [x] Inference Code
- [ ] https://github.com/octo-org/octo-repo/issues/740
- [ ] Add delight to the experience when all tasks are complete :tada:
- [x] Completed task title  

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
You can download the pre-trained model at 512x384 resolution from [Download Link](구글 폼 링크). <br>
The input data should include (1) agnostic-map (2) agnostic-mask (3) cloth (4) densepose. For testing VITONHD, the test dataset should be organized as follows:

```
test
|-- image
|-- image-densepose
|-- agnostic
|-- agnostic-mask
|-- cloth
```

### Preprocessing
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
@artical{wu2023lamp,
    title={LAMP: Learn a Motion Pattern by Few-Shot Tuning a Text-to-Image Diffusion Model},
    author={Wu, Ruiqi and Chen, Liangyu and Yang, Tong and Guo, Chunle and Li, Chongyi and Zhang, Xiangyu},
    journal={arXiv preprint arXiv:2310.10769},
    year={2023}
}
```

**Acknowledgements** Sunghyun Park is the corresponding author.

## License
All material is made available under Creative Commons BY-NC 4.0. You can use, redistribute, and adapt the material for non-commercial purposes, as long as you give appropriate credit by citing our paper and indicate any changes that you've made.