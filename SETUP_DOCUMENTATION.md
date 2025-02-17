# StableVITON Setup Documentation

## 1. Environment Setup

### 1.1 Conda Environment
```bash
conda create --name StableVITON python=3.10 -y
conda activate StableVITON
```

### 1.2 Dependencies Installation
First, installed PyTorch with CUDA support:
```bash
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
```

Other dependencies (requirements.txt):
```
pytorch-lightning==1.5.0
einops
opencv-python==4.7.0.72
matplotlib
omegaconf
albumentations
transformers==4.33.2
open-clip-torch==2.19.0
diffusers==0.20.2
scipy==1.10.1
ipython
```

Note: xformers and triton were removed due to compatibility issues on Windows.

## 2. Project Structure

### 2.1 Directory Structure
```
StableVITON/
├── weights/
│   ├── VITONHD.ckpt                    # (~7.36 GB)
│   ├── VITONHD_PBE_pose.ckpt          # (~7.36 GB)
│   └── VITONHD_VAE_finetuning.ckpt    # (~790 MB)
├── data/
│   ├── train/
│   │   ├── image/                      # Person images
│   │   ├── image-densepose/            # DensePose results
│   │   ├── agnostic/                   # Person representations
│   │   ├── agnostic-mask/              # Segmentation masks
│   │   ├── cloth/                      # Clothing items
│   │   ├── cloth_mask/                 # Clothing masks
│   │   └── gt_cloth_warped_mask/       # Ground truth warped masks
│   ├── test/
│   │   ├── image/
│   │   ├── image-densepose/
│   │   ├── agnostic/
│   │   ├── agnostic-mask/
│   │   ├── cloth/
│   │   └── cloth_mask/
│   ├── train_pairs.txt                 # Training image-cloth pairs
│   └── test_pairs.txt                  # Testing image-cloth pairs
└── create_pairs.py                     # Script to generate pairs.txt files
```

### 2.2 Dataset Statistics
- Training set: ~11,645 person images and clothing items
- Test set: ~2,032 person images and clothing items

## 3. Important Files

### 3.1 Model Weights
Three checkpoint files are required:
- `VITONHD.ckpt`: Main model checkpoint
- `VITONHD_PBE_pose.ckpt`: Pose-based embedding model
- `VITONHD_VAE_finetuning.ckpt`: Fine-tuned VAE model

### 3.2 Pairs Files Format
The `*_pairs.txt` files contain space-separated pairs of filenames:
```
person_image.jpg cloth_image.jpg
```
- `train_pairs.txt`: Contains paired data (matching person-cloth pairs)
- `test_pairs.txt`: Contains combinations of test images with multiple clothing items

### 3.3 Directory Contents
Each subdirectory contains specific types of images:
- `image/`: Original person images
- `image-densepose/`: DensePose results for person understanding
- `agnostic/`: Person representations without clothing
- `agnostic-mask/`: Segmentation masks for person parsing
- `cloth/`: Target clothing items
- `cloth_mask/`: Binary masks of clothing items
- `gt_cloth_warped_mask/`: Ground truth warped clothing masks (train only)

## 4. Scripts and Tools

### 4.1 create_pairs.py
Python script to generate train and test pair files:
```python
# Usage
python create_pairs.py
```
Functions:
- Creates paired data for training
- Generates test combinations for evaluation
- Outputs: train_pairs.txt and test_pairs.txt

## 5. Next Steps

1. Testing the Installation:
   - Run a sample inference to verify setup
   - Check GPU compatibility
   - Verify data loading

2. Potential Issues:
   - Missing xformers might affect performance
   - Windows compatibility considerations
   - GPU memory requirements (~16GB recommended)

## 6. Troubleshooting

Common issues and solutions:
1. CUDA compatibility: Ensure GPU drivers are up to date
2. Memory issues: Reduce batch size if needed
3. Missing files: Verify all directories contain the correct number of files
4. File permissions: Ensure write access to all directories

## 7. References

- Original Repository: [StableVITON GitHub](https://github.com/rlawjdghek/StableVITON)
- Dataset Source: Kaggle VITON-HD dataset
- Paper: [StableVITON: Learning Semantic Correspondence with Latent Diffusion Model for Virtual Try-On](https://arxiv.org/abs/2312.01725)
