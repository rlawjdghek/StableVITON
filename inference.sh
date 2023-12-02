# VITONHD pair
CUDA_VISIBLE_DEVICES=3 python inference.py --config_path ./configs/VITON512.yaml --batch_size 1 --model_load_path ./ckpts/VITONHD.ckpt --save_dir VITONHD

# VITONHD unpair
CUDA_VISIBLE_DEVICES=3 python inference.py --config_path ./configs/VITON512.yaml --batch_size 1 --model_load_path ./ckpts/VITONHD.ckpt --unpair --save_dir VITONHD

# VITONHD pair repaint
CUDA_VISIBLE_DEVICES=3 python inference.py --config_path ./configs/VITON512.yaml --batch_size 1 --model_load_path ./ckpts/VITONHD.ckpt --repaint --save_dir VITONHD_repaint

# VITONHD unpair repaint
CUDA_VISIBLE_DEVICES=3 python inference.py --config_path ./configs/VITON512.yaml --batch_size 1 --model_load_path ./ckpts/VITONHD.ckpt --unpair --repaint --save_dir VITONHD_repaint


# dummy
CUDA_VISIBLE_DEVICES=5 python inference.py --config_path ./configs/VITON512.yaml --batch_size 1 --model_load_path ./ckpts/VITONHD.ckpt --save_dir samples_new2