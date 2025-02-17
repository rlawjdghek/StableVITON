import os

def create_pairs_file(data_dir, split):
    # Read image and cloth filenames
    with open(os.path.join(data_dir, split, 'image_files.txt'), 'r') as f:
        image_files = [line.strip() for line in f.readlines()]
    
    with open(os.path.join(data_dir, split, 'cloth_files.txt'), 'r') as f:
        cloth_files = [line.strip() for line in f.readlines()]
    
    # Create pairs (for training, use paired data)
    pairs = []
    if split == 'train':
        # For training, use paired data (same indices)
        for i in range(min(len(image_files), len(cloth_files))):
            pairs.append(f"{image_files[i]} {cloth_files[i]}")
    else:
        # For testing, create all possible combinations
        for img in image_files[:100]:  # Limit test pairs to first 100 images
            for cloth in cloth_files[:20]:  # Try 20 different clothes on each person
                pairs.append(f"{img} {cloth}")
    
    # Write pairs file
    with open(os.path.join(data_dir, f'{split}_pairs.txt'), 'w') as f:
        f.write('\n'.join(pairs))

# Create pairs for both train and test
data_dir = 'data'
create_pairs_file(data_dir, 'train')
create_pairs_file(data_dir, 'test')
