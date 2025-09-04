import os
import sys
import json
import random
import glob
import cv2
import argparse
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import depth_pro

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as nn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode


class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext)))
            self.image_paths.extend(glob.glob(os.path.join(root_dir, ext.upper())))
        self.image_paths.sort()
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, image_path

def parse_args(argv):
    parser = argparse.ArgumentParser("Single-image Depth vs Edge Boundary Evaluation")
    parser.add_argument("--input_dir", type=str, default="scratch2/p2p-inference/COCO/SD1.4", help="Input image path")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Execution device")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Edge detection parameters
    parser.add_argument("--canny_low", type=int, default=80, help="Canny low threshold")
    parser.add_argument("--canny_high", type=int, default=160, help="Canny high threshold")
    parser.add_argument("--gaussian_ksize", type=int, default=3, help="Gaussian kernel size (odd). 0 to disable")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0, help="Gaussian σ")
    parser.add_argument("--min_component", type=int, default=10, help="Small edge component removal threshold (pixel count). 0 to disable")
    
    # SI Boundary Recall parameters
    parser.add_argument("--t_min", type=float, default=1.05, help="Minimum threshold t")
    parser.add_argument("--t_max", type=float, default=1.25, help="Maximum threshold t")
    parser.add_argument("--num_t", type=int, default=10, help="Number of thresholds N")
    
    return parser.parse_args(argv)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    # --- Device & seed ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    if args.seed is not None:
        set_seed(args.seed)
        print(f"[INFO] Frozen to Random Seed: {args.seed}")

    # --- Dataset & Dataloader ---]
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    dataset = SimpleImageDataset(root_dir=args.input_dir, transform=None)    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
    )
    print(f"[INFO] Number of Total Batches: {len(dataloader)}")

    caption_dict = {}
    caption_dir = os.path.join(args.input_dir, "captions.txt")
    with open(caption_dir, "r", encoding="utf-8") as f:
        captions = f.readlines()

    for caption in captions:
        caption = caption.strip()
        if "::" in caption:
            idx, caption_text = caption.split("::", 1)
            idx = idx.strip()
            caption_text = caption_text.strip()
            caption_dict[idx] = caption_text
            
    print(f"[INFO] Number of Total Captions: {len(caption_dict)}")

    # --- Model ---
    model, _ = depth_pro.create_model_and_transforms()
    model.eval()
    model.to(device)
    print(f"[INFO] Depth Pro Model Successfully Loaded..")


    with torch.no_grad():
        # Depth estimation
        for images, _ in tqdm(dataloader):
            print(f"Images: {images.shape}")
            print(f"_: {_}")

            break
    #         images = images.to(device)
    #         images = transform(images)

    #         output = model.infer(images, f_px=None)
    #         depth = output["depth"]

    #         # Denormalize original images
    #         original_imgs = (images * 0.5 + 0.5).clamp(0, 1)

    #         # Save depth
    #         depth = depth.detach().cpu()
    #         depth = depth.numpy()
    #         depth = depth[0]
    #         depth = depth.transpose(1, 2, 0)
    #         depth = depth * 255.0
    #         depth = depth.astype(np.uint8)
    #         cv2.imwrite(f"depth_{idx}.png", depth)

    #         # Save original images
    #         original_imgs = original_imgs.detach().cpu()
    #         original_imgs = original_imgs.numpy()
    #         original_imgs = original_imgs[0]
    #         original_imgs = original_imgs.transpose(1, 2, 0)
    #         original_imgs = original_imgs * 255.0
    #         original_imgs = original_imgs.astype(np.uint8)
    #         cv2.imwrite(f"original_{idx}.png", original_imgs)

    #         break


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
    
    
    