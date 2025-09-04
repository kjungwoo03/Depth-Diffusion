import os
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import depth_pro
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
import random
from torchvision import transforms
from torchvision.transforms import InterpolationMode


def arg_parse(argv):
    parser = argparse.ArgumentParser("PatchCraft Inference (Multi-GPU)")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/Chameleon/test/",
        help="Input image root directory (must contain class subfolders)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="batch size",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="scratch2/depth_viz/Chameleon",
        help="Output directory for depth visualizations",
    )
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

    # --- Transforms ---
    # 주의: transform에서 .to(device) 하지 말고, 배치에서 이동하세요.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # --- Dataset & Dataloader ---
    full_dataset = ImageFolder(root=args.input_dir, transform=transform)
    print(f"[INFO] class_to_idx: {full_dataset.class_to_idx}")

    # 클래스별 인덱스 모으기
    targets = full_dataset.targets  # list[int], ImageFolder가 제공
    num_classes = len(full_dataset.classes)
    per_class_quota = int(np.ceil(2000 / num_classes))

    rng = np.random.default_rng(args.seed)  # 재현성
    class_to_indices = {c: [] for c in range(num_classes)}
    for idx, y in enumerate(targets):
        class_to_indices[y].append(idx)

    # 각 클래스에서 quota만큼 샘플(있으면), 전부 합치고 총 2000으로 잘라내기
    balanced_indices = []
    for c in range(num_classes):
        idxs = class_to_indices[c]
        if len(idxs) <= per_class_quota:
            chosen = idxs
        else:
            chosen = rng.choice(idxs, size=per_class_quota, replace=False).tolist()
        balanced_indices.extend(chosen)

    # 최종 2000개로 자르고 섞기(완전 랜덤 순서 원하면)
    rng.shuffle(balanced_indices)
    balanced_indices = balanced_indices[:min(2000, len(balanced_indices))]

    dataset = torch.utils.data.Subset(full_dataset, balanced_indices)
    print(f"[INFO] Selected {len(dataset)} samples across {num_classes} classes (balanced)")


    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    print(f"[INFO] Number of Total Batches: {len(dataloader)}")

    # --- Model ---
    model, _ = depth_pro.create_model_and_transforms()
    model.eval()
    model.to(device)

    # --- Inference ---
    with torch.no_grad():
        for idx, (images, labels) in enumerate(tqdm(dataloader, desc="Inference")):
            images = images.to(device)

            # depth inference
            output = model.infer(images, f_px=None)
            depth = output["depth"] 

            # Denormalize original images
            original_imgs = (images * 0.5 + 0.5).clamp(0, 1)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # 첫 번째 이미지
            original_img = original_imgs[0].cpu().permute(1, 2, 0).numpy()
            ax1.imshow(original_img)
            ax1.set_title("Original Image")
            ax1.axis('off')

            # depth map
            depth_map = depth.cpu().numpy()
            im = ax2.imshow(depth_map, cmap='viridis')
            ax2.set_title("Depth Map")
            ax2.axis('off')
            plt.colorbar(im, ax=ax2, shrink=0.6)

            class_idx = int(labels[0].item())
            label = dataset.dataset.classes[class_idx]

            out_dir = os.path.join(args.output_dir, label)
            os.makedirs(out_dir, exist_ok=True)
            save_path = os.path.join(out_dir, f"img{idx}.png")

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

    print("[INFO] Inference finished.")



if __name__ == "__main__":
    args = arg_parse(sys.argv[1:])
    main(args)
