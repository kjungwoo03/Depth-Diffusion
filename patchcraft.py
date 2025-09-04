import os
import sys
import time
import argparse
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from numpy import mean
from tqdm import tqdm
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision

import matplotlib.pyplot as plt

from PatchCraft.networks import Net as RPTC
from PatchCraft.networks import initWeights
from PatchCraft.data.process import ED


def parse_args(argv=None):
    parser = argparse.ArgumentParser("PatchCraft/RPTC Inference (end-to-end)")
    parser.add_argument("--input_dir", type=str, default="datasets/Chameleon/test/")
    parser.add_argument("--checkpoint", type=str, default="scratch2/pretrained_models/RPTC.pth")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (pair 생성 특성상 1 권장)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resize", type=int, default=256, help="base resize (학습과 동일해야 함)")
    parser.add_argument("--norm", type=str, default="none", choices=["none", "imagenet", "minus1to1"])

    parser.add_argument("--patch_num", type=int, default=3, help="2^patch_num 그리드")
    parser.add_argument("--loadSize", type=int, default=256, help="pair 템플릿 크기")
    parser.add_argument("--candidates_mult", type=int, default=3)

    parser.add_argument("--out_dir", type=str, default="./runs/patchcraft", help="figure/log 저장 폴더")
    parser.add_argument("--save_plots", action="store_true", help="히스토그램/ROC 저장")
    
    return parser.parse_args(argv)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
