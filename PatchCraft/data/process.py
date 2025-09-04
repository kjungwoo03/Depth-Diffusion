import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
from random import random, choice
import copy

from scipy.ndimage.filters import gaussian_filter
import torch.nn.functional as F
import torch.distributed as dist
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import ImageDraw




def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:

        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)



    return Image.fromarray(img)

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def gaussian_blur_gray(img, sigma):
    if len(img.shape) == 3:
        img_blur = np.zeros_like(img)
        for i in range(img.shape[2]):
            img_blur[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    else:
        img_blur = gaussian_filter(img, sigma=sigma)
    return img_blur

def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

def cv2_jpg_gray(img, compress_val):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    decimg = cv2.imdecode(encimg, 0)
    return decimg

def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):

    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])

def processing(img, opt, name):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.CropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.CropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
    trans = transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt) if opt.isTrain else img),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN[name], std=STD[name]),
                ])
    return trans(img)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def normlize_np(img):
    img -= img.min()
    if img.max()!=0: img /= img.max()
    return img * 255.   

def ED(img):
    r1,r2 = img[:,0:-1,:], img[:,1::,:]
    r3,r4 = img[:,:,0:-1], img[:,:,1::]
    r5,r6 = img[:,0:-1,0:-1], img[:,1::,1::]
    r7,r8 = img[:,0:-1,1::], img[:,1::,0:-1]
    s1 = torch.sum(torch.abs(r1 - r2)).item()
    s2 = torch.sum(torch.abs(r3 - r4)).item()
    s3 = torch.sum(torch.abs(r5 - r6)).item()
    s4 = torch.sum(torch.abs(r7 - r8)).item() 
    return s1+s2+s3+s4


def visualize_patches_on_pil(pil_img, poor_coords, rich_coords, alpha=80):
    """
    pil_img: PIL.Image (원본)
    poor_coords, rich_coords: [(top, left, h, w), ...]
    alpha: 0~255 (투명도)
    return: PIL.Image with overlays
    """
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    # poor = red
    for (top, left, h, w) in poor_coords:
        draw.rectangle([left, top, left + w - 1, top + h - 1],
                       fill=(255, 0, 0, alpha), outline=(255, 0, 0, min(255, alpha+80)))

    # rich = blue
    for (top, left, h, w) in rich_coords:
        draw.rectangle([left, top, left + w - 1, top + h - 1],
                       fill=(0, 0, 255, alpha), outline=(0, 0, 255, min(255, alpha+80)))

    # 합성
    out = pil_img.convert("RGBA")
    out = Image.alpha_composite(out, overlay)
    return out.convert("RGB")


def processing_RPTC(pil_img, opt):
    """
    기존 로직을 유지하되,
    - RandomCrop.get_params로 (top, left, h, w) 좌표를 뽑아서 기록
    - 정렬 후 poor/rich에 사용된 패치들의 원본 좌표 목록을 리턴
    """
    rate = opt.patch_num
    num_block = int(pow(2, rate))
    patchsize = int(opt.loadSize / num_block)

    # 최소 크기 보장 (원래 코드 유지)
    minsize = min(pil_img.size)
    if minsize < patchsize:
        pil_img = torchvision.transforms.Resize((patchsize, patchsize))(pil_img)

    # Tensor 변환
    img = torchvision.transforms.ToTensor()(pil_img)  # (3, H, W)
    H, W = img.shape[1], img.shape[2]

    # 템플릿
    img_template = torch.zeros(3, opt.loadSize, opt.loadSize)
    # 랜덤 크롭 준비 (좌표를 직접 얻기 위해 get_params 사용)
    img_crops = []  # (tensor, texture_rich, (top,left,h,w))
    total_needed = num_block * num_block
    total_candidates = total_needed * 3  # 기존 로직 유지

    for _ in range(total_candidates):
        # (top, left, h, w) = (i, j, patch_h, patch_w)
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(img, (patchsize, patchsize))
        cropped = img[:, i:i+h, j:j+w]
        texture_rich = ED(cropped)  # 사용자 함수: 텍스처 풍부도 스칼라
        img_crops.append((cropped, float(texture_rich), (i, j, h, w)))

    # 정렬 (오름차순: 가난한→풍부한)
    img_crops.sort(key=lambda x: x[1])

    # poor / rich 세트 선택 (원본 좌표 기록)
    poor_set = img_crops[:total_needed]
    rich_set = img_crops[-total_needed:]  # 가장 풍부한 N개

    poor_coords = [t[2] for t in poor_set]  # [(top,left,h,w), ...]
    rich_coords = [t[2] for t in rich_set]

    # 모자이크 이미지 구성 (기존과 동일한 배치 순서)
    count = 0
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:, ii*patchsize:(ii+1)*patchsize, jj*patchsize:(jj+1)*patchsize] = poor_set[count][0]
            count += 1
    img_poor = img_template.clone().unsqueeze(0)

    count = 0
    for ii in range(num_block):
        for jj in range(num_block):
            # rich_set은 아직 오름차순이므로 뒤쪽이 진짜 리치.
            # 풍부한 것부터 배치하려면 rich_set을 뒤에서부터 쓰거나, 미리 역정렬.
            patch = rich_set[-1 - count][0]  # 뒤에서부터
            img_template[:, ii*patchsize:(ii+1)*patchsize, jj*patchsize:(jj+1)*patchsize] = patch
            count += 1
    img_rich = img_template.clone().unsqueeze(0)

    pair = torch.cat((img_poor, img_rich), dim=0)  # (2, 3, loadSize, loadSize)

    meta = {
        "num_block": num_block,
        "patchsize": patchsize,
        "poor_coords": poor_coords,  # 원본 좌표
        "rich_coords": rich_coords,  # 원본 좌표
        "orig_size": (H, W),
    }
    return pair, meta


def processing_RPTC_tensor(tensor_img, opt):
    rate = opt.patch_num
    num_block = int(pow(2, rate))
    patchsize = int(opt.loadSize / num_block)

    img = tensor_img  # (3, H, W)
    H, W = img.shape[1], img.shape[2]

    # 최소 크기 보장
    minsize = min(H, W)
    if minsize < patchsize:
        # tensor를 리사이즈
        img = F.interpolate(img.unsqueeze(0), size=(patchsize, patchsize), mode='bilinear', align_corners=False).squeeze(0)
        H, W = img.shape[1], img.shape[2]

    # 템플릿
    img_template = torch.zeros(3, opt.loadSize, opt.loadSize)
    # 랜덤 크롭 준비 (좌표를 직접 얻기 위해 get_params 사용)
    img_crops = []  # (tensor, texture_rich, (top,left,h,w))
    total_needed = num_block * num_block
    total_candidates = total_needed * 3  # 기존 로직 유지

    for _ in range(total_candidates):
        # (top, left, h, w) = (i, j, patch_h, patch_w)
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(img, (patchsize, patchsize))
        cropped = img[:, i:i+h, j:j+w]
        texture_rich = ED(cropped)  # 사용자 함수: 텍스처 풍부도 스칼라
        img_crops.append((cropped, float(texture_rich), (i, j, h, w)))

    # 정렬 (오름차순: 가난한→풍부한)
    img_crops.sort(key=lambda x: x[1])

    # poor / rich 세트 선택 (원본 좌표 기록)
    poor_set = img_crops[:total_needed]
    rich_set = img_crops[-total_needed:]  # 가장 풍부한 N개

    poor_coords = [t[2] for t in poor_set]  # [(top,left,h,w), ...]
    rich_coords = [t[2] for t in rich_set]

    # 모자이크 이미지 구성 (기존과 동일한 배치 순서)
    count = 0
    for ii in range(num_block):
        for jj in range(num_block):
            img_template[:, ii*patchsize:(ii+1)*patchsize, jj*patchsize:(jj+1)*patchsize] = poor_set[count][0]
            count += 1
    img_poor = img_template.clone().unsqueeze(0)

    count = 0
    for ii in range(num_block):
        for jj in range(num_block):
            # rich_set은 아직 오름차순이므로 뒤쪽이 진짜 리치.
            # 풍부한 것부터 배치하려면 rich_set을 뒤에서부터 쓰거나, 미리 역정렬.
            patch = rich_set[-1 - count][0]  # 뒤에서부터
            img_template[:, ii*patchsize:(ii+1)*patchsize, jj*patchsize:(jj+1)*patchsize] = patch
            count += 1
    img_rich = img_template.clone().unsqueeze(0)

    pair = torch.cat((img_poor, img_rich), dim=0)  # (2, 3, loadSize, loadSize)

    meta = {
        "num_block": num_block,
        "patchsize": patchsize,
        "poor_coords": poor_coords,  # 원본 좌표
        "rich_coords": rich_coords,  # 원본 좌표
        "orig_size": (H, W),
    }
    return pair, meta