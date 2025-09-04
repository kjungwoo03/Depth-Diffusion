import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile

import torchvision
import os
import copy
import torch
from scipy import fftpack
import imageio
from skimage.transform import resize
from .process import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize) # 随机剪裁，默认224
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img) # 不处理
    else:
        crop_func = transforms.CenterCrop(opt.cropSize) # 中心裁剪

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))

    dset = datasets.ImageFolder(
            root,
            transforms.Compose([
                rz_func,
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]))
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path




rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    width, height = img.size
    # print('before resize: '+str(width)+str(height))
    # quit()
    interp = sample_discrete(opt.rz_interp)
    img = torchvision.transforms.Resize((opt.loadSize,opt.loadSize))(img) 
    return img


def custom_augment(img, opt):
    
    # print('height, width:'+str(height)+str(width))
    # resize
    if opt.noise_type=='resize':
        if opt.detect_method=='Fusing':
            height, width = img.shape[0], img.shape[1]
            img = resize(img, (int(height/2), int(width/2)))
        else:
            height, width = img.height, img.width
            img = torchvision.transforms.Resize((int(height/2),int(width/2)))(img) 

    img = np.array(img)
    # img = img[0:-1:4,0:-1:4,:]
    if opt.noise_type=='blur':
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if opt.noise_type=='jpg':
        
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)
    
    return Image.fromarray(img)


def loadpathslist(root,flag):
    classes =  os.listdir(root)
    paths = []
    if not '1_fake' in classes:
        for class_name in classes:
            imgpaths = os.listdir(root+'/'+class_name +'/'+flag+'/')
            for imgpath in imgpaths:
                paths.append(root+'/'+class_name +'/'+flag+'/'+imgpath)
        return paths
    else:
        imgpaths = os.listdir(root+'/'+flag+'/')
        for imgpath in imgpaths:
            paths.append(root+'/'+flag+'/'+imgpath)
        return paths


    



class read_data_new():
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        real_img_list = loadpathslist(self.root,'0_real')    
        real_label_list = [0 for _ in range(len(real_img_list))]
        fake_img_list = loadpathslist(self.root,'1_fake')
        fake_label_list = [1 for _ in range(len(fake_img_list))]
        self.img = real_img_list+fake_img_list
        self.label = real_label_list+fake_label_list

        # print('directory, realimg, fakeimg:', self.root, len(real_img_list), len(fake_img_list))


    def __getitem__(self, index):
        img, target = Image.open(self.img[index]).convert('RGB'), self.label[index]
        imgname = self.img[index]
        # compute scaling
        height, width = img.height, img.width
        if (not self.opt.isTrain) and (not self.opt.isVal):
            img = custom_augment(img, self.opt)
 
        img = processing_RPTC(img, self.opt)

        return img, target

    def __len__(self):
        return len(self.label)
