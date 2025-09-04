# Introduction
This is the official test code implementation of the paper "PatchCraft: Exploring Texture Patch for Efficient AI-generated Image Detection".

# Usage
To run the detection tool, use the following command:

```bash
python eval_all.py [--batch_size <value>] [--no_flip] [--no_crop] [--no_resize] [--noise_type <option>]
```
You should set your dataroot and dataset name in `eval_config.py`
Pre-trained detection model is available in `./weights`


# Parameters
- `--model_path <model_path>`  *(Optional, default: './weights/RPTC.pth')* The path of detection model.
- `--batch_size <value>`: Input batch size
- `--no_flip`: *(Optional)* If specified, do not flip the images for data augmentation.
- `--no_crop`: *(Optional)* If specified, do not crop the images for data augmentation.
- `--no_resize`: *(Optional)* If specified, do not resize the images for data augmentation.
- `--noise_type <option>`: *(Optional, default: None)* The benign image perturbations. Choose from: `None`, `jpg`, `blur`, `resize`.
- `--blur_sig <value>`:  *(Optional, default: 1.0)* The standard deviation (σ) of the Gaussian blur kernel applied to the image, as a benign image perturbation.
- `--jpg_method <method>`:  *(Optional, default: 'pil')* The JPEG compression method used as a benign image perturbation. Choose from `pil`,`cv2`.
- `--jpg_qual <value>`: The JPEG compression quality level.
- `--patch_num <value> `: *(Optional, default: 3)* Scale images to this size, used in detection method PatchCraft.

# Example

```bash
python eval_all.py --batch_size 1 --no_flip --no_crop --no_resize
```





## Citation

If you find this work useful for your research, please kindly cite our paper:

```
@article{zhong2023patchcraft,
  title={Patchcraft: Exploring texture patch for efficient ai-generated image detection},
  author={Zhong, Nan and Xu, Yiran and Li, Sheng and Qian, Zhenxing and Zhang, Xinpeng},
  journal={arXiv preprint arXiv:2311.12397},
  year={2023}
}
```

