import os
import sys
import json
import argparse
from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

import cv2
import torch
import torch.nn.functional as nn
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor

class CrossAttnMapSaver(AttnProcessor):
    def __init__(self, storage_dict, layer_name):
        super().__init__()
        self.storage_dict = storage_dict
        self.layer_name = layer_name

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # 기본 cross-attn 계산
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        v = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)

        # shape: [B, heads, tokens, dim_head]
        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        scale = 1 / (q.shape[-1] ** 0.5)
        attention_scores = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device, dtype=q.dtype),
            q, k.transpose(-1, -2), beta=0, alpha=scale
        )
        attention_probs = attention_scores.softmax(dim=-1)

        # 👇 attention map 저장
        self.storage_dict[self.layer_name] = attention_probs.detach().cpu()

        hidden_states = torch.bmm(attention_probs, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        
        return attn.to_out[0](hidden_states)
    
    
def init_latent(model, height, width, generator, batch_size):
    latent = torch.randn(
        (1, model.unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=model.device,
        dtype=torch.float16 
    )
    latents = latent.expand(batch_size, -1, -1, -1).contiguous()
    return latents, latents



def load_coco_annotations(json_path: str, limit: int = None, start: int = 0):
    """
    returns: list of dicts [{ann_id, image_id, caption}, ...]
    if limit<=0 or None, use all annotations
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    anns = data.get("annotations", [])
    if start > 0:
        anns = anns[start:]
    if (limit is not None) and (limit > 0):
        anns = anns[:limit]
    
    # caption에서 파일 경로에 문제가 될 수 있는 문자들을 제거/치환
    def clean_caption_for_filename(caption):
        # 파일명에 사용할 수 없는 문자들을 언더스코어로 치환
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        cleaned = caption
        for char in invalid_chars:
            cleaned = cleaned.replace(char, '_')
        # 연속된 공백을 하나로 만들고, 앞뒤 공백 제거
        cleaned = ' '.join(cleaned.split())
        # 너무 긴 caption은 잘라내기 (파일명 길이 제한)
        if len(cleaned) > 100:
            cleaned = cleaned[:100]
        return cleaned
    
    out = [{"ann_id": a["id"], "image_id": a["image_id"], "caption": clean_caption_for_filename(a["caption"])} for a in anns]
    
    return out
    
def build_id_to_fname_map(coco_json_path: str) -> Dict[int, str]:
    """
    Read the images array from captions_val2017.json and
    create a mapping from image_id to file_name.
    """
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    images_info = data.get("images", [])
    id2name = {img["id"]: img["file_name"] for img in images_info}
    return id2name

def init_latent(model, height, width, generator, batch_size):
    latent = torch.randn(
        (1, model.unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=model.device,
        dtype=torch.float16 
    )
    latents = latent.expand(batch_size, -1, -1, -1).contiguous()
    return latents, latents


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="coco") 
    parser.add_argument("--model_id", type=str, default="SD2.1")
    # "CompVis/stable-diffusion-v1-4"
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_num_words", type=int, default=77)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_captions", type=int, default=-1)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="./scratch2/p2p-inference/COCO-full")
    return parser.parse_args(argv)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Using GPUs: {torch.cuda.device_count()}")
    
    save_dir = os.path.join(args.save_dir, args.model_id)
    os.makedirs(save_dir, exist_ok=True)
    
    limit = None if args.num_captions <= 0 else args.num_captions
    coco_json_path = "datasets/MS-COCO/annotations/captions_val2017.json"
    ann_list = load_coco_annotations(coco_json_path, limit=limit, start=args.start_index)
    id2name = build_id_to_fname_map(coco_json_path)

    print(f"[COCO] captions loaded: {len(ann_list)}")

    if args.model_id == "SD2.1":
        model_id = "stabilityai/stable-diffusion-2-1"
    elif args.model_id == "SD1.4":
        model_id = "CompVis/stable-diffusion-v1-4"
    else:
        raise ValueError(f"Invalid model_id: {args.model_id}")
    print(f"Now using model: {model_id}")

    ldm = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant='fp16'
    ).to(device)
    
    attention_maps = {}

    for name, module in ldm.unet.named_modules():
        if hasattr(module, 'set_processor') and 'attn2' in name:
            processor = CrossAttnMapSaver(attention_maps, name)
            module.set_processor(processor)

    tokenizer = ldm.tokenizer
    text_encoder = ldm.text_encoder

    print(f"Successfully loaded model..")

    prompts = [ann["caption"] for ann in ann_list]
    HEIGHT = 512
    WIDTH = 512
    BATCH_SIZE = 1

    captions_out = []

    for idx, ann in enumerate(tqdm(ann_list, desc="COCO captions")):
        ann_id   = ann["ann_id"]
        image_id = ann["image_id"]
        caption  = ann["caption"]

        captions_out.append(f"{idx:04d}:: {caption}")

        file_name = id2name.get(image_id, None)
        original_path = None
        if file_name is not None:
            original_path = f"datasets/MS-COCO/val2017/{file_name}"

        if original_path and os.path.exists(original_path):
            # Save original image to args.save_dir/original
            original_save_dir = os.path.join(args.save_dir, "original")
            os.makedirs(original_save_dir, exist_ok=True)

            # caption 대신 안전하게 image_id 위주로 파일명 짓는 게 좋음
            original_save_path = os.path.join(original_save_dir, f"{idx}_{args.seed}_{caption}.png")

            try:
                original_img = Image.open(original_path).convert("RGB")
                original_img.save(original_save_path)
            except Exception as e:
                print(f"[WARN] Failed to save original image for image_id={image_id}: {e}")

        else:
            # 파일이 아예 없거나 file_name이 None일 때
            print(f"[WARN] Original not found for image_id={image_id} (file_name={file_name}, path={original_path})")
            continue


        # Text embedding
        text_input = tokenizer(
            caption,
            padding='max_length',
            truncation=True,
            max_length=args.max_num_words,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            text_embeddings = text_encoder(**text_input).last_hidden_state.half()
        
        # Unconditional embedding
        uncond_input = tokenizer(
            [""],
            padding='max_length',
            truncation=True,
            max_length=args.max_num_words,
            return_attention_mask=True,
            return_tensors='pt'
        ).to(device)
        
        with torch.no_grad():
            uncond_embeddings = text_encoder(**uncond_input).last_hidden_state.half()
            
        context = torch.cat([uncond_embeddings, text_embeddings], dim=0)
        
        # print(text_embeddings.shape) # torch.Size([1, 77, 768])
        # print(uncond_embeddings.shape) # torch.Size([1, 77, 768])
        # print(context.shape) # torch.Size([2, 77, 768])
        
        # Initialize latents
        latents, _ = init_latent(
            ldm,
            HEIGHT,
            WIDTH,
            torch.Generator(device=device).manual_seed(args.seed),
            BATCH_SIZE
        )
        
        # print(latents.shape) # torch.Size([1, 4, 64, 64])
        
        ldm.scheduler.set_timesteps(args.num_inference_steps, device=device)
        timesteps = ldm.scheduler.timesteps

        with torch.no_grad():
            for t in timesteps:
                latent_input = torch.cat([latents] * 2).half()
                
                noise_pred = ldm.unet(
                    latent_input,
                    t,
                    encoder_hidden_states=context
                )["sample"]

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

                latents = ldm.scheduler.step(noise_pred, t, latents).prev_sample
      
            # Decode latents to image
            latents = 1 / 0.18215 * latents
            image = ldm.vae.decode(latents.half())["sample"]          

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image[0])
        pil_image.save(os.path.join(save_dir, f"idx{idx}_seed{args.seed}.png"))

    caption_file = os.path.join(save_dir, "captions.txt")
    with open(caption_file, "w", encoding="utf-8") as f:
        f.write("\n".join(captions_out))

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
