import os
import json
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image



def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_segmentation_map(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    seg = np.array(data['panoptic_seg'])
    segments_info = data['segments_info']
    return seg, segments_info

# === 시각화를 위한 팔레트 (필요 시 외부 JSON으로 분리 가능) ===
def get_palette():
    return np.asarray([
        [0, 0, 0],
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ])


def visualize_segmentation(seg, palette):
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        color_seg[seg == label] = color
    return Image.fromarray(color_seg)


def load_controlnet_pipeline(device):
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16
    ).to(device)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception as e:
        print("xformers not available:", e)

    pipe.enable_model_cpu_offload()
    return pipe


def run_generation(pipe, prompt, control_img, device):
    with torch.autocast("cuda" if device.type == "cuda" else "cpu"):
        result = pipe(prompt, control_img, num_inference_steps=20) # inference step 지정 가능
    return result.images[0]


def save_results(orig_img, seg_img, gen_img, result_dir, prefix="cat4"):
    os.makedirs(result_dir, exist_ok=True)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(axs, [orig_img, seg_img, gen_img], ["Original", "Segmentation", "Generated"]):
        ax.imshow(img)
        ax.set_title(f"{title} Image")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, f"{prefix}_segmentation_result.png"), dpi=150)
    plt.close(fig)

    orig_img.save(os.path.join(result_dir, f"{prefix}_orig.png"))
    gen_img.save(os.path.join(result_dir, f"{prefix}_generated.png"))

# === 메인 파이프라인 ===
def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 경로 설정
    input_path = "data/example.jpg"
    seg_json = "results/mask2former/example.json"
    result_dir = "./results/controlnet/"
    prompt = ("A high-resolution photo of an orange and white tabby cat standing on a white windowsill. "
              "The window is open, and green trees fill the background. Soft daylight comes in through the window. "
              "Shot with a 50mm lens, shallow depth of field.")

    # Load original image & segmentation
    orig_img = load_image(input_path).convert("RGB")
    seg, _ = load_segmentation_map(seg_json)
    palette = get_palette()
    seg_img = visualize_segmentation(seg, palette).convert("RGB").resize(orig_img.size, Image.NEAREST)

    # Run generation
    pipe = load_controlnet_pipeline(device)
    gen_img = run_generation(pipe, prompt, seg_img, device)

    # Save results
    save_results(orig_img, seg_img, gen_img, result_dir)

if __name__ == "__main__":
    main()