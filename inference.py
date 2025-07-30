import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from diffusers import StableDiffusionPipeline

def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A photo of an astronaut riding a horset")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)

def init_latent(model, height, width, generator, batch_size):
    in_channels = model.unet.config.in_channels  # FutureWarning 대응
    latent = torch.randn(
        (1, in_channels, height // 8, width // 8),
        generator=generator,
        device=model.device,
        dtype=torch.float16
    )
    latents = latent.expand(batch_size, -1, -1, -1).contiguous()
    return latents, latents

def decode_latents(ldm, latents):
    latents = (1 / 0.18215) * latents
    latents = latents.to(torch.float16)
    with torch.no_grad():
        image = ldm.vae.decode(latents).sample
    
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])
    return image

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_id = "CompVis/stable-diffusion-v1-4"
    ldm = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=False,
        safety_checker=None
    ).to(device)

    tokenizer = ldm.tokenizer
    text_encoder = ldm.text_encoder

    # 텍스트 인코딩
    text_input = tokenizer(args.prompt, padding='max_length', truncation=True, max_length=77, return_tensors='pt').to(device)
    with torch.no_grad():
        text_embeddings = text_encoder(**text_input).last_hidden_state.half()

    # Unconditional 인코딩
    uncond_input = tokenizer([""], padding='max_length', truncation=True, max_length=77, return_tensors='pt').to(device)
    with torch.no_grad():
        uncond_embeddings = text_encoder(**uncond_input).last_hidden_state.half()

    context = torch.cat([uncond_embeddings, text_embeddings], dim=0)

    # Latent 초기화
    latents, _ = init_latent(ldm, 512, 512, torch.Generator(device=device).manual_seed(args.seed), 1)

    # Scheduler 설정
    ldm.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = ldm.scheduler.timesteps

    # 중간 결과 저장을 위한 디렉토리 생성
    os.makedirs("results/intermediate", exist_ok=True)

    # 시각화할 timestep 간격 계산 (10개의 중간 결과)
    vis_steps = list(range(0, len(timesteps), len(timesteps)//10))

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            latent_input = torch.cat([latents] * 2).half()
            noise_pred = ldm.unet(latent_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = ldm.scheduler.step(noise_pred, t, latents).prev_sample

            # 선택된 timestep에서 중간 결과 저장
            if i in vis_steps:
                intermediate_image = decode_latents(ldm, latents)
                save_path = f"results/intermediate/step_{i:03d}.png"
                intermediate_image.save(save_path)
                print(f"Intermediate result saved to {save_path}")

    # 최종 이미지 저장
    final_image = decode_latents(ldm, latents)
    save_path = f"results/generated_{args.seed}.png"
    final_image.save(save_path)
    print(f"Final image saved to {save_path}")


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
