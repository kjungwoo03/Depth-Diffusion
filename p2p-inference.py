import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import numpy as np
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor


class CrossAttnMapSaver(AttnProcessor):
    def __init__(self, storage_dict, layer_name, timestep_ref):
        super().__init__()
        self.storage_dict = storage_dict
        self.layer_name = layer_name
        self.timestep_ref = timestep_ref

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        q = attn.to_q(hidden_states)
        k = attn.to_k(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)
        v = attn.to_v(encoder_hidden_states if encoder_hidden_states is not None else hidden_states)

        q = attn.head_to_batch_dim(q)
        k = attn.head_to_batch_dim(k)
        v = attn.head_to_batch_dim(v)

        scale = 1 / (q.shape[-1] ** 0.5)
        attention_scores = torch.baddbmm(
            torch.empty(q.shape[0], q.shape[1], k.shape[1], device=q.device, dtype=q.dtype),
            q, k.transpose(-1, -2), beta=0, alpha=scale
        )
        attention_probs = attention_scores.softmax(dim=-1)

        timestep = self.timestep_ref[0]
        if timestep not in self.storage_dict:
            self.storage_dict[timestep] = {}
        if self.layer_name not in self.storage_dict[timestep]:
            self.storage_dict[timestep][self.layer_name] = []
        self.storage_dict[timestep][self.layer_name].append(attention_probs.detach().cpu())

        hidden_states = torch.bmm(attention_probs, v)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        return attn.to_out[0](hidden_states)


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A photo of a cat wearing a space suit")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def init_latent(model, height, width, generator, batch_size):
    latent = torch.randn(
        (1, model.unet.in_channels, height // 8, width // 8),
        generator=generator,
        device=model.device,
        dtype=torch.float16
    )
    latents = latent.expand(batch_size, -1, -1, -1).contiguous()
    return latents, latents


def visualize_attention(attention_maps, tokens, save_dir="attention_viz", num_heads=8):
    os.makedirs(save_dir, exist_ok=True)
    token_indices = list(range(len(tokens)))
    timesteps = sorted(attention_maps.keys())

    for t in timesteps:
        for layer_name, maps in attention_maps[t].items():
            stacked = torch.stack(maps)  # [N, B*H, Q, K]
            avg_heads = stacked.mean(dim=0)  # [B*H, Q, K]

            # 1. Average over heads and visualize all tokens
            avg_map = avg_heads.mean(dim=0)  # [Q, K]

            fig, axs = plt.subplots(1, len(tokens), figsize=(1.5 * len(tokens), 2))
            if len(tokens) == 1:
                axs = [axs]
            for idx, token_idx in enumerate(token_indices):
                axs[idx].imshow(avg_map[:, token_idx].reshape(int(avg_map.shape[0] ** 0.5), -1), cmap='viridis')
                axs[idx].set_title(tokens[token_idx], fontsize=6)
                axs[idx].axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/layer_{layer_name.replace('.', '_')}_t{t}_avg_heads.png", dpi=300)
            plt.close()

            # 2. Individual heads
            for head in range(num_heads):
                fig, axs = plt.subplots(1, len(tokens), figsize=(1.5 * len(tokens), 2))
                if len(tokens) == 1:
                    axs = [axs]
                head_map = avg_heads[head]
                for idx, token_idx in enumerate(token_indices):
                    axs[idx].imshow(head_map[:, token_idx].reshape(int(head_map.shape[0] ** 0.5), -1), cmap='viridis')
                    axs[idx].set_title(f"{tokens[token_idx]}", fontsize=6)
                    axs[idx].axis('off')
                plt.tight_layout()
                plt.savefig(f"{save_dir}/layer_{layer_name.replace('.', '_')}_t{t}_head{head}.png", dpi=300)
                plt.close()

            # 3. Evolution over timesteps for one token
            for token_idx in token_indices:
                fig, axs = plt.subplots(len(timesteps), 1, figsize=(2, 2 * len(timesteps)))
                for i, ts in enumerate(timesteps):
                    if layer_name not in attention_maps[ts]:
                        continue
                    temp_stack = torch.stack(attention_maps[ts][layer_name])
                    avg_h = temp_stack.mean(dim=0).mean(dim=0)  # [Q, K]
                    axs[i].imshow(avg_h[:, token_idx].reshape(int(avg_h.shape[0] ** 0.5), -1), cmap='viridis')
                    axs[i].set_title(f"t={ts}, token={tokens[token_idx]}", fontsize=6)
                    axs[i].axis('off')
                plt.tight_layout()
                plt.savefig(f"{save_dir}/layer_{layer_name.replace('.', '_')}_evolution_token{token_idx}.png", dpi=300)
                plt.close()


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

    attention_maps = {}
    current_timestep = [None]

    for name, module in ldm.unet.named_modules():
        if hasattr(module, "set_processor") and "attn2" in name:
            module.set_processor(CrossAttnMapSaver(attention_maps, name, current_timestep))

    tokenizer = ldm.tokenizer
    text_encoder = ldm.text_encoder

    text_input = tokenizer(args.prompt, padding='max_length', truncation=True, max_length=77, return_tensors='pt').to(device)
    tokens = tokenizer.convert_ids_to_tokens(text_input["input_ids"][0])

    with torch.no_grad():
        text_embeddings = text_encoder(**text_input).last_hidden_state.half()

    uncond_input = tokenizer([""], padding='max_length', truncation=True, max_length=77, return_tensors='pt').to(device)
    with torch.no_grad():
        uncond_embeddings = text_encoder(**uncond_input).last_hidden_state.half()

    context = torch.cat([uncond_embeddings, text_embeddings], dim=0)

    latents, _ = init_latent(ldm, 512, 512, torch.Generator(device=device).manual_seed(args.seed), 1)

    ldm.scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = ldm.scheduler.timesteps

    record_steps = list(range(0, args.num_inference_steps, 10))

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            current_timestep[0] = int(t.item())
            latent_input = torch.cat([latents] * 2).half()
            
            noise_pred = ldm.unet(latent_input, t, encoder_hidden_states=context)["sample"]
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = ldm.scheduler.step(noise_pred, t, latents).prev_sample

    visualize_attention(attention_maps, tokens)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    main(args)
