for algo in ADM Glide VQDM cyclegan progan stable_diffusion_v_1_4 stargan stylegan2 wukong DALLE2 Midjourney biggan gaugan sd_xl stable_diffusion_v_1_5 stylegan whichfaceisreal; do
    python patchcraft-benchmark.py --input_dir datasets/AIGC/test/${algo}
done