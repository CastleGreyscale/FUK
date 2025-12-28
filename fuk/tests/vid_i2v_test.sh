python /home/brad/ai/fuked03/fuk/vendor/musubi-tuner/wan_generate_video.py --fp8 --task i2v-14B --video_size 720 1280 --video_length 21 --infer_steps 20 \
--prompt "A man looks into camera with a painful look on his face." --save_path outputs/save.mp4 --output_type both \
--dit /home/brad/ai/models/checkpoints/wan2.1_i2v_720p_14B_fp8_e4m3fn.safetensors --vae /home/brad/ai/models/vae/wan_2.1_vae.safetensors \
--t5 /home/brad/ai/models/clip/models_t5_umt5-xxl-enc-bf16.pth --clip /home/brad/ai/models/clip/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
--vae_cache_cpu \
--blocks_to_swap 20 \
--fp8_t5 \
--fp8 \
--attn_mode torch --image_path inputs/testimg01.png
