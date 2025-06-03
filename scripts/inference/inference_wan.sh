#!/bin/bash
script_name=$(basename "$0")
script_name_no_ext="${script_name%.sh}"

num_gpus=1
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29509 \
    fastvideo/sample/inference_wan.py \
    --height 480 \
    --width 832 \
    --num_frames 81 \
    --num_inference_steps 6 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 7 \
    --flow-reverse \
    --prompt_path path_to_prompt_txt \
    --seed 1024 \
    --output_path outputs_video/${script_name_no_ext}/res/ \
    --model_path ckpt/DCM_WAN/
