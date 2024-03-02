export NCCL_DEBUG=INFO
export WANDB_API_KEY="cb10a6c8187e45acea4deb3db30a0db8535a010d"
export WANDB__SERVICE_WAIT=300

# ACCELERATE_CONFIG='./accelerate_config_4gpu.yaml'
ACCELERATE_CONFIG='./accelerate_config_1gpu.yaml'
MAIN_PROCESS_PORT=12377
# pip install -e .
# ACCELERATE_CONFIG='./accelerate_config_1gpu.yaml'
# torchrun examples/scripts/ddpo_emu1.py \
accelerate  launch --config_file $ACCELERATE_CONFIG --main_process_port $MAIN_PROCESS_PORT examples/scripts/sft_emu1_unet.py \
    --num_epochs=400 \
    --train_gradient_accumulation_steps=2 \
    --sample_num_steps=10 \
    --sample_batch_size=2 \
    --train_batch_size=2 \
    --mixed_precision="bf16" \
    --train_use_8bit_adam=true \
    --sample_guidance_scale=0.0 \
    --tracker_project_name="sft_emu1_unet" \
    --log_with="wandb" \
    --main_name='sft_emu1_unet' \
