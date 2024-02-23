export NCCL_DEBUG=INFO
# ACCELERATE_CONFIG='./accelerate_config_4gpu.yaml'
ACCELERATE_CONFIG='./accelerate_config_1gpu.yaml'
# ACCELERATE_CONFIG='./ds_config_1gpu.yaml'
accelerate  launch --config_file $ACCELERATE_CONFIG examples/scripts/ddpo_emu1_lmm_unet.py \
    --num_epochs=2 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=5\
    --sample_batch_size=1 \
    --train_batch_size=1 \
    --sample_num_batches_per_epoch=1 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --mixed_precision="bf16" \
    --train_use_8bit_adam=true \
    --sample_guidance_scale=7.5 \
    --tracker_project_name="ddpo_emu1_lmm_unet_aesthetic_score" \
    --log_with="wandb" \

# accum 4
# sam bs 4
# train bs 2
# sam b pr e 4
