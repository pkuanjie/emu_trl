export NCCL_DEBUG=INFO
# ACCELERATE_CONFIG='./accelerate_config_4gpu.yaml'
ACCELERATE_CONFIG='./accelerate_config_4gpu_fsdp.yaml'
accelerate  launch --config_file $ACCELERATE_CONFIG examples/scripts/ddpo_emu1.py \
    --num_epochs=400 \
    --train_gradient_accumulation_steps=1 \
    --sample_num_steps=1\
    --sample_batch_size=4 \
    --train_batch_size=2 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --mixed_precision="bf16" \
    --sample_guidance_scale=7.5 \
    --tracker_project_name="ddpo_emu1_aesthetic_score" \
    --log_with="wandb" \

# accum 4
# sam bs 4
# train bs 2
# sam b pr e 4
