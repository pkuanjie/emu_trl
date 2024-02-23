export NCCL_DEBUG=INFO
# ACCELERATE_CONFIG='./ds_config_1gpu.yaml'
ACCELERATE_CONFIG='./accelerate_config_4gpu.yaml'
# ACCELERATE_CONFIG='./accelerate_config_1gpu.yaml'
accelerate  launch --config_file $ACCELERATE_CONFIG examples/scripts/ddpo_emu1_lmm.py \
    --num_epochs=400 \
    --train_gradient_accumulation_steps=4 \
    --sample_num_steps=50\
    --sample_batch_size=4 \
    --train_batch_size=2 \
    --sample_num_batches_per_epoch=4 \
    --per_prompt_stat_tracking=True \
    --per_prompt_stat_tracking_buffer_size=32 \
    --mixed_precision="bf16" \
    --train_use_8bit_adam=true \
    --sample_guidance_scale=7.5 \
    --tracker_project_name="ddpo_emu1_lmm_aesthetic_score" \
    --log_with="wandb"
