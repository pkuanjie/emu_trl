rm -rf ./save
export NCCL_DEBUG=INFO
CUDA_VISIBLE_DEVICES="0,1,2,3" \
accelerate launch examples/scripts/ddpo_emu1.py \
        --num_epochs=400 \
        --train_gradient_accumulation_steps=4 \
        --sample_num_steps=5 \
        --sample_batch_size=2 \
        --train_batch_size=2 \
        --sample_num_batches_per_epoch=4 \
        --per_prompt_stat_tracking=True \
        --per_prompt_stat_tracking_buffer_size=32 \
        --mixed_precision="bf16" \
        --tracker_project_name="ddpo_emu1_aesthetic_score" \
        --log_with="wandb" \
