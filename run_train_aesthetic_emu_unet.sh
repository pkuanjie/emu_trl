export CUDA_VISIBLE_DEVICES='0'
NUM_GPUS=1
ACCELERATE_CONFIG='./accelerate_config.yaml'
rm -rf ./save
accelerate launch --config_file $ACCELERATE_CONFIG --num_processes=$NUM_GPUS examples/scripts/ddpo_emu.py \
        --num_epochs=400 \
        --train_gradient_accumulation_steps=4 \
        --sample_num_steps=50 \
        --sample_batch_size=1 \
        --train_batch_size=1 \
        --sample_num_batches_per_epoch=4 \
        --per_prompt_stat_tracking=True \
        --per_prompt_stat_tracking_buffer_size=32 \
        --mixed_precision="bf16" \
        --tracker_project_name="ddpo_emu_aesthetic_score" \
        --log_with="wandb" \
