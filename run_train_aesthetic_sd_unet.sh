export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2
accelerate launch --num_processes=$NUM_GPUS examples/scripts/ddpo.py
