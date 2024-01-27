export NCCL_P2P_LEVEL=NVL
export CUDA_VISIBLE_DEVICES='0,1'
NUM_GPUS=2
ACCELERATE_CONFIG='./accelerate_config.yaml'
rm -rf ./save
accelerate launch --num_processes=$NUM_GPUS examples/scripts/ddpo.py
# accelerate launch --config_file $ACCELERATE_CONFIG --num_processes=$NUM_GPUS examples/scripts/ddpo.py
# accelerate launch --num_processes=$NUM_GPUS --gpu_ids=$CUDA_VISIBLE_DEVICES examples/scripts/ddpo.py
