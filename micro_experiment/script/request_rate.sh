#!/bin/bash
online_model_list=("resnet50" "vgg19" "unet" "deeplabv3" "mobilenet_v2" "bert")
workdir=/data/zbw/inference_system/MIG_MPS
sudo nvidia-smi -i 0 -mig 0
export CUDA_VISIBLE_DEVICES=0 && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log && sudo -E nvidia-cuda-mps-control -d 
python $workdir/warmup.py
percentage_list=(10 20 30 40 50 60 70 80 90)

for model in ${offline_model_list[@]}; do
    for percentage in "${percentage_list[@]}"; do
        python $workdir/MPS_server.py --percentage $percentage && python $workdir/micro_experiment/requset_rate.py --task $model --SM percentage
    done
done