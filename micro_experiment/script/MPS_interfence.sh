#!/bin/bash
online_model_list=("resnet50"  "resnet152" "vgg19" "vgg16" "unet" "deeplabv3" "mobilenet_v2" "alexnet" "bert")
workdir=/data/zbw/inference_system/MIG_MPS
log_path=/data/zbw/inference_system/MIG_MPS/log/unet_MPS_QPS
percentage_list=(100 90 80 70 60 50 40 30 20 10)
# export CUDA_VISIBLE_DEVICES=0 && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log && sudo -E nvidia-cuda-mps-control -d 

for percentage in "${percentage_list[@]}"; do
    echo "percentage: $percentage"
    echo set_active_thread_percentage 1023136 $percentage| nvidia-cuda-mps-control

    for model_name in "${online_model_list[@]}"; do
        cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
        && export CUDA_VISIBLE_DEVICES=MIG-e806816b-27b9-54dd-87dd-c52b4e695397 && python entry.py --task $model_name --file_name /data/zbw/inference_system/MIG_MPS/log/${model_name}_MPS_RPS --config $percentage
    done
done