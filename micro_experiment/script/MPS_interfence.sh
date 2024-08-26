#!/bin/bash
online_model_list=("resnet50"  "resnet152" "vgg19" "vgg16" "inception_v3" "unet" "deeplabv3" "mobilenet_v2" "alexnet" "bert")
workdir=/data/zbw/inference_system/MIG_MPS
log_path=/data/zbw/inference_system/MIG_MPS/log/unet_MPS_QPS
percentage_list=(100 90 80 70 60 50 40 30 20 10)
# export CUDA_VISIBLE_DEVICES=0 && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log && sudo -E nvidia-cuda-mps-control -d 

for percentage in "${percentage_list[@]}"; do
    echo "percentage: $percentage"
    echo set_active_thread_percentage 1271225 $percentage| nvidia-cuda-mps-control
    for i in {10..85} ; do
        cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-MIG-8d8edcc0-a345-5c28-a7d3-994dd98b522f  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-MIG-8d8edcc0-a345-5c28-a7d3-994dd98b522f \
        && export CUDA_VISIBLE_DEVICES=MIG-8d8edcc0-a345-5c28-a7d3-994dd98b522f && python entry.py --task mobilenet_v2 --batch $i --concurrent_profile True --config $percentage+$i --file_name $log_path

        echo "Iteration: $i"
    done
done