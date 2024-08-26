#!/bin/bash
online_model_list=("resnet50"  "resnet152" "vgg19" "vgg16" "inception_v3" "unet" "deeplabv3" "mobilenet_v2" "alexnet" "bert")
workdir=/data/zbw/inference_system/MIG_MPS
log_path=/data/zbw/inference_system/MIG_MPS/log/MPS_QPS
percentage_list=(90 80 70 60 50 40 30 20 10)
# export CUDA_VISIBLE_DEVICES=0 && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log && sudo -E nvidia-cuda-mps-control -d 

for percentage in "${percentage_list[@]}"; do
    echo "percentage: $percentage"
    echo set_active_thread_percentage 530252 $percentage| nvidia-cuda-mps-control
    for i in {16..54} ; do
        cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
        && export CUDA_VISIBLE_DEVICES=MIG-2428a716-ba1a-5eae-959f-22f6c93b0f14 && python entry.py --task resnet50 --batch $i --concurrent_profile True --config $percentage+$i --file_name $log_path

        echo "Iteration: $i"
    done
done