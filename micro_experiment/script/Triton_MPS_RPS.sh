#!/bin/bash
online_model_list=("resnet50"  "resnet152" "vgg19" "vgg16" "inception_v3" "unet" "deeplabv3" "mobilenet_v2" "alexnet" "bert")
workdir=/data/zbw/inference_system/MIG_MPS
log_path=/data/zbw/inference_system/MIG_MPS/log/unet_MPS_QPS
percentage_list=(100 90 80 70 60 50 40 30 20 10)
#  percentage_list=(100)
# export CUDA_VISIBLE_DEVICES=0 && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log && sudo -E nvidia-cuda-mps-control -d 


for percentage in "${percentage_list[@]}"; do
    echo set_active_thread_percentage 2670804 $percentage| nvidia-cuda-mps-control
    docker run  --ipc=host --gpus "device=MIG-e806816b-27b9-54dd-87dd-c52b4e695397" --rm -p8000:8000 -p8001:8001 -p8002:8002 \
    -v /data/zbw/inference_system/MIG_MPS/inference_system/model_repository:/models  -v /tmp/nvidia-mps:/tmp/nvidia-mps -e CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps nvcr.io/nvidia/tritonserver:23.05-py3 tritonserver --model-repository=/models &
    (sleep 60 && cd /data/zbw/inference_system/MIG_MPS/inference_system && /home/zbw/anaconda3/envs/Abacus/bin/python client.py --task resnet50 --config $percentage && (docker stop $(docker ps -q)))
    sleep 30
done
