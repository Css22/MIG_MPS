#!/bin/bash
# online_model_list=("mobilenet_v2" "alexnet" "resnet50" "resnet101" "resnet152" "vgg19" "vgg16" "unet" "deeplabv3")
online_model_list=("resnet50" "resnet101" "resnet152" "vgg19" "vgg16" "unet" "deeplabv3")
uuid=GPU-906931c6-0f94-edc4-3f18-17fc4e477e53
for model_name in "${online_model_list[@]}"; do
    echo $model_name 
    cd /data/zbw/inference_system/MIG_MPS/baseline/gpulet && \
    export CUDA_VISIBLE_DEVICES=$uuid && \
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$uuid && \
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$uuid && \
    /data/zbw/anaconda3/envs/Abacus/bin/python gpulet_scheduler.py --task $model_name 

done