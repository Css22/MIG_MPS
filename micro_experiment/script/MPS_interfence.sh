#!/bin/bash
online_model_list=("resnet50"  "resnet152" "vgg19" "vgg16" "inception_v3" "unet" "deeplabv3" "mobilenet_v2" "alexnet" "bert")
workdir=/data/zbw/inference_system/MIG_MPS
sudo nvidia-smi -i 0 -mig 0
export CUDA_VISIBLE_DEVICES=0 && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log && sudo -E nvidia-cuda-mps-control -d 
python $workdir/warmup.py
    


(cd /data/zbw/inference_system/MIG_MPS/Jobs && python entry.py --task resnet50 --batch 32 --concurrent_profile True --jobs resnet50-32,resnet50-32 --file_name micro) &
(cd /data/zbw/inference_system/MIG_MPS/Jobs && python entry.py --task resnet50 --batch 32 --concurrent_profile True --jobs resnet50-32,resnet50-32 --file_name micro)