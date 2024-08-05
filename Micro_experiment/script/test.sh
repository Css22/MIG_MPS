#!/bin/bash


online_model_list=("resnet50"  "resnet152" "vgg19" "vgg16" "inception_v3" "unet" "deeplabv3" "mobilenet_v2" "alexnet" "bert")


for model1 in ${online_model_list[@]}; do
    cd /data/zbw/inference_system/MIG_MPS/Jobs && python entry.py --task $model1 --batch 32
done

