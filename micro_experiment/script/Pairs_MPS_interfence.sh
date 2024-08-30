#!/bin/bash
online_model_list=("resnet50"  "resnet152" "vgg19" "vgg16" "inception_v3" "unet" "deeplabv3" "mobilenet_v2" "alexnet" "bert")
workdir=/data/zbw/inference_system/MIG_MPS
log_path=/data/zbw/inference_system/MIG_MPS/log/Pairs_MPS_QPS
percentage_list=(90 80 70 60 50)


declare -A range_dict
range_dict[90]="42 531507836"
range_dict[80]="42 48"
range_dict[70]="39 47"
range_dict[60]="35 42"
range_dict[50]="29 40"
range_dict[40]="26 33"
range_dict[30]="20 26"
range_dict[20]="13 17"
range_dict[10]="1 6"


for percentage in "${percentage_list[@]}"; do
    range=(${range_dict[$percentage]})
    min=${range[0]}
    max=${range[1]}

    remain=$((100 - percentage))
    remain_range=(${range_dict[$remain]})
    remain_min=${remain_range[0]}
    remain_max=${remain_range[1]}

    echo "percentage: $percentage"  
    for (( i=$min; i<=$max; i++ )); do
        for (( j=$remain_min; j<=$remain_max; j++ )); do
            (cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
            && echo set_active_thread_percentage 530252 $percentage | nvidia-cuda-mps-control \
            && export CUDA_VISIBLE_DEVICES=MIG-2428a716-ba1a-5eae-959f-22f6c93b0f14 && python entry.py --task resnet50 --batch $i --concurrent_profile True --config $percentage+$i --file_name $log_path) \
            &  (sleep 5 && cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
            && echo set_active_thread_percentage 530252 $remain | nvidia-cuda-mps-control \
            && export CUDA_VISIBLE_DEVICES=MIG-2428a716-ba1a-5eae-959f-22f6c93b0f14 && python entry.py --task resnet50 --batch $j --concurrent_profile True --config $remain+$j --file_name $log_path) 

            wait
        done
    done

done

# for percentage in "${percentage_list[@]}"; do
#     echo "percentage: $percentage"
#     echo set_active_thread_percentage 530252 $percentage| nvidia-cuda-mps-control
#     for i in {16..54} ; do
#         cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
#         && export CUDA_VISIBLE_DEVICES=MIG-2428a716-ba1a-5eae-959f-22f6c93b0f14 && python entry.py --task resnet50 --batch $i --concurrent_profile True --config $percentage+$i --file_name $log_path
        

#         echo "Iteration: $i"
#     done
# done