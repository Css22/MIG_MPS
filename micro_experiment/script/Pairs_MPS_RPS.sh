#!/bin/bash
online_model_list=("bert")
workdir=/data/zbw/inference_system/MIG_MPS
log_path=/data/zbw/inference_system/MIG_MPS/log/
percentage_list=(90 80 70 60 50)



for model in "${online_model_list[@]}"; do

    for percentage in "${percentage_list[@]}"; do

        remain=$((100 - percentage))

        output=$(cd /data/zbw/inference_system/MIG_MPS/util && python util.py --task $model --SM $percentage)
        min_batch_1=$(echo $output | awk '{print $1}')
        max_batch_1=$(echo $output | awk '{print $2}')


        output=$(cd /data/zbw/inference_system/MIG_MPS/util && python util.py --task $model --SM $remain)
        min_batch_2=$(echo $output | awk '{print $1}')
        max_batch_2=$(echo $output | awk '{print $2}')

        min_batch_1=1
        min_batch_2=1

        for (( i=$min_batch_1; i<=$max_batch_1; i++ )); do

            for (( j=$min_batch_2; j<=$max_batch_2; j++ )); do
                
                echo $model $percentage $i $j
                (cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
                && echo set_active_thread_percentage 75743 $remain | nvidia-cuda-mps-control \
                && export CUDA_VISIBLE_DEVICES=MIG-e806816b-27b9-54dd-87dd-c52b4e695397 && python entry.py --task $model --config $remain --batch $j --concurrent_profile --test) \
                &  (sleep 10 && cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
                && echo set_active_thread_percentage 75743 $percentage | nvidia-cuda-mps-control \
                && export CUDA_VISIBLE_DEVICES=MIG-e806816b-27b9-54dd-87dd-c52b4e695397 && python entry.py --task $model --config $percentage --batch $i --concurrent_profile --test) 

                wait
            done

        done

    done

done 






# for percentage in "${percentage_list[@]}"; do

#     remain=$((100 - percentage))
#     remain_range=(${range_dict[$remain]})

#     for model in "${online_model_list[@]}"; do


#     done

#     echo "percentage: $percentage"  
#     for (( i=$min; i<=$max; i++ )); do
#         for (( j=$remain_min; j<=$remain_max; j++ )); do
                # (cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
                # && echo set_active_thread_percentage 530252 $percentage | nvidia-cuda-mps-control \
                # && export CUDA_VISIBLE_DEVICES=MIG-2428a716-ba1a-5eae-959f-22f6c93b0f14 && python entry.py --task resnet50 --concurrent_profile True --config $percentage+$i --file_name $log_path) \
                # &  (sleep 5 && cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
                # && echo set_active_thread_percentage 530252 $remain | nvidia-cuda-mps-control \
                # && export CUDA_VISIBLE_DEVICES=MIG-2428a716-ba1a-5eae-959f-22f6c93b0f14 && python entry.py --task resnet50 --concurrent_profile True --config $remain+$j --file_name $log_path) 
#             wait
#         done
#     done

# done

# for percentage in "${percentage_list[@]}"; do
#     echo "percentage: $percentage"
#     echo set_active_thread_percentage 530252 $percentage| nvidia-cuda-mps-control
#     for i in {16..54} ; do
#         cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
#         && export CUDA_VISIBLE_DEVICES=MIG-2428a716-ba1a-5eae-959f-22f6c93b0f14 && python entry.py --task resnet50 --batch $i --concurrent_profile True --config $percentage+$i --file_name $log_path
        

#         echo "Iteration: $i"
#     done
# done