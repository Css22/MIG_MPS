#!/bin/bash
online_model_list=("bert")
workdir=/data/zbw/inference_system/MIG_MPS
log_path=/data/zbw/inference_system/MIG_MPS/log/
declare -A seen

# 枚举所有可能的分配组合
for model in "${online_model_list[@]}"; do
    for A in {10..80..10}; do
        for B in {10..80..10}; do
            for C in {10..80..10}; do
                # 确保总和为100
                if [ $((A + B + C)) -eq 100 ]; then
                    array=($A $B $C)
                    sorted_array=($(echo "${array[@]}" | tr ' ' '\n' | sort -n | tr '\n' ' '))
                    key="${sorted_array[0]}_${sorted_array[1]}_${sorted_array[2]}"
                    if [ -z "${seen[$key]}" ]; then
                        echo "A=$A B=$B C=$C"
                        
                        output=$(cd /data/zbw/inference_system/MIG_MPS/util && python util.py --task $model --SM $A)
                        min_batch_1=$(echo $output | awk '{print $1}')
                        max_batch_1=$(echo $output | awk '{print $2}')


                        output=$(cd /data/zbw/inference_system/MIG_MPS/util && python util.py --task $model --SM $B)
                        min_batch_2=$(echo $output | awk '{print $1}')
                        max_batch_2=$(echo $output | awk '{print $2}')
                        
                        output=$(cd /data/zbw/inference_system/MIG_MPS/util && python util.py --task $model --SM $C)
                        min_batch_3=$(echo $output | awk '{print $1}')
                        max_batch_3=$(echo $output | awk '{print $2}')
                        
                        min_batch_1=1
                        min_batch_2=1
                        min_batch_3=1
                        

                        for (( i=$min_batch_1; i<=$max_batch_1; i++ )); do

                            for (( j=$min_batch_2; j<=$max_batch_2; j++ )); do
                                
                                for (( k=$min_batch_3; z<=$max_batch_3; k++ )); do


                                    (cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
                                    && echo set_active_thread_percentage 75743 $A | nvidia-cuda-mps-control \
                                    && export CUDA_VISIBLE_DEVICES=MIG-e806816b-27b9-54dd-87dd-c52b4e695397 && python entry.py --task $model --config $A --batch $i --concurrent_profile --test) \

                                    &  (sleep 5 && cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
                                    && echo set_active_thread_percentage 75743 $B | nvidia-cuda-mps-control \
                                    && export CUDA_VISIBLE_DEVICES=MIG-e806816b-27b9-54dd-87dd-c52b4e695397 && python entry.py --task $model --config $B --batch $j --concurrent_profile --test) 


                                    &  (sleep 10 && cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log \
                                    && echo set_active_thread_percentage 75743 $C | nvidia-cuda-mps-control \
                                    && export CUDA_VISIBLE_DEVICES=MIG-e806816b-27b9-54dd-87dd-c52b4e695397 && python entry.py --task $model --config $C --batch $k --concurrent_profile --test) 

                                    wait

                                done

                            done

                        done
                        
                        seen[$key]=1
                    fi
                fi
            done
        done
    done
done
