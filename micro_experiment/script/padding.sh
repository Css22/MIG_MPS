#!/bin/bash
workdir=/data/zbw/inference_system/MIG_MPS
log_path=/data/zbw/inference_system/MIG_MPS/log/

model1=$1
model2=$2
sm1=$3
sm2=$4
batch1=$5
batch2=$6


                
echo $model1 $sm1 $batch1 
echo $model2 $sm2 $batch2

device=MIG-e806816b-27b9-54dd-87dd-c52b4e695397

(cd /data/zbw/inference_system/MIG_MPS/jobs &&  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$device  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$device \
&& echo set_active_thread_percentage 1688702 $sm1 | nvidia-cuda-mps-control \
&& export CUDA_VISIBLE_DEVICES=$device &&  /home/zbw/anaconda3/envs/Abacus/bin/python entry.py --task $model1 --config $sm1 --batch $batch1 --concurrent_profile --test) \
&  (sleep 5 && cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$device  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$device \
&& echo set_active_thread_percentage 1688702 $sm2 | nvidia-cuda-mps-control \
&& export CUDA_VISIBLE_DEVICES=$device &&  /home/zbw/anaconda3/envs/Abacus/bin/python entry.py --task $model2 --config $sm2 --batch $batch2 --concurrent_profile --test) 

wait
echo "padding done"
            




