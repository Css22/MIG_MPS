#!/bin/bash
workdir=/data/zbw/inference_system/MIG_MPS
log_path=/data/zbw/inference_system/MIG_MPS/log/

model1=$1
model2=$2
sm1=$3
sm2=$4
batch=$5
max_RPS=$6
server_id=$7

device=GPU-906931c6-0f94-edc4-3f18-17fc4e477e53
echo $model1 $model2 $sm1 $sm2 $batch $max_RPS $server_id
(cd /data/zbw/inference_system/MIG_MPS/jobs &&  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$device  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$device \
&& echo set_active_thread_percentage $server_id $sm1 | nvidia-cuda-mps-control \
&& export CUDA_VISIBLE_DEVICES=$device &&  /data/zbw/anaconda3/envs/Abacus/bin/python entry.py --task $model1 --config $sm1 --batch $batch --concurrent_profile --test --bayes --feedback --running) \
&  (sleep 5 && cd /data/zbw/inference_system/MIG_MPS/jobs &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$device  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$device \
&& echo set_active_thread_percentage $server_id $sm2 | nvidia-cuda-mps-control \
&& export CUDA_VISIBLE_DEVICES=$device &&  /data/zbw/anaconda3/envs/Abacus/bin/python entry.py --task $model2 --config $sm2 --RPS $max_RPS --concurrent_profile --test --bayes --feedback) 

wait
echo "padding done"
            




