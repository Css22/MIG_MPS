#!/bin/bash
workdir=/data/wyh/MIG_MPS/jobs
python_path=/home/wyh/miniconda3/envs/Abacus/bin/python



model1=$1
model2=$2
sm1=$3
sm2=$4
batch=$5
max_RPS=$6
server_id=$7
device=$8
port=$9

echo $model $sm1 $sm2 $batch $max_RPS $server_id
(cd $workdir &&  export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$device  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$device \
&& echo set_active_thread_percentage $server_id $sm1 | nvidia-cuda-mps-control \
&& export CUDA_VISIBLE_DEVICES=$device &&  $python_path entry.py --task $model1 --config $sm1 --batch $batch --concurrent_profile --test --bayes --feedback --running --port $port --GI $device) \
&  (sleep 10 && cd $workdir &&  export export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$device  &&  export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$device \
&& echo set_active_thread_percentage $server_id $sm2 | nvidia-cuda-mps-control \
&& export CUDA_VISIBLE_DEVICES=$device &&  $python_path entry.py --task $model2 --config $sm2 --RPS $max_RPS --concurrent_profile --test --bayes --feedback --port $port --GI $device) 

wait
echo "padding done"
            




