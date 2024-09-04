online_model_list=("mobilenet_v2" "alexnet" "resnet50" "resnet101" "resnet152" "vgg19" "vgg16" "unet" "deeplabv3" "bert")

uuid=MIG-d82118da-7798-5081-959f-c8bbf24989b3
for model_name in "${online_model_list[@]}"; do
    echo $model_name 
    cd /data/zbw/inference_system/MIG_MPS/baseline/gpulet && \
    export CUDA_VISIBLE_DEVICES=$uuid && \
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-$uuid && \
    export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-$uuid && \
    /home/zbw/anaconda3/envs/Abacus/bin/python gpulet_scheduler.py --task $model_name 

done