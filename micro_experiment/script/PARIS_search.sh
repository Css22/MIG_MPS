online_model_list=("mobilenet_v2" "alexnet" "resnet50" "resnet101" "resnet152" "vgg19" "vgg16" "unet" "deeplabv3" "bert")


for model_name in "${online_model_list[@]}"; do
    cd /data/zbw/inference_system/MIG_MPS/baseline/PARIS_ELSA && \
    /home/zbw/anaconda3/envs/Abacus/bin/python PARIS_search.py --task $model_name
done