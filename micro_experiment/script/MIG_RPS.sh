#!/bin/bash


# GPU  0 Profile ID 19 Placements: {0,1,2,3,4,5,6}:1  MIG 1g.10gb
# GPU  0 Profile ID 20 Placements: {0,1,2,3,4,5,6}:1  MIG 1g.10gb+me
# GPU  0 Profile ID 14 Placements: {0,2,4}:2          MIG 2g.20gb 
# GPU  0 Profile ID  9 Placements: {0,4}:4            MIG 3g.40gb
# GPU  0 Profile ID  5 Placement : {0}:4              MIG 4g.40gb
# GPU  0 Profile ID  0 Placement : {0}:8              MIG 7g.80gb

CU_DEV=CUDA_VISIBLE_DEVICES
dev0='GPU-08dffabe-6be4-81d7-ba7d-1d96612fb099'
GPU_ID=0
workdir=/data/zbw/inference_system/MIG_MPS

online_model_list=("resnet50" "resnet101" "resnet152" "vgg19" "vgg16" "unet" "deeplabv3" "mobilenet_v2" "alexnet" "bert")

GPU_list=$workdir/tmp/MIG_partitions.txt
InstanceID_list=$workdir/tmp/ID.txt

get_uuid_pick="python $workdir/util/getUuidInfo.py --input ${GPU_list} --raw 0 --pick 1 --ch"
get_uuid_unpick="python $workdir/util/getUuidInfo.py --input ${GPU_list} --raw 0 --pick 0 --ch"
get_IID="python $workdir/util/readID.py --input ${InstanceID_list}"

# source /home/hpcroot/anaconda3/etc/profile.d/conda.sh


function create {
    sudo nvidia-smi mig -cgi $1 -C > ${InstanceID_list}
    nvidia-smi -L >${GPU_list}
    python $workdir/util/getUuidAll.py --input $GPU_list --raw 1
    echo "Successfully created"
    ID=$(${get_IID})
}

function destroy {
    sudo nvidia-smi mig -dci -i 0 -gi $1 -ci 0
    sudo nvidia-smi mig -dgi -i 0 -gi $1
}

function test {
    prof_id=$1
    info=$2


   
    echo "Testing on mig: " $info
    create $prof_id
    uuid=$(${get_uuid_pick} ${info})
    
    echo $uuid

    for model_name in "${online_model_list[@]}"; do
        cd /data/zbw/inference_system/MIG_MPS/jobs && export CUDA_VISIBLE_DEVICES=$uuid && python entry.py --task $model_name --file_name /data/zbw/inference_system/MIG_MPS/log/${model_name}_MIG_RPS --config $info
    done
    

    sleep 3
    destroy $ID
}

sudo nvidia-smi -i 0 -mig 0

sleep 3
for model in ${online_model_list[@]}; do
    python $workdir/jobs/online/entry.py --config baseline  --task $model --batch 32
done

echo 'open MIG model'

sudo nvidia-smi -i 0 -mig 1

test 0 1c-7g-80gb 
test 5 1c-4g-40gb 
test 9 1c-3g-40gb 
test 14 1c-2g-20gb
test 19 1c-1g-10gb

echo 'close MIG model'
sleep 5
sudo nvidia-smi -i 0 -mig 0