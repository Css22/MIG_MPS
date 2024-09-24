import sys
import torch
import time
import pandas as pd 
import numpy as np
import argparse
from bert import BertModel
from alexnet import alexnet
from open_unmix import open_unmix
from mobilenet_v2 import mobilenet
from deeplabv3 import deeplabv3
from Unet import unet
from vgg_splited import vgg16, vgg19
from resnet import resnet50,resnet101,resnet152
from inception_ve import inception_v3
from transformer import transformer_layer
import signal
import math
import logging


path = "/data/zbw/inference_system/MIG_MPS/jobs/"
sys.path.append(path)
flag_path = "/data/zbw/MIG/MIG/MIG_Schedule/flag"
result_path = "/data/zbw/inference_system/MIG_MPS/log/"

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


model_list = {
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
    "vgg19": vgg19,
    "vgg16": vgg16,
    "inception_v3": inception_v3,
    'unet': unet,
    'deeplabv3':deeplabv3,
    'mobilenet_v2': mobilenet,
    # 'open_unmix':open_unmix,
    'alexnet': alexnet,
    'bert': BertModel,
    'transformer': transformer_layer,
}

input_list = {
    "resnet50": [3, 244, 244],
    "resnet101": [3, 244, 244],
    "resnet152": [3, 244, 244],
    "vgg19": [3, 244, 244],
    "vgg16": [3, 244, 244],
    "inception_v3": [3, 299, 299],
    "unet": [3,256,256],
    'deeplabv3': [3,256,256],
    'mobilenet_v2': [3,244,244],
    # 'open_unmix': [2,100000],
    'alexnet': [3,244,244],
    'bert': [1024,768],
}

QoS_map = {
    'resnet50': 108,
    'resnet101': 108,
    'resnet152': 108,
    'vgg16':  142,
    'vgg19': 142,
    'mobilenet_v2': 64,
    'unet': 120,
    'bert': 400,
    'deeplabv3': 300,
    'alexnet': 80,
}

max_RPS_map = {
    'resnet50': 2000,
    'resnet101': 1500,
    'resnet152': 1000,
    'vgg16': 1500,
    'vgg19': 1300,
    'mobilenet_v2': 4000,
    'unet': 1300,
    'bert': 250, 
    'deeplabv3': 300,
    'alexnet' : 7000,
}
min_RPS_map = {
    'resnet50': 1,
    'resnet101': 1,
    'resnet152': 1,
    'vgg16': 1,
    'vgg19': 1,
    'mobilenet_v2': 200,
    'unet': 1,
    'bert': 1,
    'deeplabv3': 1,
    'alexnet' : 500,
}

def handle_terminate(signum, frame):
    pass


def get_model(model_name):
    return  model_list.get(model_name)

def get_input(model_name, k):
    input = input_list.get(model_name)
    if model_name == 'bert':
        input = torch.FloatTensor(np.random.rand(k, 1024, 768))
        masks = torch.FloatTensor(np.zeros((k, 1, 1, 1024)))
        return input,masks
    
    if model_name == 'transformer':
        input = torch.randn(512, k, 768)
        masks = torch.ones(512, 512)

        return input,masks
    if len(input) == 3:
        return torch.randn(k, input[0], input[1], input[2])
    else:
        return torch.randn(k, input[0], input[1])


def get_p95(data):
    data = np.array(data)
    percentile_95 = np.percentile(data, 95)
    return percentile_95

def get_p99(data):
    data = np.array(data)
    percentile_99 = np.percentile(data, 99)
    return percentile_99

def get_p98(data):
    data = np.array(data)
    percentile_98 = np.percentile(data, 98)
    return percentile_98


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--RPS", type=str)
    parser.add_argument("--SM", type=str)
    args = parser.parse_args()

    task = args.task
    RPS = int(args.RPS)
    SM = args.SM

    
    QoS = QoS_map.get(task)
    half_QoS = QoS/2
    batch = math.floor(RPS/1000 * half_QoS)


    if task == 'bert':  
        model = get_model(task)
        model = model().half().cuda(0).eval()
    else:
        model = get_model(task)
        model = model().cuda(0).eval()
    
    max_epoch = 1000
    valid_list = []
    with torch.no_grad():
        for i in range(0, max_epoch):
            if task == 'bert':
                input,masks = get_input(task, batch)
                input = input.half()
                masks = masks.half()
                
            elif task == 'transformer':
                input,masks = get_input(task, batch)
            else:
                input = get_input(task, batch)

            start_time = time.time()
            if task == 'bert':
                input = input.cuda(0)
                masks = masks.cuda(0)
            elif task == 'transformer':
                input = input.cuda(0)
                masks = masks.cuda(0)
            else:
                input = input.cuda(0)

            if task == 'bert':
                output= model.run(input,masks,0,12).cpu()
            elif task == 'transformer':

                outputs = model(input, input, src_mask=masks, tgt_mask=masks).cpu()
                
            elif task == 'deeplabv3':
                output= model(input)['out'].cpu()
            else:
                output=model(input).cpu()
            end_time = time.time()

            valid_list.append((end_time - start_time) * 1000)

    filtered_result = valid_list[200:]
    p99 = get_p95(filtered_result)

    data = {
        'task': task,
        'RPS': str(RPS),
        'SM': SM,
        'latency': p99
    }

    file_path = '/data/zbw/inference_system/MIG_MPS/tmp/bayesian_tmp.txt'
    with open(file_path, 'a+') as file:
        file.write(f"task: {data['task']}, RPS: {data['RPS']}, SM: {data['SM']}, latency: {data['latency']}\n")


