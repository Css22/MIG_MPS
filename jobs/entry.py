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

path = "/data/zbw/inference_system/MIG_MPS/jobs/"
sys.path.append(path)
flag_path = "/data/zbw/MIG/MIG/MIG_Schedule/flag"
result_path = "/data/zbw/inference_system/MIG_MPS/Result/"
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

def get_model(model_name):
    return  model_list.get(model_name)

def get_input(model_name, k):
    input = input_list.get(model_name)
    if model_name == 'bert':
        input = torch.FloatTensor(np.random.rand(k, 1024, 768)).cuda(0)
        masks = torch.FloatTensor(np.zeros((k, 1, 1, 1024))).cuda(0)
        return input,masks
    
    if model_name == 'transformer':
        input = torch.randn(512, k, 768).cuda(0)  # 随机输入数据
        masks = torch.ones(512, 512).cuda(0)  # 注意力掩码

        return input,masks
    if len(input) == 3:
        return torch.randn(k, input[0], input[1], input[2]).cuda(0)
    else:
        return torch.randn(k, input[0], input[1]).cuda(0)

def handle_valid_data(valid_list, jobs, file_name):
    file_name = result_path + file_name
    data = np.array(valid_list)
    percentile_99 = np.percentile(data, 99)
    
    with open(file_name, 'a+') as file:
        file.write(f"Jobs: {jobs}, 99th percentile: {percentile_99}\n")

def get_p99(data):
    data = np.array(data)
    percentile_99 = np.percentile(data, 99)
    return percentile_99


def signal_handler(sig, frame):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--concurrent_profile", default=False, type=bool)
    parser.add_argument("--jobs", default='', type=str)
    parser.add_argument("--file_name", type=str, default='result')
    args = parser.parse_args()
    task = args.task
    batch = args.batch
    concurrent_profile = args.concurrent_profile
    jobs = args.jobs
    file_name = args.file_name
    start_time = time.time()

    if concurrent_profile:
            signal.signal(signal.SIGTERM, signal_handler)
            if task == 'bert':  
                model = get_model(task)
                model = model().half().cuda(0).eval()
            else:
                model = get_model(task)
                model = model().cuda(0).eval()

            if task == 'bert':
                input,masks = get_input(task, batch)
            else:
                input = get_input(task, batch)
            

            valid_list = []
            while True:
                execute_start_time = time.time()
                if task == 'bert':
                    output= model.run(input,masks,0,12).cpu()
                elif task == 'deeplabv3':
                    output= model(input)['out'].cpu()
                else:
                    output=model(input).cpu()
                execute_end_time = time.time()
                current_time = time.time()
                if current_time - start_time <= 30:
                    continue
                elif current_time - start_time <= 30 + 60 * 1:
                    valid_list.append((execute_end_time - execute_start_time) * 1000)
                else:
                    break

            handle_valid_data(valid_list, jobs, file_name)



    else:
        signal.signal(signal.SIGTERM, signal_handler)
        if task == 'bert':  
            model = get_model(task)
            model = model().cuda(0).eval()
        
        else:
            model = get_model(task)
            model = model().cuda(0).eval()

      
        if task == 'bert':
            input,masks = get_input(task, batch)
        elif task == 'transformer':
            input,masks = get_input(task, batch)
        else:
            input = get_input(task, batch)

        data = []
        while True:
            start_time = time.time()
            


            if task == 'bert':
                output= model.run(input,masks,0,12).cpu()
            elif task == 'transformer':

                outputs = model(input, input, src_mask=masks, tgt_mask=masks).cpu()
            elif task == 'deeplabv3':
                output= model(input)['out'].cpu()
            else:
                output=model(input).cpu()
            end_time = time.time()
            data.append((end_time - start_time) * 1000)
            if len(data) % 100 == 0:
                print(get_p99(data))
                data = []      

