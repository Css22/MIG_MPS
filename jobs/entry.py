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

path = "/data/zbw/inference_system/MIG_MPS/jobs/"
sys.path.append(path)
flag_path = "/data/zbw/MIG/MIG/MIG_Schedule/flag"
result_path = "/data/zbw/inference_system/MIG_MPS/log/"

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
    'resnet50': 3200
}
min_RPS_map = {
    'resnet50': 2000
}

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

def handle_concurrent_valid_data(valid_list, task, config, batch):

    file_name = result_path + f"{task}_Pairs_MPS_RPS"

    data = np.array(valid_list)
    percentile_99 = np.percentile(data, 99)
    
    with open(file_name, 'a+') as file:
        file.write(f"task: {task}, SM: {config}, batch: {batch}, 99th percentile: {percentile_99}\n")

def get_p95(data):
    data = np.array(data)
    percentile_95 = np.percentile(data, 95)
    return percentile_95

def get_p99(data):
    data = np.array(data)
    percentile_99 = np.percentile(data, 99)
    return percentile_99



def record_result(path, config, RPS ,result):
    filtered_result = result[200:]
    p95 = get_p95(filtered_result)
    with open(path, 'a+') as file:
        file.write(f"Config: {config}, P99: {p95}, RPS: {RPS}\n")
        file.close()

def execute_entry(task, RPS, max_epoch):
    QoS = QoS_map.get(task)
    half_QoS = QoS/2
    batch = math.floor(RPS/1000 * half_QoS)
    valid_list = []

    if task == 'bert':  
        model = get_model(task)
        model = model().half().cuda(0).eval()
    else:
        model = get_model(task)
        model = model().cuda(0).eval()

    with torch.no_grad():
        for i in range(0, max_epoch):
            if task == 'bert':
                input,masks = get_input(task, batch)
            elif task == 'transformer':
                input,masks = get_input(task, batch)
            else:
                input = get_input(task, batch)

            start_time = time.time()
            if task == 'bert':
                input = input.half().cuda(0)
                masks = masks.half().cuda(0)
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
        p95 = get_p95(filtered_result)
        print(p95, half_QoS, RPS)
        if p95 > half_QoS:
            print(task, p95, RPS)
            # record_result(path=file_name, config=config, RPS=RPS, result=valid_list)
            return False
        else:
            record_result(path=file_name, config=config, RPS=RPS, result=valid_list)
            return True
        
def binary_search_max_true(task ,min_RPS, max_RPS, max_epoch):
    left = min_RPS
    right = max_RPS

    while left < right:
        mid = (left + right + 1) // 2
        if execute_entry(task=task, RPS=mid, max_epoch=max_epoch):
            left = mid  # mid可能是解，所以保留在left
        else:
            right = mid - 1  # mid不是解，所以舍弃

    return left  # 最后返回left就是满足条件的最大值
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--concurrent_profile", action='store_true')
    parser.add_argument("--config", default='', type=str)
    parser.add_argument("--file_name", type=str, default='result')
    parser.add_argument("--test", action='store_false')
    parser.add_argument("--RPS", type=int)
    args = parser.parse_args()

    task = args.task
    concurrent_profile = args.concurrent_profile
    config = args.config
    file_name = args.file_name
    test = args.test
    RPS = args.RPS
    batch = args.batch

    max_epoch = 500
    min_RPS = min_RPS_map.get(task)
    max_RPS = max_RPS_map.get(task)
    min_RPS = 100
    max_RPS = 400

    if test:
        print(test)
        QoS = QoS_map.get(task)
        half_QoS = QoS/2
        batch = math.floor(RPS/1000 * half_QoS)
      
        if task == 'bert':  
            model = get_model(task)
            model = model().half().cuda(0).eval()
        else:
            model = get_model(task)
            model = model().cuda(0).eval()


        with torch.no_grad():
            while True:
                valid_list = []
                for i in range(0, 200):
                    if task == 'bert':
                        input,masks = get_input(task, batch)
                    elif task == 'transformer':
                        input,masks = get_input(task, batch)
                    else:
                        input = get_input(task, batch)

                    start_time = time.time()

                    if task == 'bert':
                        input = input.half().cuda(0)
                        masks = masks.half().cuda(0)
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
                print(get_p99(valid_list))  

    elif concurrent_profile:
        if task == 'bert':  
            model = get_model(task)
            model = model().half().cuda(0).eval()
        else:
            model = get_model(task)
            model = model().cuda(0).eval()


        with torch.no_grad():
            valid_list = []
            for i in range(0, 2000):
                if task == 'bert':
                    input,masks = get_input(task, batch)
                elif task == 'transformer':
                    input,masks = get_input(task, batch)
                else:
                    input = get_input(task, batch)

                start_time = time.time()
                
                if task == 'bert':
                    input = input.half().cuda(0)
                    masks = masks.half().cuda(0)
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

            handle_concurrent_valid_data(valid_list[1000:], task, config, batch)

    else:
        binary_search_max_true(task=task, min_RPS=min_RPS, max_RPS=max_RPS, max_epoch=max_epoch)
    
    
    # min_RPS = 100
    # max_RPS = 5000
    # min_RPS = 2000
    # max_RPS = min_RPS
    # if task == 'bert':  
    #     model = get_model(task)
    #     model = model().half().cuda(0).eval()
    # else:
    #     model = get_model(task)
    #     model = model().cuda(0).eval()

    # for i in range(min_RPS, max_RPS+1, 10):
    #     RPS = i
    #     QoS = QoS_map.get(task)
    #     half_QoS = QoS/2
    #     batch = math.floor(RPS/1000 * half_QoS)
    #     valid_list = []
    #     with torch.no_grad():
    #         for j in range(0, max_epoch):
    #             if task == 'bert':
    #                 input,masks = get_input(task, batch)
    #             elif task == 'transformer':
    #                 input,masks = get_input(task, batch)
    #             else:
    #                 input = get_input(task, batch)

    #             start_time = time.time()
    #             if task == 'bert':
    #                 input = input.cuda(0)
    #                 masks = masks.cuda(0)
    #             elif task == 'transformer':
    #                 input = input.cuda(0)
    #                 masks = masks.cuda(0)
    #             else:
    #                 input = input.cuda(0)

    #             if task == 'bert':
    #                 output= model.run(input,masks,0,12).cpu()
    #             elif task == 'transformer':

    #                 outputs = model(input, input, src_mask=masks, tgt_mask=masks).cpu()
                    
    #             elif task == 'deeplabv3':
    #                 output= model(input)['out'].cpu()
    #             else:
    #                 output=model(input).cpu()
    #             end_time = time.time()

    #             valid_list.append((end_time - start_time) * 1000)
    #         filtered_result = valid_list[200:]
    #         p95 = get_p95(filtered_result)
    #         if p95 > half_QoS:
    #             print(task, p95, RPS)
    #             # record_result(path=file_name, config=config, RPS=RPS, result=valid_list)
    #             break
    #         else:
    #             print(p95, half_QoS)
    #             record_result(path=file_name, config=config, RPS=RPS, result=valid_list)
    
    # if concurrent_profile:
    #     if task == 'bert':  
    #         model = get_model(task)
    #         model = model().half().cuda(0).eval()
    #     else:
    #         model = get_model(task)
    #         model = model().cuda(0).eval()

    #     if task == 'bert':
    #         input,masks = get_input(task, batch)
    #     else:
    #         input = get_input(task, batch)
        

    #     valid_list = []
    #     with torch.no_grad():
    #         min_RPS = min_RPS_map.get(task)
    #         max_RPS = max_RPS_map.get(task)
    #         for i in range(min_RPS, max_RPS+1, 10):
    #             for j in range(0, max_epoch):
    #                 if task == 'bert':
    #                     input,masks = get_input(task, batch)
    #                 elif task == 'transformer':
    #                     input,masks = get_input(task, batch)
    #                 else:
    #                     input = get_input(task, batch)

    #                 start_time = time.time()
    #                 if task == 'bert':
    #                     input = input.cuda(0)
    #                     masks = masks.cuda(0)
    #                 elif task == 'transformer':
    #                     input = input.cuda(0)
    #                     masks = masks.cuda(0)
    #                 else:
    #                     input = input.cuda(0)

    #                 if task == 'bert':
    #                     output= model.run(input,masks,0,12).cpu()
    #                 elif task == 'transformer':

    #                     outputs = model(input, input, src_mask=masks, tgt_mask=masks).cpu()
                        
    #                 elif task == 'deeplabv3':
    #                     output= model(input)['out'].cpu()
    #                 else:
    #                     output=model(input).cpu()
    #                 end_time = time.time()
    #                 valid_list.append((end_time - start_time) * 1000)
    #             record_result(path=file_name, config=config, result=valid_list)

    # else:
    #     if task == 'bert':  
    #         model = get_model(task)
    #         model = model().cuda(0).eval()
        
    #     else:
    #         model = get_model(task)
    #         model = model().cuda(0).eval()

      
    #     if task == 'bert':
    #         input,masks = get_input(task, batch)
    #     elif task == 'transformer':
    #         input,masks = get_input(task, batch)
    #     else:
    #         input = get_input(task, batch)
    #     data = []
    #     while True:

    #         with torch.no_grad():
    #             if task == 'bert':
    #                 input,masks = get_input(task, batch)
    #             elif task == 'transformer':
    #                 input,masks = get_input(task, batch)
    #             else:
    #                 input = get_input(task, batch)
    #             start_time = time.time()

    #             if task == 'bert':
    #                 input = input.cuda(0)
    #                 masks = masks.cuda(0)
    #             elif task == 'transformer':
    #                 input = input.cuda(0)
    #                 masks = masks.cuda(0)
    #             else:
    #                 input = input.cuda(0)
    #             if task == 'bert':
    #                 output= model.run(input,masks,0,12).cpu()
    #             elif task == 'transformer':

    #                 outputs = model(input, input, src_mask=masks, tgt_mask=masks).cpu()
                    
    #             elif task == 'deeplabv3':
    #                 output= model(input)['out'].cpu()
    #             else:
    #                 output=model(input).cpu()
    #             end_time = time.time()
    #             data.append((end_time - start_time) * 1000)
    #             if len(data) % 100 == 0:
    #                 print(get_p95(data))
    #                 data = []      

