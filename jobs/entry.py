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
from filelock import FileLock

import socket
import threading

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
path = "/data/zbw/inference_system/MIG_MPS/jobs/"
sys.path.append(path)
flag_path = "/data/zbw/MIG/MIG/MIG_Schedule/flag"
result_path = "/data/zbw/inference_system/MIG_MPS/log/"

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(42)  

running_tcp_ip = '127.0.0.1'
running_tcp_port = 12345

binary_tcp_ip =  '127.0.0.1'
binary_tcp_port = 12344

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

def handle_concurrent_valid_data(valid_list, task, config, batch):

    file_name = result_path + f"{task}_Pairs_MPS_RPS"

    data = np.array(valid_list)
    percentile_95 = np.percentile(data, 95)
    
    with open(file_name, 'a+') as file:
        file.write(f"task: {task}, SM: {config}, batch: {batch}, 99th percentile: {percentile_95}\n")

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


def record_result(path, config, RPS ,result):
    filtered_result = result[300:]
    p99 = get_p99(filtered_result)
    with open(path, 'a+') as file:
        file.write(f"Config: {config}, P99: {p99}, RPS: {RPS}\n")
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
        # print(p99, half_QoS, RPS)
        if p99 > half_QoS:
            # print(task, p99, RPS)
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
            left = mid  
        else:
            right = mid - 1  

    return left  

def feedback_execute_entry(task, RPS):

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
        for i in range(0, 50):
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

            if i == 0 :
                send_tcp_message(running_tcp_ip, running_tcp_port, 'start')
            else:
                valid_list.append((end_time - start_time) * 1000)
        

        send_tcp_message(running_tcp_ip, running_tcp_port, 'finish')

        data = np.array(valid_list[5:])
        percentile_95 = np.percentile(data, 95)
        time.sleep(1)

        if percentile_95 > half_QoS or float(tcp_control.get_latency()) > half_QoS:
            # print(f"batch is {batch} latency: {percentile_95}, remote_latency: {float(tcp_control.get_latency())}" )
            return False
        else:
            # print(f"batch is {batch} latency: {percentile_95}, remote_latency: {float(tcp_control.get_latency())}" )
            return True

def feedback_search_max_true(task, RPS):
    min_RPS = 1
    max_RPS = RPS

    right = max_RPS
    left = min_RPS

    while left < right:
        mid = (left + right + 1) // 2
        if feedback_execute_entry(task=task, RPS=mid):
            left = mid  
        else:
            right = mid - 1  

    return left 



class TCPControl:
    def __init__(self):
        self.state = None
        self.latency = None

    def set_state(self, state):
        self.state = state
        # print(f"set state {state}")

    def set_latency(self, latency):
        self.latency = float(latency)
        # print(f"set latency {latency}")

    def get_state(self):
        return self.state

    def get_latency(self):
        return self.latency
    
    def reset_state(self):
        self.state = None
        self.latency = None



def tcp_server(host, port, control):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    # print(f"Server is listening on {host}:{port}")

    while True:
        client_socket, addr = server_socket.accept()
        message = client_socket.recv(1024).decode().strip()
        if message in ['start', 'finish']:
            control.set_state(message)

        elif message == 'succeed':
            # print("Received 'succeed', shutting down server.")
            control.set_state(message)
            client_socket.close() 
            break
        else:
            control.set_latency(message)
        client_socket.close()

    server_socket.close()
    # print("Server socket has been closed.")

tcp_control = TCPControl()

def start_server(host, port):
    server_thread = threading.Thread(target=tcp_server, args=(host, port, tcp_control))
    server_thread.start()



def send_tcp_message(host, port, message):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            # print(f"send message to {host} {port} {message}")
            sock.sendall(message.encode())
    except Exception as e:
        print(f"Error sending TCP message: {e}")






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--config", default='', type=str)
    parser.add_argument("--file_name", type=str, default='result')
    parser.add_argument("--RPS", type=int)
    parser.add_argument("--test", action='store_false')
    parser.add_argument("--concurrent_profile", action='store_true')
    parser.add_argument("--gpulet", action='store_true')
    parser.add_argument("--worker_id", type=int)
    parser.add_argument("--bayes", action='store_true')
    parser.add_argument("--feedback", action='store_true')
    parser.add_argument("--running", action="store_true")
    args = parser.parse_args()

    task = args.task
    concurrent_profile = args.concurrent_profile
    config = args.config
    file_name = args.file_name
    test = args.test
    RPS = args.RPS
    batch = args.batch
    gpulet = args.gpulet
    bayes = args.bayes
    feedback = args.feedback
    running = args.running


    max_epoch = 1000
    min_RPS = min_RPS_map.get(task)
    max_RPS = max_RPS_map.get(task)

    if test:
        QoS = QoS_map.get(task)
        half_QoS = QoS/2
        if batch:
            pass
        else:

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
                for i in range(0, 100):
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
                    print((end_time - start_time) * 1000)
                    valid_list.append((end_time - start_time) * 1000)

                print("P99: ", get_p95(valid_list))

              

    elif concurrent_profile:  

        if task == 'bert':  
            model = get_model(task)
            model = model().half().cuda(0).eval()
        else:
            model = get_model(task)
            model = model().cuda(0).eval()

        print("finish memory")

        if (not bayes) or (not feedback):
          
            with torch.no_grad():
                valid_list = []
                
                for i in range(0, 500):
                    

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

                    if not bayes:
                        handle_concurrent_valid_data(valid_list[200:], task, config, batch)

                    else:
                        data = np.array(valid_list[200:])
                        percentile_95 = np.percentile(data, 95)
                        file_path = '/data/zbw/inference_system/MIG_MPS/tmp/bayesian_tmp.txt'
                        lock_path = file_path + '.lock'  

        
                        lock = FileLock(lock_path)

                        with lock:
                            with open(file_path, 'a+') as file:
                                file.write(f"{task} {batch} {config} {percentile_95}\n")
            
        elif running:
            QoS = QoS_map.get(task)
            half_QoS = QoS/2
            if task == 'bert':  
                model = get_model(task)
                model = model().half().cuda(0).eval()
            else:
                model = get_model(task)
                model = model().cuda(0).eval()
            
            start_server(host=running_tcp_ip, port=running_tcp_port)

            tmp_list = []

            for i in range(0, 50):
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
                tmp_list.append((end_time - start_time) * 1000)

            
            data = np.array(tmp_list[5:])
            percentile_95 = np.percentile(data, 95)
            file_path = '/data/zbw/inference_system/MIG_MPS/tmp/bayesian_tmp.txt'

            lock_path = file_path + '.lock'  
            lock = FileLock(lock_path)

            with lock:
                with open(file_path, 'a+') as file:
                    file.write(f"latency: {percentile_95}\n")
                    
            if percentile_95 <= half_QoS:
                with torch.no_grad():
                    valid_list = []
                    while True:
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
                        

                        if tcp_control.get_state() == 'start':
                            valid_list.append((end_time - start_time) * 1000)



                        if tcp_control.get_state() == 'finish':
                            data = np.array(valid_list[5:])
                            if len(data) == 0:
                                percentile_95 = half_QoS + 1
                            else:
                                percentile_95 = np.percentile(data, 95)
                            send_tcp_message(host=binary_tcp_ip, port=binary_tcp_port, message=str(percentile_95))
                            valid_list = []
                            tcp_control.reset_state()  

                        if tcp_control.get_state() == 'succeed':
                            
                            send_tcp_message(host=binary_tcp_ip, port=binary_tcp_port, message='succeed')

                            break
            else:
                send_tcp_message(host=running_tcp_ip, port=running_tcp_port, message='succeed')
                    
        else:

            start_server(host=binary_tcp_ip, port=binary_tcp_port)
            file_path = '/data/zbw/inference_system/MIG_MPS/tmp/bayesian_tmp.txt'
            latency = None

            with open(file_path, 'r') as file:
                line = file.readline().strip()

                if line.startswith('latency:'):
                    value = float(line.split(':')[1].strip())  # 提取 latency 的值
                    latency = float(value)

                else:
                    print("error!")

            QoS = QoS_map.get(task)
            half_QoS = QoS/2

            if float(latency) < half_QoS:
                vaild_RPS = feedback_search_max_true(task=task, RPS=RPS)

                send_tcp_message(host=running_tcp_ip, port=running_tcp_port, message='succeed')

                print(f"find largest valid RPS: {vaild_RPS}" )
                file_path = '/data/zbw/inference_system/MIG_MPS/tmp/bayesian_tmp.txt'
                lock_path = file_path + '.lock'  


                lock = FileLock(lock_path)

                with lock:
                    with open(file_path, 'w') as file:
                        file.write(f"valid_RPS: {vaild_RPS}\n")
            else:
                send_tcp_message(host=binary_tcp_ip, port=binary_tcp_port, message='succeed')
            


                
    elif gpulet:

        QoS = QoS_map.get(task)
        half_QoS = QoS/2
        print(f"start gpulet worker for {task} {RPS}", flush=True)
        if batch:
            pass
        else:

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

                p99 = get_p99(valid_list[10:])
                if p99 > half_QoS:
                    print(f"{task} {RPS} QoS violate", flush=True)
                    # logging.info(f"{task} {RPS} QoS violate")
            
                else:
                    print(f"{task} {RPS} {p99} {half_QoS}", flush=True)
                    # logging.info(f"{task} {RPS} {p99} {half_QoS}")

       


    else:
        binary_search_max_true(task=task, min_RPS=min_RPS, max_RPS=max_RPS, max_epoch=max_epoch)
    
    