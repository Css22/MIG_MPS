

import math
import re
import subprocess
import time
import argparse


task = None
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

def read_RPS(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'Config: (\w+), P99: ([\d.]+), RPS: (\d+)', line)
            if match:
                config = int(match.group(1))
                percentile = float(match.group(2))
                RPS = int(match.group(3))
                data.append({"config": config, "RPS": RPS, "percentile": percentile})
    return data

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r'task: (\w+), SM: (\d+), batch: (\d+), 99th percentile: ([\d.]+)', line)
            if match:
                task = match.group(1)
                sm = int(match.group(2))
                batch = int(match.group(3))
                percentile = float(match.group(4))
                data.append({"task": task, "SM": sm, "batch": batch, "percentile": percentile})
    return data

def get_maxRPSInCurSM(serve, sm, halfQoS):


    file_path = '/data/zbw/inference_system/MIG_MPS/log/'+serve+'_MPS_RPS'
    data_list = read_RPS(file_path)
    filtered_data = [item for item in data_list if item['config'] == sm]

    sorted_items = sorted(filtered_data, key=lambda x: x['percentile'])

    max_item = None
    for item in sorted_items:
        if item['percentile'] <= halfQoS:
            max_item = item
        else:
            break

    maxRPS = max_item['RPS']
    return maxRPS

def objective_feedback(SM, RPS):
    result = 0

    remain_SM = 100  - SM

    half_QoS = QoS_map[task]/2
    
    search_SM = (int(remain_SM/10) + 1) * 10
    max_RPS = get_maxRPSInCurSM(task, search_SM, half_QoS)

    batch = math.floor(float(RPS)/1000 * half_QoS)

    print("MPSSERVER!")

    server_id = 3146149

    script_path = '/data/zbw/inference_system/MIG_MPS/micro_experiment/script/padding_feedback.sh'
 
    BO_args= [task, SM, remain_SM, batch, max_RPS, server_id]
    BO_args = [str(item) for item in BO_args]
    process = subprocess.Popen([script_path] + BO_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line, end='')  

    for line in process.stderr:
        print(line, end='') 

    process.wait()

    file_path = '/data/zbw/inference_system/MIG_MPS/tmp/bayesian_tmp.txt'

    latency = None
    valid_RPS = None

    with open(file_path, 'r') as file:
        line = file.readline().strip()

        if line.startswith('latency:'):
            value = float(line.split(':')[1].strip())  # 提取 latency 的值
            latency = float(value)
        
        elif line.startswith('valid_RPS:'):
            value = float(line.split(':')[1].strip()) 
            valid_RPS = int(value)

        else:
            print("no result!")




    with open(file_path, 'w') as file:
        file.write('')


    if latency:
        result = 0.5 * min(1, half_QoS/ latency)
        print(f"result is {result}")
        return result
        
    elif valid_RPS:

        RPS100 = get_maxRPSInCurSM(task, 100, half_QoS)
        result = 0.5 + 0.5/ 2 * (valid_RPS + RPS) / RPS100
        print(f"RPS IS {valid_RPS + RPS} and result is {result}")
        return result
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--SM", type=int)
    parser.add_argument("--RPS", type=int)
    args = parser.parse_args()


    task = args.task
    RPS = args.RPS
    SM = args.SM

    result = objective_feedback(SM=SM, RPS=RPS)
    print(f'task {task} RPS {RPS} SM {SM} and the result {result + RPS}')


