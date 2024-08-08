import subprocess
import argparse
import torch
import time
import torch.multiprocessing as mp
import util.util
import torch.nn as nn
import re
import subprocess
start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--percentage", type=int)
args = parser.parse_args()


percentage = args.percentage


output = subprocess.check_output(['nvidia-smi', '-L']).decode('utf-8')

output_lines = output.strip().split('\n')
UUID_list = []
for line in output_lines:
    line = line.strip()

    uuid_match = re.search(r"UUID:\s*([a-zA-Z0-9-]+)", line)
    
UUID = None
if uuid_match:
    uuid = uuid_match.group(1)
else:
    print("UUID not found.")
UUID = uuid
while True:
    command = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && echo get_server_list |  nvidia-cuda-mps-control'
    result = subprocess.check_output(command, shell=True)
    server_id = result.decode().strip()


    command2 = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} &&  echo set_active_thread_percentage {server_id} {percentage} | nvidia-cuda-mps-control'
    result = subprocess.check_output(command2, shell=True)
    result =  result.decode().strip()


    pattern = r'Server (\d+) not found' 
    match = re.search(pattern, result)
    if match:
        print(f"Matched: {match.group(0)}")
        if time.time() - start_time >= 1:
            break
        else:
            continue
    else:
        print('successful')
        break

while True:
    if time.time() - start_time >= 1:
        break




