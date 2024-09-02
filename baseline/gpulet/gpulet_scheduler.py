import threading
import subprocess
import numpy as np
import argparse






QPS_space = {
    'bert': 10
}

knee_point = {
    'bert': 40
}


def get_p99(data):
    data = np.array(data)
    percentile_99 = np.percentile(data, 99)
    return percentile_99


def run_command(command):
    # 使用 subprocess 运行命令
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--RPS", type=int)
    args = parser.parse_args()

    task = args.task
    RPS = args.RPS

    command = "export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-MIG-d82118da-7798-5081-959f-c8bbf24989b3  \
            && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-MIG-d82118da-7798-5081-959f-c8bbf24989b3 \
            && export CUDA_VISIBLE_DEVICES=MIG-d82118da-7798-5081-959f-c8bbf24989b3 \
            && cd /data/zbw/inference_system/MIG_MPS/jobs \
            && echo set_active_thread_percentage 1986291 40| sudo -E  nvidia-cuda-mps-control && python entry.py  --task bert --RPS 90"
    run_command(command)