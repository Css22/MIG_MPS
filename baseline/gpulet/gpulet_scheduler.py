import threading
import subprocess
import numpy as np
import argparse
import logging
import time
import os
import threading
lock = threading.Lock()


env = os.environ.copy()  # 复制当前环境变量
env["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps-MIG-d82118da-7798-5081-959f-c8bbf24989b3"
env["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log-MIG-d82118da-7798-5081-959f-c8bbf24989b3"
env["CUDA_VISIBLE_DEVICES"] = "MIG-d82118da-7798-5081-959f-c8bbf24989b3"

working_directory = "/data/zbw/inference_system/MIG_MPS/jobs"

LOG_FILE = "/data/zbw/inference_system/MIG_MPS/baseline/gpulet/gptlet.log"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a', filename=LOG_FILE)





model_list = ["resnet50", "resnet101", "resnet152", "vgg19", "vgg16", "unet", "deeplabv3", "mobilenet_v2", "alexnet", "bert"]
file_path = '/data/zbw/inference_system/MIG_MPS/log/'


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


half_QoS_map = {

}


SM_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

for i in QoS_map.keys():
    half_QoS_map[i] = QoS_map[i]/2


class ConfigData:
    def __init__(self, sm, rps, p99):
        self.sm = int(sm)  # SM值
        self.RPS = int(rps)  # RPS值
        self.p99 = float(p99)  # P99值

    def __repr__(self):
        return f"ConfigData(SM: {self.sm}, RPS: {self.RPS}, P99: {self.p99})"

def parse_line(line):
    parts = line.strip().split(", ")
    SM = parts[0].split(": ")[1]
    P99 = parts[1].split(": ")[1]
    RPS = parts[2].split(": ")[1]
    return ConfigData(SM, RPS, P99)

def read_data_from_file(file_path):
    config_list = []
    with open(file_path, 'r') as file:
        for line in file:
            config = parse_line(line)
            config_list.append(config)
    return config_list



def find_max_rps_under_p99(config_list, sm_value, p99_threshold):
    max_rps = None
    for config in config_list:
        if config.sm == sm_value and config.p99 <= p99_threshold:
            if max_rps is None or config.RPS > max_rps:
                max_rps = config.RPS
    return max_rps

model_QPS_list = {}

for i in model_list:
    model_log_path = file_path + f"{i}_MPS_RPS"
    config_list = read_data_from_file(model_log_path)
   

    model_QPS_list[i] = []
    for sm_value in SM_list:
        QPS = find_max_rps_under_p99(config_list, sm_value, half_QoS_map.get(i))
        model_QPS_list[i].append(QPS)


process_list = []

RPS_tolerate = {
    'bert': 10,
    'resnet50': 50,
    'resnet101': 40,
    'resnet152': 30,
    'vgg19': 20,
    'vgg16': 20,
    'unet': 20,
    'deeplabv3':5,
    'mobilenet_v2': 20,
    'alexnet': 30,

}


knee_point = {
    'bert': 40,
    'resnet50': 30,
    'resnet101': 30,
    'resnet152': 50,
    'vgg19': 40,
    'vgg16': 40,
    'unet': 40,
    'deeplabv3':30,
    'mobilenet_v2': 70,
    'alexnet': 100,
}



def SetPercentage(UUID, Percentage):
    cmd = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && echo  get_server_list | sudo -E nvidia-cuda-mps-control'
    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()
    read = str(p.stdout.read().decode())
    server_ID = int(read)

    cmd = f'export CUDA_VISIBLE_DEVICES={UUID} && export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-{UUID} && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-{UUID} && sudo echo set_active_thread_percentage {server_ID} {Percentage} |sudo -E nvidia-cuda-mps-control'

    p = subprocess.Popen([cmd], shell=True, stdout=subprocess.PIPE)
    p.wait()

    logging.info(f"finish MPS {Percentage}")

def get_p99(data):
    data = np.array(data)
    percentile_99 = np.percentile(data, 99)
    return percentile_99


def stream_output(process, worker_id, RPS):
    while True:
        output = process.stdout.readline()
        
        if output == '' and process.poll() is not None:
            break

        if output:     
            with lock:
                logging.info(f"Worker {worker_id} : {output.strip()}") 
                if "QoS violate" in output:
                    logging.info(f"worker {worker_id} with {RPS} QoS violate")

                    if worker_id + 1 > len(process_list) or (not process_list[worker_id + 1]['state']):
                    
                        adjust_RPS = process_list[worker_id]['RPS'] - RPS_tolerate.get(process_list[worker_id]['task']) 
                        if adjust_RPS > 0:
                            logging.info(f"update worker {worker_id} due to QoS violate, change {worker_id} and {process_list[worker_id]['task']} RPS from {process_list[worker_id]['RPS']} to {adjust_RPS}")
                            
                            process_list[worker_id]['process'].terminate()
                            run_command(process_list[worker_id]['task'], adjust_RPS, process_list[worker_id]['SM'], worker_id)
                        else:
                            logging.info(f"due to interfence, close worker {worker_id}")
                            process_list[worker_id]['process'].terminate()
                            
                    else:
                        last_worker = 0
                        for i in range(0, len(process_list)):
                            if process_list[i]['state']:
                                last_worker = i

                        adjust_RPS = process_list[last_worker]['RPS'] - RPS_tolerate.get(process_list[last_worker]['task']) 
                        if adjust_RPS > 0 :
                            logging.info(f"update worker {last_worker} due to QoS violate, change {last_worker} and {process_list[last_worker]['task']} RPS from {process_list[last_worker]['RPS']} to {adjust_RPS}")
                            process_list[last_worker]['process'].terminate()
                            run_command(process_list[last_worker]['task'], adjust_RPS, process_list[last_worker]['SM'], last_worker)
                        else:
                            logging.info(f"due to interfence, close worker {worker_id}")
                            process_list[worker_id]['process'].terminate()




def run_command(task, RPS , SM, worker_id):

    # command = f"export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-MIG-d82118da-7798-5081-959f-c8bbf24989b3  \
    #     && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-MIG-d82118da-7798-5081-959f-c8bbf24989b3 \
    #     && export CUDA_VISIBLE_DEVICES=MIG-d82118da-7798-5081-959f-c8bbf24989b3 \
    #     && cd /data/zbw/inference_system/MIG_MPS/jobs \
    #     && echo set_active_thread_percentage 1677256 {SM}| sudo -E nvidia-cuda-mps-control && python entry.py  --task {task} --RPS {RPS} --gpulet --test"


 
    SetPercentage( "MIG-d82118da-7798-5081-959f-c8bbf24989b3", SM)
    # MPS_command = f"echo set_active_thread_percentage 1677256 {SM}| sudo -E nvidia-cuda-mps-control"
    # logging.info(f"start MPS_command {MPS_command}")
    # MPS_processs =  subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)


    command = ["/home/zbw/anaconda3/envs/Abacus/bin/python", "entry.py", "--task", task, "--RPS", str(RPS), "--gpulet", "--test"]
    logging.info(f"start command {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, cwd=working_directory)

  

  

    process_list[worker_id]['task'] = task
    process_list[worker_id]['RPS'] = RPS
    process_list[worker_id]['process'] = process
    process_list[worker_id]['state'] = True
    process_list[worker_id]['SM'] = SM
    

    # MPS_output_thread = threading.Thread(target=stream_output, args=(MPS_processs, worker_id, RPS))
    # MPS_output_thread.start()

    output_thread = threading.Thread(target=stream_output, args=(process, worker_id, RPS))
    output_thread.daemon = True
    output_thread.start()
    logging.info("sleep for stable QoS")

    time.sleep(360)

            
            

def deploy(task, knee, instance_count, remain_sm):

    for i in range(0, instance_count):
        index = SM_list.index(knee)
        RPS = model_QPS_list.get(task)[index]
        run_command(task, RPS, knee , i)
        


    index = SM_list.index(remain_sm)
    RPS = model_QPS_list.get(task)[index]
    run_command(task, RPS, remain_sm , i+1)

def generate_solution(knee):

    instance_count = int( 100 / knee)
    remain_sm = 100 % knee
    return instance_count, remain_sm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--RPS", type=int)

    args = parser.parse_args()

    task = args.task
    RPS = args.RPS

    if RPS:
        logging.info(f"start gpulets for {task} and {RPS}") 
    else:
        logging.info(f"start gpulets for {task} to find max_RPS") 

        knee = knee_point.get(task)
        instance_count, remain_sm = generate_solution(knee)

        logging.info(f"get the solution from gpulets with {knee} * {instance_count} and one instance with {remain_sm}")

        logging.info("start deploy")

        for i in range(0, instance_count+1):
            process_list.append({})
            process_list[i]['state'] = False


        deploy(task, knee, instance_count, remain_sm)

        total_RPS = 0

        for i in range(0, instance_count+1):
            total_RPS = process_list[i]['RPS'] + total_RPS
        
        logging.info(f"finish gpulet scheduling {task} and the total RPS is {total_RPS}")

        for i in range(0, instance_count+1):
            logging.info(f"worker {i} with RPS {process_list[i]['RPS']}")

        for i in range(0, len(process_list)):
            logging.info(f"close the worker {i}")
            process_list[i]['process'].terminate()
        
        
