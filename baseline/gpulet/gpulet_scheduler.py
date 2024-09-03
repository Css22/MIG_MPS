import threading
import subprocess
import numpy as np
import argparse
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


process_list = []

RPS_tolerate = {

}


knee_point = {
    'bert': 40
}


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
            print(output.strip())
            if "QoS violate" in output:
                logging.info(f"worker {worker_id} with {RPS} QoS violate")

                if worker_id + 1 > len(process_list) or (not process_list[worker_id + 1]['state']):
                    adjust_RPS = process_list[worker_id]['RPS'] - RPS_tolerate.get(process_list[worker_id]['task']) 
                    logging.info(f'update worker {worker_id} due to QoS violate, change {worker_id} and {process_list[worker_id]['task']} RPS from {process_list[worker_id]['RPS']} to {adjust_RPS}')
                    process_list[worker_id]['process'].terminate()
                    run_command(process_list[worker_id]['task'], adjust_RPS, process_list[worker_id]['SM'], worker_id)
                    
                else:
                    adjust_RPS = process_list[worker_id+1]['RPS'] - RPS_tolerate.get(process_list[worker_id+1]['task']) 
                    logging.info(f'update worker {worker_id+1} due to QoS violate, change {worker_id+1} and {process_list[worker_id+1]['task']} RPS from {process_list[worker_id+1]['RPS']} to {adjust_RPS}')
                    process_list[worker_id+1]['process'].terminate()
                    run_command(process_list[worker_id+1]['task'], adjust_RPS, process_list[worker_id+1]['SM'], worker_id+1)




def run_command(task, RPS , SM, worker_id):
    command = ""
    logging.info(f"start command {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    process_list[worker_id]['task'] = task
    process_list[worker_id]['RPS'] = RPS
    process_list[worker_id]['process'] = process
    process_list[worker_id]['state'] = True
    process_list[worker_id]['SM'] = SM
    
    output_thread = threading.Thread(target=stream_output, args=(process, worker_id, RPS))
    output_thread.start()


            
            

def deploy(task, knee, instance_count, remain_sm):

    for i in range(0, instance_count):
        RPS = 0
        run_command(task, RPS, knee , i)
        logging.info("sleep for stable QoS")

    run_command(task, RPS, remain_sm , i)

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
        
        logging.info(f"finish gpulet scheduling and the total RPS is {total_RPS}")
        for i in range(0, instance_count+1):
            logging.info(f"worker {i} with RPS {process_list[i]["RPS"]}")
    # command = "export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-MIG-d82118da-7798-5081-959f-c8bbf24989b3  \
    #         && export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log-MIG-d82118da-7798-5081-959f-c8bbf24989b3 \
    #         && export CUDA_VISIBLE_DEVICES=MIG-d82118da-7798-5081-959f-c8bbf24989b3 \
    #         && cd /data/zbw/inference_system/MIG_MPS/jobs \
    #         && echo set_active_thread_percentage 1986291 40| sudo -E  nvidia-cuda-mps-control && python entry.py  --task bert --RPS 90"

    # run_command(command)