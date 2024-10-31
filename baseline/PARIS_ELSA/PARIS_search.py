from itertools import product
import subprocess
import numpy as np
import argparse
import logging

LOG_FILE = "/data/zbw/inference_system/MIG_MPS/baseline/PARIS_ELSA/PARIS.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a', filename=LOG_FILE)

request = []
task_list = []
class ConfigData:
    def __init__(self, MIG_config, rps, p99):
        self.MIG_config = str(MIG_config)  # SM值
        self.RPS = int(rps)  # RPS值
        self.p99 = float(p99)  # P99值

    def __repr__(self):
        return f"ConfigData(SM: {self.MIG_config}, RPS: {self.RPS}, P99: {self.p99})"

def parse_line(line):
    parts = line.strip().split(", ")
    MIG_config = parts[0].split(": ")[1]
    P99 = parts[1].split(": ")[1]
    RPS = parts[2].split(": ")[1]
    return ConfigData(MIG_config, RPS, P99)

def read_data_from_file(task, file_path):

    config_list = []
    with open(file_path, 'r') as file:
        for line in file:
            config = parse_line(line)
            config_list.append(config)
    file.close()


    for i in config_map.keys():
        max_RPS = 0
        for j in config_list:
            if j.MIG_config == i:
                if j.RPS > max_RPS:
                    max_RPS = j.RPS
        RPS_map[task][config_map.get(i)] = max_RPS

    

    return config_list


log_path = '/data/zbw/inference_system/MIG_MPS/log/'

possilble_solution=[[7], [4,3], [4,2,1], [4,1,1,1], [3,3], [3,2,1], [3,1,1,1],[2,2,3], [2,1,1,3], [1,1,2,3], [1,1,1,1,3], [2,2,2,1], [2,1,1,2,1],[1,1,2,2,1],[2,1,1,1,1,1],
                   [1,1,2,1,1,1], [1,1,1,1,2,1], [1,1,1,1,1,1,1]]


config_map = {
    '1c-7g-80gb': 7,
    '1c-4g-40gb': 4,
    '1c-3g-40gb': 3,
    '1c-2g-20gb': 2,
    '1c-1g-10gb': 1,
}

RPS_map  = {}

def read_MIG_RPS(task):
    logging.info("start search log file")


    for i in task_list:
        RPS_map[i] = {}
        file_path = log_path + f'{i}_MIG_RPS'
        read_data_from_file(i, file_path)
    logging.info("finish search log file")
    logging.info(f'best MIG partition list: {RPS_map}')
    return RPS_map


def search_solution(serve_num):
    logging.info("start search best config")

    output_RPS = None
    best_RPS = 0
    best_config = None
    best_deployment = None

    if serve_num >= 2:    
        relationship = request[1]/request[0]
        for solution in possilble_solution:
            model_combinations = []
            for value in solution:
                possible_values = [(model_map[value], model_name) for model_name, model_map in RPS_map.items()]
                model_combinations.append(possible_values)

            all_combinations = list(product(*model_combinations))

            for combo in all_combinations:
                total_task_0 = 0
                total_task_1 = 0
                for i in combo:
                    if i[1] == task_list[0]:
                        total_task_0 = total_task_0 + int(i[0])
                    if i[1] == task_list[1]:
                        total_task_1 = total_task_1 + int(i[0])
                
                unite_RPS = min(relationship * total_task_0, total_task_1)
                if best_RPS < unite_RPS:
                    best_RPS = unite_RPS
                    output_RPS = (task_list[0], total_task_0, task_list[1], total_task_1, best_RPS)
                    best_deployment = [i[1] for i in combo] 
                    best_config =  solution

        best_RPS = output_RPS
    else:
        for solution in possilble_solution:
            RPS = 0
            for j in solution:
                RPS = RPS_map[task_list[0]].get(j) + RPS
            
            if RPS > best_RPS:
                best_RPS = RPS
                best_config = solution

    logging.info("finish search best config")

    return best_RPS, best_config, best_deployment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_num", type=int)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    
    task = args.task
    serve_num = args.server_num

    task = [s.strip() for s in task.split(',')]
    for i in range(0, serve_num):
        request.append(int(task[i*2 + 1]))
        task_list.append(task[i*2])

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    RPS_map = read_MIG_RPS(task)
    max_RPS, config, deployment = search_solution(serve_num)

    logging.info(RPS_map)
    logging.info(f'PARIS search best config for {task}, best config: {config}, best deployment: {deployment} and best RPS: {max_RPS}')
    