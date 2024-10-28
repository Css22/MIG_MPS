import subprocess
import numpy as np
import argparse
import logging

LOG_FILE = "/data/zbw/inference_system/MIG_MPS/baseline/PARIS_ELSA/PARIS.log"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='a', filename=LOG_FILE)

request = []

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

def read_data_from_file(file_path):
    config_list = []
    with open(file_path, 'r') as file:
        for line in file:
            config = parse_line(line)
            config_list.append(config)

    logging.info(f"find valid {len(config_list)} MIG solution")

    logging.info(f"start search best MIG partition")

    for i in config_map.keys():
        max_RPS = 0
        for j in config_list:
            if j.MIG_config == i:
                if j.RPS > max_RPS:
                    max_RPS = j.RPS
        RPS_map[config_map.get(i)] = max_RPS
    logging.info(f"finish search best MIG partition")
    logging.info(f'best MIG partition list: {RPS_map}')
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
    file_path = log_path + f'{task}_MIG_RPS'
    read_data_from_file(file_path)
    logging.info("finish search log file")
    return RPS_map


def search_solution():
    max_RPS = 0
    config = None
    logging.info("start search best config")
    for i in possilble_solution:
        RPS = 0
        for j in i:
            RPS = RPS_map.get(j) + RPS
           
        if RPS > max_RPS:
            max_RPS = RPS
            config = i

    logging.info("finish search best config")
    return max_RPS, config

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

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    RPS_map = read_MIG_RPS(task)
    max_RPS, config = search_solution()

    logging.info(f'PARIS search best config for {task}, best config: {config} and best RPS: {max_RPS}')