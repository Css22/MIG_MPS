import subprocess
import numpy as np
import argparse
import logging

log_path = '/data/zbw/inference_system/MIG_MPS/log/'

possilble_solution=[[7], [4,3], [4,2,1], [4,1,1,1], [3,3], [3,2,1], [3,1,1,1],[2,2,3], [2,1,1,3], [1,1,2,3], [1,1,1,1,3], [2,2,2,1], [2,1,1,2,1],[1,1,2,2,1],[2,1,1,1,1,1],
                   [1,1,2,1,1,1], [1,1,1,1,2,1], [1,1,1,1,1,1,1]]

RPS_map  = {}

def read_MIG_RPS(task):
    logging.info("start search log file")
    file_path = log_path + f'{task}_MIG_RPS'
    
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
    return max_RPS, i

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    task = args.task

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    RPS_map = read_MIG_RPS(task)
    max_RPS, config = search_solution()

    logging.info(f'PARIS search best config for {task}, best config: {config} and best RPS: {max_RPS}')