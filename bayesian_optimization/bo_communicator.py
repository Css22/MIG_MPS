import numpy as np
from skopt import gp_minimize
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
import argparse
import logging
import re
import math
import subprocess
import time
import pickle
from collections import defaultdict
import json
import os


padding_dir = '/data/wyh/MIG_MPS/micro_experiment/script/padding.sh'
paddingFeedback_dir = '/data/wyh/MIG_MPS/micro_experiment/script/padding_feedback.sh'
logdir = '/data/wyh/MIG_MPS/tmp/'

class bo_communicator:
    def __init__(self, task, load, numOfGI, GIList):
        self.task = task
        self.load = load
        self.numOfGI = numOfGI
        self.GIList = GIList
        self.processes = [] 
    
    def start_bo(self):
        arg_task = self.task[0]+","+self.load[0]+","+self.task[1]+","+self.load[1]
        arg_server_num = len(self.task)
        print(arg_task)
        print(arg_server_num)
        for i in range(self.numOfGI):
            custom_env = os.environ.copy()  # 复制当前环境变量
            custom_env["CUDA_MPS_PIPE_DIRECTORY"] = "/tmp/nvidia-mps-"+self.GIList[i]
            custom_env["CUDA_MPS_LOG_DIRECTORY"] = "/tmp/nvidia-log-"+self.GIList[i]
            custom_env["CUDA_VISIBLE_DEVICES"] = self.GIList[i]

            # 定义要运行的 Python 文件和参数
            command = [
                "python", 
                "bayesian_pruning.py", 
                "--task", arg_task, 
                "--server_num", str(arg_server_num), 
                "--feedback", 
                "--device", self.GIList[i],
                "--port", str(12334+i*2),
                "--idxGI", str(i),
                "--numOfGI",str(self.numOfGI)
                ]
            print(command)
            process = subprocess.Popen(command, env=custom_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, text=True, bufsize=1)
            
            self.processes.append(process)
        
        # for process in self.processes:
        #     process.wait()
        #     for line in process.stdout:
        #         print(line, end="")
        #     for line in process.stderr:
        #         print(line, end="")
    
    def foundBox(self,historyFile):
        ConfigList_RPS = []
        ConfigList_SM = []
        for fileName in historyFile:
            with open(fileName, 'r') as file:
                data = json.load(file)
                max_target_entry = max(data, key=lambda x: x['target'])
                RPS_t =  max_target_entry ['params']['RPS0']
                SM_t =  max_target_entry ['params']['SM0']
                ConfigList_RPS.append(RPS_t)
                ConfigList_SM.append(SM_t)
        min_RPS = min(ConfigList_RPS)
        max_RPS = max(ConfigList_RPS)
        min_SM = min(ConfigList_SM)
        max_SM = max(ConfigList_SM)
        return [min_SM,max_SM,min_RPS,max_RPS]
    
    def foundMax(self,historyFile):
        maxPointList = []
        for fileName in historyFile:
            with open(fileName, 'r') as file:
                data = json.load(file)
                max_target_entry = max(data, key=lambda x: x['target'])
                v = max_target_entry ['target']
                RPS_t =  max_target_entry ['params']['RPS0']
                SM_t =  max_target_entry ['params']['SM0']
                maxPointList.append([v,RPS_t,SM_t])
        return maxPointList




    def mapData(self):
        his_log = [logdir+ self.task[0]+","+self.load[0]+","+self.task[1]+","+self.load[1]+ "-" +self.GIList[i]+".json" for i in range(len(self.GIList))]
        bounding_box = self.foundBox(his_log)
        print(bounding_box)
        maxPointList = self.foundMax(his_log)
        print(maxPointList)
        return [bounding_box,maxPointList]
        

    def boardData(self,bounding_box,maxPointList):
        if bounding_box is not None:
            with open(logdir+'map.txt', 'w') as f:
                f.write(','.join([str(num) for num in bounding_box]))
        else:
            with open(logdir+'map.txt', 'w') as f:
                for sublist in maxPointList:
                    f.write(','.join(map(str, sublist)) + '\n')
        
        
            

if __name__ == "__main__":
    bc = bo_communicator(["resnet101","vgg16"],["250","250"],3,["MIG-4b83173e-7c28-59a4-948a-c31604e90796","MIG-d57cc68f-14fd-5d4c-9471-1264dedbc42e","MIG-70a238a8-c394-5645-9e02-09046901dfdc"])
    bc.start_bo()
    print("bo started")
    received_signals = 0
    while received_signals < 3:
        for i, process in enumerate(bc.processes):
            if received_signals==3: break
            line = process.stdout.readline().strip()  # 读取子进程输出
            if line:
                print(line)
                if line == "checkpoint reached":
                    print(f"[Main] Received from Worker-{i}: {line}")
                    received_signals += 1
                    

    print("mapData start")
    # mapRes = bc.mapData()
    # bc.boardData(None,mapRes[1])
    print("mapData over")

    print("[Main] All processes reached checkpoint. Sending continue signals.")
    for process in bc.processes:
        process.stdin.write("continue\n")
        process.stdin.flush()
    
    for process in bc.processes:
        process.wait()
    for line in process.stdout:
        print(line, end="")
    for line in process.stderr:
        print(line, end="")

    