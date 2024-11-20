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
                ]
            print(command)
            process = subprocess.Popen(command, env=custom_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,bufsize=1)
            
            self.processes.append(process)
        
        for process in self.processes:
            process.wait()
            for line in process.stdout:
                print(line, end="")
            for line in process.stderr:
                print(line, end="")
    
    def readFileList(self,historyFile):
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
        print("bound")
        print([min_SM,max_SM,min_RPS,max_RPS])
        return [min_SM,max_SM,min_RPS,max_RPS]

    def mapData(self):
        his_log = [logdir+ self.task[0]+"_"+self.load[0]+"_"+self.task[1]+"_"+self.load[1] +self.GIList[i] for i in range(len(self.GIList))]
        bounding_box = self.readFileList(his_log)
            

if __name__ == "__main__":
    bc = bo_communicator(["resnet101","vgg16"],["250","250"],3,["MIG-df7dfc1c-5a5a-5da5-9ef1-f653e43cb501","MIG-a85ec0ea-ee78-5d2d-a0fe-bd7754779dfb","MIG-32522c13-1a59-5776-9d30-e0ae7b6a4874"])
    bc.start_bo()

    