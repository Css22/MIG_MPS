from skopt import gp_minimize
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import argparse
import logging
import re
import math
import subprocess
from collections import defaultdict

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

optimizer =  None


# 定义目标函数
# 这里需要解决的问题：
# 1. 如何获取该配置下，在线任务返回的结果
#   1.1 ：先利用系统数据库中的结果直接获取
# 2. 如何利用1中所获得的结果，定义objective的值
#   2.1: 对于相同serve而言：
        # 2.1.1: 需要平滑化整个曲线（这里的意思是对于违反QoS的任务，我们需要对其给一个正确的值，类似之前的映射方式）
        # 2.1.2: 需要限制搜索空间，以减少开销 （这里指的SM, RPS, Model并行。SM, RPS都有资源的上限）
        # 2.1.3: 需要定义一个良好的初始解
#   2.2: 对于不同的serve而言：
        # 2.1.1: 我们需要根据各种需求定义联合优化目标（即评估各种解决方案）
task = 'resnet152'

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

def get_configuration_result(configuration_list, serve):
    print(len(configuration_list))
    print("cur SM and RPS are")
    print(configuration_list[0]['SM'])
    print(configuration_list[0]['RPS'])
    print(configuration_list[1]['SM'])
    print(configuration_list[1]['RPS'])

    if serve_num == 1:
        QoS = QoS_map.get(task[0])
        half_QoS = [QoS/2,QoS/2]
    else:
        QoS1 = QoS_map.get(task[0])
        QoS2 = QoS_map.get(task[1])
        half_QoS = [QoS1/2,QoS2/2]

    batch1 = math.floor(float(configuration_list[0]['RPS'])/1000 * half_QoS[0])
    batch2 = math.floor(float(configuration_list[1]['RPS'])/1000 * half_QoS[1])

    file_path = '/data/zbw/inference_system/MIG_MPS/log/'+serve+'_Pairs_MPS_RPS'
    data_list = read_data(file_path)
    for i in range(0, len(data_list)-1, 2):  
        if i + 1 < len(data_list):

            item1 = data_list[i]
            item2 = data_list[i + 1]

            if int(item1['SM']) == int(configuration_list[0]['SM']) and int(item2['SM']) == int(configuration_list[1]['SM']) \
            and int(batch1) == int(item1['batch']) and int(batch2) == int(item2['batch']):
                latency1 = item1['percentile']
                latency2 = item2['percentile']
                return latency1, latency2
            elif int(item2['SM']) == int(configuration_list[0]['SM']) and int(item1['SM']) == int(configuration_list[1]['SM']) \
            and int(batch2) == int(item1['batch']) and int(batch1) == int(item2['batch']):
                latency1 = item2['percentile']
                latency2 = item1['percentile']
                return latency1, latency2


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




def objective(configuration_list):
    result = 0

    RPS1 = configuration_list[0]['RPS']
    RPS2 = configuration_list[1]['RPS']
    SM1 = configuration_list[0]['SM']
    SM2 = configuration_list[1]['SM']
    tmp = get_configuration_result(configuration_list, args.task)

    print(tmp)
    
    print(RPS1, RPS2, SM1, SM2)

    if tmp is None:
        print("illegal RPS and SM!")
        m1 = task[0]
        m2 = task[0]
        if serve_num == 1:
            QoS = QoS_map.get(task[0])
            half_QoS = [QoS/2,QoS/2]
        else:
            QoS1 = QoS_map.get(task[0])
            QoS2 = QoS_map.get(task[1])
            half_QoS = [QoS1/2,QoS2/2]

        batch1 = math.floor(float(RPS1)/1000 * half_QoS[0])
        batch2 = math.floor(float(RPS2)/1000 * half_QoS[1])
        if serve_num == 2:
            m2 = task[1]
        
        server_id = 4076752
        script_path = '/data/zbw/inference_system/MIG_MPS/micro_experiment/script/padding.sh'

        if SM1 < SM2:

            BO_args= [m1,m2,SM1,SM2,batch1,batch2, server_id]
            BO_args = [str(item) for item in BO_args]
            result = subprocess.run([script_path] + BO_args, capture_output=True, text=True)

        else:

            BO_args= [m2,m1,SM2,SM1,batch2,batch1, server_id]
            BO_args = [str(item) for item in BO_args]
            result = subprocess.run([script_path] + BO_args, capture_output=True, text=True)
        print(result.stdout)

        print(result.stderr)


        file_path = '/data/zbw/inference_system/MIG_MPS/tmp/bayesian_tmp.txt'
        lock_path = file_path + '.lock' 


        with open(file_path, 'r') as file:
            lines = file.readlines()


        for line in lines:
            context = line.strip().split(" ")
            if m1 == context[0] and int(batch1) == int(context[1]) and int(SM1) == int(context[2]):
                latency1 = float(context[3])
            else:
                latency2 = float(context[3])

        with open(file_path, 'w') as file:
            file.write('')
            
    else:
        latency1, latency2 = tmp
    
   

    if serve_num ==1 :
        QoS = QoS_map[task[0]]/2
        if latency1 > QoS or latency2 > QoS:
            result = 0.5 * math.sqrt(min(1,QoS/latency1)*min(1,QoS/latency2))

        else:
            RPS100 = get_maxRPSInCurSM(task[0], 100, QoS)
            result = 0.5 + 0.5/ 2 * (RPS1+RPS2) / RPS100
            print("RPS1+RPS2={}".format(RPS1+RPS2))

    else:
        QoS1 = QoS_map[task[0]]/2
        QoS2 = QoS_map[task[1]]/2
        
        if latency1 > QoS1 or latency2 > QoS2:
            result = 0.5 * math.sqrt(min(1,QoS1/latency1)*min(1,QoS2/latency2))
        else:
            RPS1_alone = get_maxRPSInCurSM(task[0], SM1, QoS1)
            RPS2_alone = get_maxRPSInCurSM(task[1], SM2, QoS2)
            result = 0.5 + 0.5 * math.sqrt(RPS1/RPS1_alone*RPS2/RPS2_alone)


    return result



def get_task_num(task):
    # 解析字符串，判断需要服务的个数，因为现在只有bert_2，我们将服务直接指定为1
    return 1


def wrapped_objective(SM1, RPS1, RPS2):
    configuration_list = []

    configuration_list = [{'SM':int(SM1)*10, 'RPS':int(RPS1)},{'SM':100-int(SM1)*10, 'RPS':int(RPS2)}]

    return objective(configuration_list)


def get_pbounds(num_task):
    pbounds = {'x': (-10, 10)}

    pass

def init_optimizer(num_task):

    pbounds = get_pbounds(num_task)

    optimizer =BayesianOptimization(
        wrapped_objective,{'SM1':(1,9),
            'RPS1':(300,600),
            'RPS2':(300,600)
        },
        random_state = 1
    )
    
    return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    #args.task形如resnet50,resnet101的字符串
    task = args.task
    serve_num = get_task_num(task)
    task = [s.strip() for s in task.split(',')]


    optimizer = init_optimizer(1)

    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    optimizer.maximize(
        init_points=30,  
        n_iter=50,      
        acquisition_function=utility  
    )

    print(optimizer.max)


