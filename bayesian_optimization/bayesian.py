from skopt import gp_minimize
import numpy as np
from bayes_opt import BayesianOptimization
import argparse
import logging
import re
import math

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


def get_configuration_result(configuration_list):
    file_path = '/data/zbw/inference_system/MIG_MPS/log/resnet152_Pairs_MPS_RPS'
    data_list = read_data(file_path)



    for i in range(0, len(data_list)-1, 2):  
        if i + 1 < len(data_list):

            item1 = data_list[i]
            item2 = data_list[i + 1]


           
            QoS = QoS_map.get(task)
            half_QoS = QoS/2

            batch1 = math.floor(float(configuration_list[0]['RPS'])/1000 * half_QoS)
            batch2 = math.floor(float(configuration_list[1]['RPS'])/1000 * half_QoS)

       
            if int(item1['SM']) == int(configuration_list[0]['SM']) and int(item2['SM']) == int(configuration_list[1]['SM']) \
            and int(batch1) == int(item1['batch']) and int(batch2) == int(item2['batch']):
                latency1 = item1['percentile']
                latency2 = item2['percentile']
                return latency1, latency2


def objective(configuration_list):
    result = 0
    latency1, latency2 = get_configuration_result(configuration_list)
    RPS1 = configuration_list[0]['RPS']
    RPS2 = configuration_list[1]['RPS']

    
    return result



def get_task_num(task):
    # 解析字符串，判断需要服务的个数，因为现在只有bert_2，我们将服务直接指定为1
    return 1


def wrapped_objective(**kwargs):
    configuration_list = []
    num_maps = len(kwargs) // 2 

    for i in range(1, num_maps + 1):
        map_data = {
            'SM': kwargs[f'SM{i}'],
            'RPS': kwargs[f'RPS{i}'],
            # 'parallelism': kwargs[f'parallelism{i}']
        }

        configuration_list.append(map_data)

    return objective(configuration_list)


def get_pbounds(num_task):
    pbounds = {'x': (-10, 10)}

    pass

def init_optimizer(num_task):

    pbounds = get_pbounds(num_task)

    # optimizer =BayesianOptimization(
    #     f=objective,
    #     pbounds=pbounds,
    #     random_state=1,
    # )
    
    return optimizer



# 定义参数的边界

# 初始化贝叶斯优化对象


# 执行优化
# optimizer.maximize(
#     init_points=2,  # 初始化步数
#     n_iter=25,      # 迭代步数
# )


# 输出结果
# for i, res in enumerate(optimizer.res):
#     print(f"Iteration {i+1}: x={res['params']['x']}, target={res['target']}")


# 这个版本由于现在数据库的限制，其仅仅能搜索 bert 在model_parrlism为2的情况下的，两个相同的bert模型的rps只和。
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    args = parser.parse_args()

    task = args.task
    serve_num = get_task_num(task)

    wrapped_objective(SM1=50, RPS1=300, SM2=50, RPS2=300)
