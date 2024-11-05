import numpy as np
from skopt import gp_minimize
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import argparse
import logging
import re
import math
import subprocess
import time
import pickle
from collections import defaultdict

padding_dir = '/data/wyh/MIG_MPS/micro_experiment/script/padding.sh'
paddingFeedback_dir = '/data/wyh/MIG_MPS/micro_experiment/script/padding_feedback.sh'
MPS_PID = 2272851
logdir = '/data/wyh/MIG_MPS/tmp/bayesian_tmp.txt'

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

max_RPS_map = {'resnet50': 1500, 'resnet152': 1100, 'vgg16':1300, 'bert': 200, 'mobilenet_v2': 3200}
min_RPS_map = {'resnet50': 500, 'resnet152': 200, 'vgg16': 150, 'bert': 40, "mobilenet_v2": 600}
SM_map = {'resnet50': (30, 90), 'resnet152': (10, 90), 'vgg16': (10, 90), 'bert': (10, 90), "mobilenet_v2": (10,50)}



optimizer =  None
task = None
request = []
test = False

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
    # print(len(configuration_list))
    # print("cur SM and RPS are")
    # print(configuration_list[0]['SM'])
    # print(configuration_list[0]['RPS'])
    # print(configuration_list[1]['SM'])
    # print(configuration_list[1]['RPS'])
    # print()
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

    # open history log
    tmp = get_configuration_result(configuration_list, args.task)
    #tmp = None

    if tmp is None:
        
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
        
        print(f"start executor, the task {m1} {SM1} {batch1} {m2} {SM2} {batch2}")
        server_id = MPS_PID
        script_path = padding_dir

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


        file_path = logdir
        lock_path = file_path + '.lock' 


        with open(file_path, 'r') as file:
            lines = file.readlines()
            print("content of tmp.txt")
            print(lines)
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
            print(f"QoS Violation latency1: {latency1} latency2: {latency2}")

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


def map_to_range(x, min_val, max_val):
    return 0.5 + ((x - min_val) / (max_val - min_val)) * (1 - 0.5)


def objective_feedback(configuration_list):
    #暂时只支持两个任务的feedback，未来可以考虑进行进一步的扩展

    num = len(configuration_list)

    result = 0
    task1 = task[0]
    task2 = task[2]
    
    SM = configuration_list[0]['SM']
    RPS = configuration_list[0]['RPS']

    remain_SM = 100  - SM

    half_QoS = QoS_map[task1]/2
    half_QoS2 = QoS_map[task2]/2

    search_SM = (int(remain_SM/10) + 1) * 10
    max_RPS = get_maxRPSInCurSM(task2, search_SM, half_QoS2)

    batch = math.floor(float(RPS)/1000 * half_QoS)


    server_id = MPS_PID

    script_path = paddingFeedback_dir
    
    BO_args= [task1, task2, SM, remain_SM, batch, max_RPS, server_id]
    BO_args = [str(item) for item in BO_args]
    process = subprocess.Popen([script_path] + BO_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    for line in process.stdout:
        print(line, end='')  

    for line in process.stderr:
        print(line, end='') 

    process.wait()

    file_path = logdir

    latency = None
    valid_RPS = None

    with open(file_path, 'r') as file:
        line = file.readline().strip()
        match = re.search(r"model:\s*(\S+)\s+latency:\s*([\d.]+)", line)

        if match:
            model = match.group(1)
            latency = float(match.group(2))
        # if line.startswith('latency:'):
        #     value = float(line.split(':')[1].strip())  # 提取 latency 的值
        #     latency = float(value)
        
        elif line.startswith('valid_RPS:'):
            value = float(line.split(':')[1].strip()) 
            valid_RPS = int(value)

        else:
            print("no result!")




    with open(file_path, 'w') as file:
        file.write('')

    # latency = 1
    # valid_RPS = 300

    if latency:

        result = 0.5 * min(1, half_QoS/ latency)
        print(f"result is {result}")
        return result
        
    elif valid_RPS:
        # half_QoS2 = QoS_map[task2]/2
        # RPS100 = get_maxRPSInCurSM(task2, 100, half_QoS2)
        # result = 0.5 + 0.5/ 2 * (valid_RPS + RPS) / RPS100
        # 这里需要计算weight权重（可能会涉及到和总global的通讯，以及分布式bayes优化）
        weight0 = 1
        weight1 = 1
        #
        if not test:
            
            result = RPS/request[0] * weight0 + valid_RPS/request[1] * weight1
            
            mapped_value = map_to_range(result, 0, num)
            print(f"RPS IS {valid_RPS + RPS} and result is {mapped_value}")
            return mapped_value
        
        else:   
            half_QoS2 = QoS_map[task2]/2
            # RPS100_1 = get_maxRPSInCurSM(task1, 100 ,half_QoS)
            RPS100_2 = get_maxRPSInCurSM(task2, 100, half_QoS2)
            RPS100_1 = get_maxRPSInCurSM(task1, 100, half_QoS)

            RPS100_baseline = min(RPS100_1, RPS100_2)
            relationship = request[1]/request[0]
            unite_RPS = min(RPS * relationship, valid_RPS)

        
            result = 0.5 + 0.5 * (unite_RPS/RPS100_baseline)

            print(f"RPS IS {RPS} and {valid_RPS} , {unite_RPS} and result is {result}")

            return result
    time.sleep(1)


def get_task_num(task):
    return 1

def wrapped_objective_feedback(**kwargs):
    configuration_list = []

    for i in range(len(kwargs) // 2):
        sm_key = f'SM{i}'
        rps_key = f'RPS{i}'

        if sm_key in kwargs and rps_key in kwargs:
            sm_value = int(kwargs[sm_key])
            rps_value = int(kwargs[rps_key]) 
            configuration_list.append({'SM': sm_value, 'RPS': rps_value})

    return objective_feedback(configuration_list)


# def wrapped_objective_feedback(SM, RPS):
#     configuration_list = []
#     configuration_list = [{'SM': int(SM), 'RPS': int(RPS)}] 
#     return objective_feedback(configuration_list)

def wrapped_objective(SM1, RPS1, RPS2):

    configuration_list = []

    configuration_list = [{'SM':int(SM1), 'RPS':int(RPS1)},{'SM':100-int(SM1), 'RPS':int(RPS2)}]


    return objective(configuration_list)



def init_optimizer(num_task, task):

    

    optimizer =BayesianOptimization(
        wrapped_objective,{'SM1':(10,90),
            'RPS1':(300,600),
            'RPS2':(300,600)
        },
        random_state = 1
    )
    
    return optimizer


def init_optimizer_feedback(server_num, config):

    search_list = {}
    for i in range(0, server_num):
        if i != serve_num - 1: 
            search_list[f'SM{i}'] = SM_map.get(config[i*2])

            max_RPS = max_RPS_map.get(config[i*2])
            min_RPS = min_RPS_map.get(config[i*2])

            if int(config[i*2+1]) < max_RPS_map.get(config[i*2]):
                max_RPS = int(config[i*2+1])
            
            if int(config[i*2+1]) < min_RPS_map.get(config[i*2]):
                min_RPS = 0

            search_list[f'RPS{i}'] = (min_RPS, max_RPS)

        else:
            continue

    print(search_list)

    optimizer =BayesianOptimization(
        wrapped_objective_feedback,
        search_list,
    )
    
    return optimizer


def pruningByHistory(historyFile):
    #{'target': 1.0270000000000001, 'params': {'RPS0': 314.36846711090175, 'SM0': 33.244359265825736}}
    #{'target': 1.042, 'params': {'RPS0': 698.0847392884298, 'SM0': 66.32660892016011}}
    ConfigList_RPS = []
    ConfigList_SM = []
    for fileName in historyFile:
        with open(fileName, 'rb') as file:
            max_Config = pickle.load(file)
            print(max_Config)
            RPS_t = max_Config['params']['RPS0']
            SM_t = max_Config['params']['SM0']
            ConfigList_RPS.append(RPS_t)
            ConfigList_SM.append(SM_t)
    min_RPS = min(ConfigList_RPS)
    max_RPS = max(ConfigList_RPS)
    min_SM = min(ConfigList_SM)
    max_SM = max(ConfigList_SM)
    return [min_SM,max_SM,min_RPS,max_RPS]



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_num", type=int)
    parser.add_argument("--task", type=str)
    parser.add_argument("--feedback", action='store_true')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()


    test = args.test
    serve_num = args.server_num
    task = args.task
    feedback = args.feedback

    task = [s.strip() for s in task.split(',')]

    for i in range(0, serve_num):
        request.append(int(task[i*2 + 1]))

    print(task)
    print(request)

    if not feedback:
        pass

    else:
        start = time.time()
        optimizer = init_optimizer_feedback(serve_num, task)
        utility = UtilityFunction(kind="ei", kappa=5, xi=0.2)
        
        # file_prefix = '../tmp/resnet152_1000_resnet152_1000_'
        # relatedTaskNum = 4
        # file_list = [file_prefix+str(i)+'.pkl' for i in range(2,relatedTaskNum+1)]
        # print(file_list)
        # his = pruningByHistory(file_list)
        # optimizer.set_bounds(new_bounds={"SM0":(his[0],his[1]),"RPS0":(his[2],his[3])})

        optimizer.maximize(
            init_points=5,  
            n_iter=5,      
            acquisition_function=utility  
        )

        print(optimizer.max)
        with open('../tmp/resnet152_1000_vgg16_1000_res_1.pkl', 'wb') as file:
            pickle.dump(optimizer.max, file)

        # with open('./tmp/my_list_5_20_o.pkl', 'rb') as file:
        #     TLH = pickle.load(file)
        end = time.time()
        print(end - start)

    # if not feedback:

    #     optimizer = init_optimizer(1)

    #     utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    #     optimizer.maximize(
    #         init_points=30,  
    #         n_iter=50,      
    #         acquisition_function=utility  
    #     )

    #     print(optimizer.max)
    
    # else:
    #     start = time.time()
    #     optimizer = init_optimizer_feedback()
    #     utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    #     optimizer.maximize(
    #         init_points=5,  
    #         n_iter=10,      
    #         acquisition_function=utility  
    #     )

    #     print(optimizer.max)
    #     end = time.time()
    #     print(end - start)


