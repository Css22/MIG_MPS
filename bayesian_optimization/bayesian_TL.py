import numpy as np
from openbox import Observation, History, Advisor, space as sp, logger
from openbox import Optimizer
import argparse
import logging
import re
import math
import subprocess
import time
from collections import defaultdict
import pickle

padding_dir = '/data/wyh/MIG_MPS/micro_experiment/script/padding.sh'
paddingFeedback_dir = '/data/wyh/MIG_MPS/micro_experiment/script/padding_feedback.sh'
MPS_PID = 531132 
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

optimizer =  None
task =None

transfer_learning_history = list()  

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
    RPS1 = configuration_list['RPS1']
    RPS2 = configuration_list['RPS2']
    SM1 = configuration_list['SM1']
    SM2 = 100-SM1

    tmp_config=[{'RPS':RPS1,'SM':SM1},{'RPS':RPS2,'SM':SM2}]
    # open history log
    tmp = get_configuration_result(tmp_config, args.task)
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

        # print(result.stdout)

        # print(result.stderr)


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

    return -result



def objective_feedback(configuration_list):
    result = 0

    SM = configuration_list['SM']
    RPS = configuration_list['RPS']

    remain_SM = 100  - SM

    half_QoS = QoS_map[task]/2
    
    search_SM = (int(remain_SM/10) + 1) * 10
    max_RPS = get_maxRPSInCurSM(task, search_SM, half_QoS)

    batch = math.floor(float(RPS)/1000 * half_QoS)


    server_id = MPS_PID

    script_path = paddingFeedback_dir
 
    BO_args= [task, SM, remain_SM, batch, max_RPS, server_id]
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

        if line.startswith('latency:'):
            value = float(line.split(':')[1].strip())  # 提取 latency 的值
            latency = float(value)
        
        elif line.startswith('valid_RPS:'):
            value = float(line.split(':')[1].strip()) 
            valid_RPS = int(value)

        else:
            print("no result!")




    with open(file_path, 'w') as file:
        file.write('')


    if latency:
        result = 0.5 * min(1, half_QoS/ latency)
        print(f"result is {result}")
        return result
        
    elif valid_RPS:

        RPS100 = get_maxRPSInCurSM(task, 100, half_QoS)
        result = 0.5 + 0.5/ 2 * (valid_RPS + RPS) / RPS100
        print(f"RPS IS {valid_RPS + RPS} and result is {result}")
        return result
    time.sleep(1)

def get_task_num(task):
    return 1




def wrapped_objective_feedback(SM, RPS):
    print(task)
    configuration_list = []
    configuration_list = [{'SM': int(SM), 'RPS': int(RPS)}] 
    return objective_feedback(configuration_list)

def wrapped_objective(SM1, RPS1, RPS2):

    configuration_list = []

    configuration_list = [{'SM':int(SM1), 'RPS':int(RPS1)},{'SM':100-int(SM1), 'RPS':int(RPS2)}]

    return objective(configuration_list)

space =sp.Space()
SM1= sp.Int("SM1",20,80)
RPS1 = sp.Int("RPS1",300,600) 
RPS2 = sp.Int("RPS2",300,600)
space.add_variables([SM1,RPS1,RPS2])

space_feedback = sp.Space()
SM1_feedback = sp.Int("SM",20,80)
RPS1_feedback = sp.Int("RPS",300,800)
space_feedback.add_variables([SM1_feedback,RPS1_feedback])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--feedback", action='store_true')
    args = parser.parse_args()

    task = args.task
    feedback = args.feedback
    serve_num = get_task_num(task)

    task = [s.strip() for s in task.split(',')]
    #task = task[0]

    # # Generate history data for transfer learning. transfer_learning_history requires a list of History.
    # transfer_learning_history = list()  # type: List[History]
    # # 3 source tasks with 50 evaluations of random configurations each
    # # one task is relevant to the target task, the other two are irrelevant
    # num_history_tasks, num_results_per_task = 5, 20
    # for task_idx in range(num_history_tasks):
    #     # Build a History class for each source task based on previous observations.
    #     # If source tasks are also optimized by Openbox, you can get the History by
    #     # using the APIs from Optimizer or Advisor. E.g., history = advisor.get_history()
    #     history = History(task_id=f'history{task_idx}', config_space=space)

    #     for _ in range(num_results_per_task):
    #         print("task_index={},_={}".format(task_idx,_))
    #         config = space.sample_configuration()
    #         if task_idx == 0:  # relevant task
    #             y = objective(config)+0.01
    #         elif task_idx == 1:  # relevant task
    #             y = objective(config)-0.01
    #         elif task_idx == 2:
    #             y = objective(config) 
    #         elif task_idx == 3:
    #             y = objective(config)* 0.5
    #         else: # irrelevant tasks
    #             y = np.random.random()-1
    #         # build and update observation
    #         observation = Observation(config=config, objectives=[y])
    #         history.update_observation(observation)

    #     transfer_learning_history.append(history)

    # with open('../tmp/my_list_5_20_o.pkl', 'wb') as file:
    #     pickle.dump(transfer_learning_history, file)

    with open('../tmp/my_list_5_20_o.pkl', 'rb') as file:
        TLH = pickle.load(file)

    # Run
    opt = Optimizer(
        objective,
        space,
        max_runs=20,
        surrogate_type='gp',          # try using 'auto'!
        task_id='quick_start',
    )

    opt_feedback = Optimizer(
        objective_feedback,
        space_feedback,
        max_runs=20,
        surrogate_type='gp',          # try using 'auto'!
        task_id='quick_start',
    )

    tlbo_advisor = Advisor(
        config_space=space,
        num_objectives=1,
        num_constraints=0,
        initial_trials=5,
        transfer_learning_history=TLH,  # type: List[History]
        surrogate_type='tlbo_rgpe_gp',
        acq_type='ei',
        acq_optimizer_type='random_scipy',
        task_id='TLBO',
    )

    if not feedback:

        starttime = time.time()
        for i in range(15):
            config = tlbo_advisor.get_suggestion()
            res = objective(config)
            logger.info(f'Iteration {i+1}, result: {res}')
            observation = Observation(config=config, objectives=[res])
            tlbo_advisor.update_observation(observation)

        history = tlbo_advisor.get_history()
        print(history)

        # history = opt.run()
        # print(type(opt.get_history()))
        # print(history)
        print("time is {}ms".format((time.time()-starttime)*1000))
    
    else:
        start = time.time()

        history = opt_feedback.run()
        print(type(opt_feedback.get_history()))
        print(history)
        print("time is {}ms".format((time.time()-start)*1000))

        end = time.time()
        print(end - start)


