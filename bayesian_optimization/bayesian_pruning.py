import numpy as np
from skopt import gp_minimize
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import SequentialDomainReductionTransformer
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
import sys
from scipy.stats import linregress

padding_dir = '/data/wyh/MIG_MPS/micro_experiment/script/padding.sh'
paddingFeedback_dir = '/data/wyh/MIG_MPS/micro_experiment/script/padding_feedback.sh'
MPS_PID = 354993
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

max_RPS_map = {'resnet50': 1500, 'resnet152': 1100, 'vgg16':1300, 'bert': 200, 'mobilenet_v2': 3200,'resnet101':1300}
min_RPS_map = {'resnet50': 500, 'resnet152': 200, 'vgg16': 150, 'bert': 40, "mobilenet_v2": 600,'resnet101':150}
SM_map = {'resnet50': (30, 90), 'resnet152': (10, 90), 'vgg16': (10, 90), 'bert': (10, 90), "mobilenet_v2": (10,50),'resnet101':(10,90)}



optimizer =  None
task = None
request = []
test = False

dict_list = []
changeFlag = False
changeParams = None
changeRes = 0
slope = 0
intercept = 0

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
        
        #print(f"start executor, the task {m1} {SM1} {batch1} {m2} {SM2} {batch2}")
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

        #print(result.stdout)

        #print(result.stderr)


        file_path = logdir
        lock_path = file_path + '.lock' 


        with open(file_path, 'r') as file:
            lines = file.readlines()
            #print("content of tmp.txt")
            #print(lines)
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
            #print(f"QoS Violation latency1: {latency1} latency2: {latency2}")

        else:
            RPS100 = get_maxRPSInCurSM(task[0], 100, QoS)
            result = 0.5 + 0.5/ 2 * (RPS1+RPS2) / RPS100
            #print("RPS1+RPS2={}".format(RPS1+RPS2))

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
    global slope
    global intercept
    if RPS > SM * slope + intercept:
        return 0

    remain_SM = 100  - SM

    half_QoS = QoS_map[task1]/2
    half_QoS2 = QoS_map[task2]/2

    search_SM = (int(remain_SM/10) + 1) * 10
    max_RPS = get_maxRPSInCurSM(task2, search_SM, half_QoS2)

    batch = math.floor(float(RPS)/1000 * half_QoS)

    global dict_list
    global changeFlag
    global changeRes
    if changeFlag:
        if changeParams[2]==0:
                changeRes = 3
                return 3
        else:
            changeRes = 0.5 * min(1, half_QoS/ changeParams[2])
            return 0.5 * min(1, half_QoS/ changeParams[2])

    server_id = MPS_PID


    script_path = paddingFeedback_dir
    
    BO_args= [task1, task2, SM, remain_SM, batch, max_RPS, server_id, args.device, args.port]
    BO_args = [str(item) for item in BO_args]

    process = subprocess.Popen([script_path] + BO_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # for line in process.stdout:
    #     print(line, end='')  

    # for line in process.stderr:
    #     print(line, end='') 

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

    stepDir = dict()
    tmpParams = dict()
    tmpParams["RPS0"] = RPS
    tmpParams["RPS1"] = valid_RPS if not latency else 0
    tmpParams["SM0"] = SM
    tmpParams["SM1"] = remain_SM
    tmpParams["lc0"] = latency if latency else 0
    stepDir["params"] = tmpParams

    if latency:

        result = 0.5 * min(1, half_QoS/ latency)
        #print(f"result is {result}")
        stepDir["target"] = result
        dict_list.append(stepDir)
        return result
        
    elif valid_RPS:
        # half_QoS2 = QoS_map[task2]/2
        # RPS100 = get_maxRPSInCurSM(task2, 100, half_QoS2)
        # result = 0.5 + 0.5/ 2 * (valid_RPS + RPS) / RPS100
        # 这里需要计算weight权重（可能会涉及到和总global的通讯，以及分布式bayes优化）
        weight0 = 1
        weight1 = 1
        if not test:
            
            result = 0.5+ 0.5*(RPS/request[0] * weight0 + valid_RPS/request[1] * weight1)
            #优化的初始阶段，要求两个任务的RPS和尽量的大

            #mapped_value = map_to_range(result, 0, num)
            mapped_value = result
            #print(f"RPS IS {valid_RPS + RPS} and result is {mapped_value}")
            stepDir["target"] = mapped_value
            dict_list.append(stepDir)
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

            #print(f"RPS IS {RPS} and {valid_RPS} , {unite_RPS} and result is {result}")
            stepDir["target"] = result
            dict_list.append(stepDir)
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
            # 这是之前配置完全相同的优化器并行搜索的代码，已废弃
            # SM_box = SM_map.get(config[i*2])
            # interval = (SM_box[1]-SM_box[0])/numOfGI
            # search_list[f'SM{i}'] = (SM_box[0]+idxGI*interval,SM_box[0]+(idxGI+1)*interval)

            search_list[f'SM{i}'] = SM_map.get(config[i*2])

            max_RPS = max_RPS_map.get(config[i*2])
            min_RPS = min_RPS_map.get(config[i*2])
            
            if int(config[i*2+1]) < max_RPS_map.get(config[i*2]):
                max_RPS = int(config[i*2+1])
            
            if int(config[i*2+1]) < min_RPS_map.get(config[i*2]):
                min_RPS = 0

            search_list[f'RPS{i}'] = (min_RPS, max_RPS)
            print(search_list)
        else:
            continue


    bounds_transformer = SequentialDomainReductionTransformer(minimum_window=10)
    optimizer =BayesianOptimization(
        f=wrapped_objective_feedback,
        pbounds=search_list,
        verbose = 2,
        #bounds_transformer=bounds_transformer
    )
    
    return optimizer


def pruningByHistory(historyFile):
    ConfigList_RPS = []
    ConfigList_SM = []
    for fileName in historyFile:
        with open(fileName, 'r') as file:
            data = json.load(file)
            max_target_entry = max(data, key=lambda x: x['target'])
            print(max_target_entry)
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

def changeFileFormat(filepath):
    # 打开原始文件进行处理
    with open(filepath, 'r') as file:
        # 读取文件内容并分割每一行
        lines = file.readlines()

    # 去除每行的换行符并添加逗号
    formatted_data = '[\n' + ',\n'.join(line.strip() for line in lines) + '\n]'

    # 保存成JSON格式文件
    with open(filepath, 'w') as json_file:
        json_file.write(formatted_data)




def start_mps_daemon(gpu_id):
    """
    启动 MPS 守护进程,并返回其 PID
    """
    pipe_dir = f"/tmp/nvidia-mps-{gpu_id}"
    log_dir = f"/tmp/nvidia-log-{gpu_id}"

    # 设置环境变量
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = log_dir
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 创建 MPS 守护进程
    process = subprocess.Popen(
        ["nvidia-cuda-mps-control", "-d"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print(f"MPS Daemon started for GPU {gpu_id}. Waiting for initialization...")
    time.sleep(1)  # 等待守护进程初始化

    # 获取 MPS 守护进程的 PID
    ps_output = subprocess.run(["ps", "-ef"], stdout=subprocess.PIPE, text=True).stdout
    for line in ps_output.splitlines():
        if "nvidia-cuda-mps-control" in line:
            parts = line.split()
            pid = parts[1]
            # 检查环境变量
            with open(f"/proc/{pid}/environ", "r") as env_file:
                environ = env_file.read()
                if pipe_dir in environ:
                    print(f"Found MPS Daemon PID for {pipe_dir}: {pid}")
                    return pid


def run_simple_cuda_program(cuda_script_path, pipe_dir):
    """
    运行一个简单的 CUDA 程序
    """
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = pipe_dir

    try:
        # 调用简单的 Python CUDA 脚本
        process = subprocess.run(
            ["python", cuda_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        print("CUDA Program Output:")
        print(process.stdout)
        if process.stderr:
            print("CUDA Program Errors:")
            print(process.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error while running CUDA program: {e.stderr}")


def get_mps_server_pid(mps_pid):
    """
    根据 MPS 守护进程的 PID,获取对应 MPS Server 的 PID
    """
    ps_output = subprocess.run(["ps", "-ef"], stdout=subprocess.PIPE, text=True).stdout
    for line in ps_output.splitlines():
        if "nvidia-cuda-mps-server" in line and f" {mps_pid} " in line:
            parts = line.split()
            server_pid = parts[1]
            print(f"Found MPS Server PID for MPS Daemon {mps_pid}: {server_pid}")
            return server_pid

    print(f"Error: Could not find MPS Server PID for MPS Daemon {mps_pid}.")
    return None

def changeSurface(bo):
    new_dict_list = []
    global dict_list
    global changeParams 
    global changeRes
    for dic in dict_list:
        dic_copy = dic
        changeParams = [dic["params"]["RPS0"],dic["params"]["RPS1"],dic["params"]["lc0"]]
        bo.probe(
            params={"SM0": dic["params"]["SM0"], "RPS0":dic["params"]["RPS0"]},
            lazy=True,
        )
        bo.maximize(init_points=0, n_iter=0)
        dic_copy["target"]=changeRes

        new_dict_list.append(dic_copy)
    dict_list = new_dict_list



def staticPruning():
    task0 = task[0]
    # MIG2SM的值待定，需要通过MISO的方法来估计
    MIG2SM=max(10,35)
    SM_idx = MIG2SM//10+1
    # 读取 JSON 文件
    with open("/data/wyh/MIG_MPS/log/SM2RPS.json", "r") as file:
        data = json.load(file)

    # 打印内容
    SM2RPS = data[task0][:SM_idx]
    print(SM2RPS)
    SM2RPS = [0]+SM2RPS
    print(SM2RPS)
    x_axis = [i*10 for i in range(len(SM2RPS))]
    print(x_axis)

    x = np.array(x_axis,dtype=float)
    y = np.array(SM2RPS,dtype=float)

    s, i, r_value, p_value, std_err = linregress(x, y)
    print("res is y={}x+{}".format(s,i))
    global slope
    global intercept
    slope = s
    intercept = i
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_num", type=int)
    parser.add_argument("--task", type=str)
    parser.add_argument("--feedback", action='store_true')
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--device",type = str)
    parser.add_argument("--port",type = int)
    parser.add_argument("--idxGI",type = int, default=0)
    parser.add_argument("--numOfGI",type =int,default=1)
    args = parser.parse_args()


    test = args.test
    serve_num = args.server_num
    task = args.task
    feedback = args.feedback
    logdir = logdir.replace('.txt', f'_{args.device}.txt')
    idxGI = int(args.idxGI)
    numOfGI = int(args.numOfGI)
    changeFlag = False
    changeParams = None
    changeRes = 0
    task = [s.strip() for s in task.split(',')]

    #创建这个优化器的mps守护进程，mps_server
    mps_pid = start_mps_daemon(args.device)
    server_id = None
    run_simple_cuda_program("./start_mps.py","/tmp/nvidia-mps-"+args.device)
    if mps_pid:
        # 获取 MPS Server 的 PID
        server_id = get_mps_server_pid(mps_pid)
        MPS_PID = server_id

    # ## 根据传入的参数设定任务的rps搜索范围，覆盖初始值,例如传入vgg16，500，表示希望这个实例负载500的rps，搜索范围为+-200.
    # for i in range(len(task)):
    #     max_RPS_map[task[0]] = min(int(task[1])+200,max_RPS_map[task[0]])
    #     min_RPS_map[task[0]] = max(int(task[1])-200,0)
    for i in range(0, serve_num):
        request.append(int(task[i*2 + 1]))


    if not feedback:
        pass

    else:
        start = time.time()

        for idx in range(0,1):
            staticPruning()
            optimizer = init_optimizer_feedback(serve_num, task)
            utility = UtilityFunction(kind="ei", kappa=5, xi=0.2)
        
            stepLogDir = "../tmp/"+args.task+"-"+args.device+"_3.json"
            dict_list = []

            optimizer.maximize(
                init_points=5,  
                n_iter=10,      
                acquisition_function=utility  
            )
            print(optimizer.max)

            print("checkpoint reached")  # 向主进程发送信号
            time.sleep(1)
            sys.stdout.flush()  # 确保信号立即发送

            #等待主进程的继续信号
            print("[Worker] Waiting for continue signal...")
            while True:
                line = sys.stdin.readline().strip()  # 阻塞等待主进程输入
                if line == "continue":
                    print("[Worker] Received continue signal. Resuming task...")
                    break  # 退出等待，继续运行任务
                else:
                    print(f"[Worker] Received unknown signal: {line}. Waiting again.")

            optimizer = init_optimizer_feedback(serve_num, task)
            changeFlag = True
            changeSurface(optimizer)
            changeFlag = False
            optimizer.maximize(
                init_points=0,  
                n_iter=1,      
                acquisition_function=utility  
            )

            print(optimizer.max)

            with open(stepLogDir, "w") as json_file:
                json.dump(dict_list, json_file, indent=4)


        end = time.time()
        print(end - start)

