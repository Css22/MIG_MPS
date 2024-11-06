from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
model_list = ["resnet50", "resnet101", "resnet152", "vgg19", "vgg16", "unet", "deeplabv3", "mobilenet_v2", "alexnet", "bert"]
file_path = '/data/zbw/inference_system/MIG_MPS/log/'
SM_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

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


half_QoS_map = {

}

for i in QoS_map.keys():
    half_QoS_map[i] = QoS_map[i]/2


class ConfigData:
    def __init__(self, sm, rps, p99):
        self.sm = int(sm)  # SM值
        self.RPS = int(rps)  # RPS值
        self.p99 = float(p99)  # P99值

    def __repr__(self):
        return f"ConfigData(SM: {self.sm}, RPS: {self.RPS}, P99: {self.p99})"

def parse_line(line):
    parts = line.strip().split(", ")
    SM = parts[0].split(": ")[1]
    P99 = parts[1].split(": ")[1]
    RPS = parts[2].split(": ")[1]
    return ConfigData(SM, RPS, P99)

def read_data_from_file(file_path):
    config_list = []
    with open(file_path, 'r') as file:
        for line in file:
            config = parse_line(line)
            config_list.append(config)
    return config_list



def find_max_rps_under_p99(config_list, sm_value, p99_threshold):
    max_rps = None
    for config in config_list:
        if config.sm == sm_value and config.p99 <= p99_threshold:
            if max_rps is None or config.RPS > max_rps:
                max_rps = config.RPS
    return max_rps

model_QPS_list = {}
for i in model_list:
    model_log_path = file_path + f"{i}_MPS_RPS"
    config_list = read_data_from_file(model_log_path)

    model_QPS_list[i] = []
    for sm_value in SM_list:
        QPS = find_max_rps_under_p99(config_list, sm_value, half_QoS_map.get(i))
        model_QPS_list[i].append(QPS)


def cal_similarity(vec1, vec2):
    sum = 0
    for i in range(len(vec1)):
        sum += (vec1[i] - vec2[i])**2
    return (sum/len(vec1))**(1/2)

# for model in model_list:
#     vec = model_QPS_list.get(model)
#     m ,b = np.polyfit(SM_list,vec,1)
#     print(model + "'s k is {}".format(m))

unique_pairs = list(combinations(model_list, 2))
unique_pairs_cov = {}
for pair in unique_pairs:
    vec1 = model_QPS_list.get(pair[0])
    vec2 = model_QPS_list.get(pair[1])
    m1,b1 = np.polyfit(SM_list,vec1,1)
    m2,b2 = np.polyfit(SM_list,vec2,1)

    similarity = min(m1,m2)/max(m1,m2)
    unique_pairs_cov[pair] = similarity

for key, value in unique_pairs_cov.items():
    print(f"key: {key}, val: {value}")
