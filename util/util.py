import argparse
import math


file_path = '/data/zbw/inference_system/MIG_MPS/log/'
model_list = ["resnet50", "resnet101", "resnet152", "vgg19", "vgg16", "unet", "deeplabv3", "mobilenet_v2", "alexnet", "bert"]


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
SM_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
for i in model_list:
    model_log_path = file_path + f"{i}_MPS_RPS"
    config_list = read_data_from_file(model_log_path)
    

    model_QPS_list[i] = []
    for sm_value in SM_list:
        QPS = find_max_rps_under_p99(config_list, sm_value, half_QoS_map.get(i))
        model_QPS_list[i].append(QPS)


def check_potential_RPS(model, SM):
    QPS_list = model_QPS_list.get(model)
    index = SM_list.index(SM)

    if index == 0:

        half_QoS = half_QoS_map.get(model)
        min_batch = math.floor(QPS_list[0]/1000 * half_QoS)
        print(min_batch, min_batch)

    else:
        half_QoS = half_QoS_map.get(model)
        min_batch = math.floor(QPS_list[index-1]/1000 * half_QoS)
        max_batch = math.floor(QPS_list[index]/1000 * half_QoS)

        print(min_batch, max_batch)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str)
    parser.add_argument("--SM", type=int)
    args = parser.parse_args()

    task = args.task
    SM = args.SM

    check_potential_RPS(task, SM)