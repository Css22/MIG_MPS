import numpy as np
import matplotlib.pyplot as plt
from openbox import Observation, History, Advisor, space as sp, logger
import pickle
from openbox import Optimizer
from skopt import gp_minimize
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction
import json

# 打开并读取 JSON 文件
with open('./tmp/resnet152_1000_resnet152_1000_1.json', 'r') as file:
    data = json.load(file)

# 找到 target 最大的行
max_target_entry = max(data, key=lambda x: x['target'])
print(max_target_entry)
print(max_target_entry['params']['RPS0'])






