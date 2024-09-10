from skopt import gp_minimize
import numpy as np
from bayes_opt import BayesianOptimization


# 定义目标函数
def objective(x):
    return -(x - 2) ** 2 + 1

# 定义参数的边界
pbounds = {'x': (-10, 10)}

# 初始化贝叶斯优化对象
optimizer = BayesianOptimization(
    f=objective,
    pbounds=pbounds,
    random_state=1,
)

# 执行优化
optimizer.maximize(
    init_points=2,  # 初始化步数
    n_iter=25,      # 迭代步数
)

# 输出结果
for i, res in enumerate(optimizer.res):
    print(f"Iteration {i+1}: x={res['params']['x']}, target={res['target']}")