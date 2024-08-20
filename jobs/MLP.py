import torch
import torch.nn as nn
import time
import numpy as np
def get_p99(data):
    data = np.array(data)
    percentile_99 = np.percentile(data, 99)
    return percentile_99

# 定义一个简单的 MLP 模型
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# 初始化模型
input_size = 100  # 输入层大小
hidden_size = 10280  # 隐藏层大小
output_size = 2   # 输出层大小
model = SimpleMLP(input_size, hidden_size, output_size)
model = model.cuda()
# 随机生成输入数据
batch_size = 1280  # 批次大小
input_data = torch.randn(batch_size, input_size)
input_data = input_data.cuda()
# 推理
data = []
while True:
    with torch.no_grad():
        start_time = time.time()
        output = model(input_data).cpu()
        end_time = time.time()
        data.append((end_time - start_time) * 1000)
        if len(data) % 1000 == 0:
            print(get_p99(data))
            data = []   

