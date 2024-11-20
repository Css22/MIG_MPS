import torch
import torch.nn as nn
import torch.optim as optim

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# 定义一个简单的神经网络模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# 创建模型并移动到 GPU（如果可用）
model = SimpleModel().to(device)

# 定义一个简单的输入张量并移动到 GPU
input_data = torch.randn(64, 10).to(device)

# 定义一个目标张量（用于损失计算）
target = torch.randn(64, 1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 进行一次前向传播、计算损失并进行反向传播
for epoch in range(100):  # 运行 100 次迭代
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    #print(f"Epoch {epoch+1}/100, Loss: {loss.item()}")