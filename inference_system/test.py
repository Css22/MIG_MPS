import torch

# 尝试加载模型
model = torch.jit.load("/data/zbw/inference_system/MIG_MPS/inference_system/model_repository/resnet50/1/model.pt")

model.eval()
model = model.cuda()
print("Model Inputs:")
for i, input in enumerate(model.graph.inputs()):
    print(f"Input {i+1}: {input.debugName()}")

print("\nModel Outputs:")
for i, output in enumerate(model.graph.outputs()):
    print(f"Output {i+1}: {output.debugName()}")
# 打印模型结构以确认加载成功
print(model)


# 进行一次前向推理
input_tensor = torch.randn(1, 3, 224, 224).cuda()
output = model(input_tensor)
print("Model output:", output)
import numpy as np

data = [1,2,3,4,5,6,7,8,9,10,11]
print(np.array(data)[2:])