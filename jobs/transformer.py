import torch
import torch.nn as nn

# 设置随机输入的参数
k = 2  # 批次大小
seq_len = 512  # 序列长度
d_model = 768  # 嵌入维度

# # 随机生成输入数据
# input = torch.randn(seq_len, k, d_model).cuda(0)  # 随机输入数据
# masks = torch.ones(seq_len, seq_len).cuda(0)  # 注意力掩码

# 创建一个单层Transformer
# transformer_layer = nn.Transformer(d_model=d_model, nhead=8, num_encoder_layers=1, num_decoder_layers=1).cuda(0)
num_encoder_layers = 1
num_decoder_layers = 1

def transformer_layer():
    return  nn.Transformer(d_model=d_model, nhead=8, num_encoder_layers=1, num_decoder_layers=1)
# 推理
# while True:
#     with torch.no_grad():
#         output = transformer_layer(input, input, src_mask=masks, tgt_mask=masks)

#     # 输出结果在GPU上，可以使用 .cpu() 移动到CPU上
#     output_cpu = output.cpu()
#     print(output_cpu)
