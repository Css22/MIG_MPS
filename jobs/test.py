from transformers import GPTJForCausalLM, AutoTokenizer
import torch

# 加载 GPT-J 模型
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# 输入文本
input_text = "Once upon a time"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.cuda()

# 推理生成
with torch.no_grad():
    outputs = model.generate(input_ids, max_length=50)

# 解码输出
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(output_text)
