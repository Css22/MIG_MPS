from transformers import GPTJForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import random

# 加载 GPT-J 模型
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

# 输入文本
dataset = load_dataset("imdb", split="train")
sentences = dataset['text']

# 从数据集中随机选择一个句子

# 推理生成
while True:
    with torch.no_grad():
        random_text = random.choice(sentences)
        input_ids = tokenizer(random_text, return_tensors="pt").input_ids.cuda()        
        outputs = model.generate(input_ids, max_length=50)

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(output_text)
