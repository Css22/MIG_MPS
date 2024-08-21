from transformers import GPTJForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import random
import time

# 加载 GPT-J 模型
model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").cuda()
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")


random.seed(32)
# 输入文本
dataset = load_dataset("imdb", split="train")
sentences = dataset['text']

random_text = random.choice(sentences)
input_ids = tokenizer(random_text, return_tensors="pt").input_ids.cuda()
# 从数据集中随机选择一个句子

# 推理生成
while True:
    with torch.no_grad():

        start_time = time.time()        
        outputs = model.generate(input_ids, max_length=100)

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = time.time()
        print((end_time - start_time) * 1000)
