import torch
import torch.nn as nn
import time
import numpy as np
# 模拟大规模嵌入表
num_users = 10000000
num_items = 10000000
embedding_dim = 120


# 创建用户和物品嵌入表
user_embedding = nn.Embedding(num_users, embedding_dim).cuda()
item_embedding = nn.Embedding(num_items, embedding_dim).cuda()

# 随机生成用户和物品ID
user_ids = torch.randint(0, num_users, (1280,)).cuda()  # batch size = 32
item_ids = torch.randint(0, num_items, (1280,)).cuda()

def get_p99(data):
    data = np.array(data)
    percentile_99 = np.percentile(data, 99)
    return percentile_99


data= []
while True:
    with torch.no_grad():
        start_time = time.time()
                
        user_vectors = user_embedding(user_ids)
        item_vectors = item_embedding(item_ids)

            
        scores = torch.sum(user_vectors * item_vectors, dim=1)


        end_time = time.time()
        data.append((end_time - start_time) * 1000)
        if len(data) % 10000 == 0:
            print(get_p99(data))
            data = []   
while True:
    user_vectors = user_embedding(user_ids)
    item_vectors = item_embedding(item_ids)

    # 计算推荐得分 (这里简单使用点积)
    scores = torch.sum(user_vectors * item_vectors, dim=1)

    print(scores)
