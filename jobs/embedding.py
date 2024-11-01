import numpy as np
import torch
import torch.nn as nn
import time

# 模拟大规模嵌入表
num_users = 10000000
num_items = 10000000
embedding_dim = 120


# 创建用户和物品嵌入表
user_embedding = nn.Embedding(num_users, embedding_dim)
item_embedding = nn.Embedding(num_items, embedding_dim)

nn.init.normal_(user_embedding.weight, mean=0, std=0.01)
nn.init.normal_(item_embedding.weight, mean=0, std=0.01)

user_embedding = user_embedding.cuda()
item_embedding = item_embedding.cuda()
# 随机生成用户和物品ID
user_ids = torch.randint(0, num_users, (128000,)).cuda()  # batch size = 32
item_ids = torch.randint(0, num_items, (128000,)).cuda()

def get_p99(data):
    data = np.array(data)
    percentile_99 = np.percentile(data, 99)
    return percentile_99


data= []
while True:
    # user_ids = torch.randint(0, num_users, (12800,)).cuda()  # batch size = 32
    # item_ids = torch.randint(0, num_items, (12800,)).cuda()
    with torch.no_grad():
        start_time = time.time()
        user_vectors1 = user_embedding(user_ids)
        item_vectors1 = item_embedding(item_ids)

        user_vectors2 = user_embedding(user_ids)
        item_vectors2 = item_embedding(item_ids)
        
        user_vectors3 = user_embedding(user_ids)
        item_vectors3 = item_embedding(item_ids)

        combined_user_vectors = user_vectors1 + user_vectors2 + user_vectors3
        combined_item_vectors = item_vectors1 + item_vectors2 + item_vectors3


        combined_user_vectors = combined_user_vectors.cpu()
        # combined_item_vectors = combined_item_vectors.cpu()
            
        # scores = torch.sum(user_vectors * item_vectors, dim=1)


        end_time = time.time()
        # data.append((end_time - start_time) * 1000)
        print((end_time - start_time) * 1000)
        # if len(data) % 100 == 0:
        #     print(get_p99(data))
        #     data = []   
while True:
    user_vectors = user_embedding(user_ids)
    item_vectors = item_embedding(item_ids)

    # 计算推荐得分 (这里简单使用点积)
    scores = torch.sum(user_vectors * item_vectors, dim=1)

    print(scores)
