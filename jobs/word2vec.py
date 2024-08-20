import gensim.downloader as api
import torch
import random
import time
# 加载预训练的Word2Vec模型（Google News数据集）
model = api.load('word2vec-google-news-300')

# 获取词向量并将其转换为PyTorch张量
word_vectors = torch.tensor(model.vectors)

# 将词向量移动到GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
word_vectors = word_vectors.to(device)

# 随机选择单词并获取其向量表示
random_word = random.choice(model.index_to_key)
random_vector = torch.tensor(model[random_word]).to(device)
top_n = 5
# 在GPU上计算随机单词与词汇表中所有单词的余弦相似度
while True:
    start_time = time.time()
    cosine_similarities = torch.nn.functional.cosine_similarity(random_vector.unsqueeze(0), word_vectors)


    top_n_similarities, top_n_indices = torch.topk(cosine_similarities, top_n)

    similar_words = [model.index_to_key[idx] for idx in top_n_indices.cpu().numpy()]
    end_time = time.time()
    print((end_time - start_time) * 1000)


