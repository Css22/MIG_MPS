import fasttext
import torch
import time

model = fasttext.load_model('cc.en.300.bin')

vocab = model.get_words()
vectors = torch.tensor([model.get_word_vector(word) for word in vocab])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vectors = vectors.to(device)

word = 'motherland'
word_vector = torch.tensor(model.get_word_vector(word)).to(device)


while True:
    with torch.no_grad():
        start_time = time.time()
        cosine_similarities = torch.nn.functional.cosine_similarity(word_vector.unsqueeze(0), vectors)

        top_n = 5
        top_n_similarities, top_n_indices = torch.topk(cosine_similarities, top_n)
        similar_words = [vocab[idx] for idx in top_n_indices.cpu().numpy()]
        end_time = time.time()
        print((end_time - start_time) * 1000)