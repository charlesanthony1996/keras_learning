import numpy as np

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)


def attention(query, keys, values):
    scores = np.dot(keys, query)
    weights = softmax(scores)
    return np.dot(weights, values)

def feed_forward(x):
    variation = np.array([(i % 5) - 2 for i in range(len(x))], dtype=np.float32)
    h = x * variation
    return np.maximum(0, h)

def cosine_similarity(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(v1, v2) / (norm1 * norm2)

def predict_next(output_vector, all_embeddings, avoid_word):
    best_word = None
    best_sim = -1
    for word, emb in all_embeddings.items():
        if word in ["<start>", avoid_word]:
            continue
        sim = cosine_similarity(output_vector, emb)
        if sim > best_sim:
            best_sim = sim
            best_word = word
    
    return best_word


# run the model
encoder_word = "apple"
glove_path = "glove.6B.300d.txt"
word_embeddings = load_glove_embeddings(glove_path)

dim = len(next(iter(word_embeddings.values())))
word_embeddings["<start>"] = np.zeros(dim, dtype=np.float32)

if encoder_word not in word_embeddings:
    print(f"{encoder_word} not found in glove embeddings")
    exit(1)

# should print out 300 because of the number of dimensions. read the name of the dataset again
# print(dim)

encoder_output = word_embeddings[encoder_word].reshape(1, -1)
decoder_input = word_embeddings["<start>"]

cross_attn_out = attention(decoder_input, encoder_output, encoder_output)
ffn_out = feed_forward(cross_attn_out)
final_output = cross_attn_out + ffn_out

predicted_word = predict_next(final_output, word_embeddings, avoid_word=encoder_word)

print("Encoder word: ", encoder_word)
print(predicted_word)
