import math

def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            embeddings[word] = vector
    return embeddings


def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def add(v1, v2):
    return [x + y for x, y in zip(v1, v2)]

def scalar_mult(scalar, vector):
    return [scalar * x for x in vector]

def softmax(scores):
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores)
    return [s / total for s in exp_scores]

def attention(query, key, value):
    scores = [dot(query, k) for k in key]
    weights = softmax(scores)
    weighted_sum = [0.0] * len(value[0])
    for i in range(len(value)):
        weighted_sum = add(weighted_sum, scalar_mult(weights[i], value[i]))
    return weighted_sum

# def feed_forward(x):
#     h = scalar_mult(2, x)
#     h_relu = [max(0, v) for v in h]
#     return scalar_mult(0.5, h_relu)

def feed_forward(x):
    # Simulate non-identity weights
    out = []
    for i in range(len(x)):
        new_val = x[i] * ((i % 5) - 2)  # just some variation
        out.append(max(0, new_val))
    return out


def cosine_similarity(v1, v2):
    dot_prod = dot(v1, v2)
    norm1 = math.sqrt(dot(v1, v1))
    norm2 = math.sqrt(dot(v2, v2))
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_prod / (norm1 * norm2)

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


encoder_word = "rapping"


glove_path = "glove.6B.300d.txt"
word_embeddings = load_glove_embeddings(glove_path)


word_embeddings["<start>"] = [0.0] * 50


if encoder_word not in word_embeddings:
    print(f"'{encoder_word}' not found in GloVe embeddings.")
    exit(1)


encoder_outputs = [word_embeddings[encoder_word]]


decoder_input = [word_embeddings["<start>"]]
self_attn_out = decoder_input[0]


cross_attn_out = attention(self_attn_out, encoder_outputs, encoder_outputs)
ffn_out = feed_forward(cross_attn_out)
final_output = add(cross_attn_out, ffn_out)


predicted_word = predict_next(final_output, word_embeddings, avoid_word=encoder_word)

print("Encoder word:", encoder_word)
print("Predicted next word:", predicted_word)
