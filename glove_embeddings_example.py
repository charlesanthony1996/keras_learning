import math

def load_glove_embeddings(file_path, vocab):
    embeddings = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                vector = [float(x) for x in parts[1:]]
                embeddings[word] = vector
            if len(embeddings) == len(vocab):
                break
    return embeddings

vocab = ["hello", "world", "bonjour", "monde", "le", "yellow", "black", "white", "dark", "darkness", "night", "<start>"]
word_embeddings = load_glove_embeddings("glove.6B.50d.txt", vocab)

# Fix 1: Insert this manually
word_embeddings["<start>"] = [0.0] * 50

# Fix 2: Define encoder outputs BEFORE using it
encoder_outputs = [word_embeddings["dark"]]

decoder_input = [word_embeddings["black"]]  # or use "bonjour" instead
self_attn_out = decoder_input[0]



self_attn_out = decoder_input[0]

def dot(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

# def zip(v1, v2):
#     return sum(zip(v1, v2))

def add(v1, v2):
    return [x + y for x, y in zip(v1, v2)]

# def softmax(scores):
#     exp_scores = [math.exp(s) for s in scores]
#     total = sum()

def softmax(scores):
    exp_scores = [math.exp(s) for s in scores]
    total = sum(exp_scores)
    return [s / total for s in exp_scores]


def scalar_mult(scalar, vector):
    return [scalar * x for x in vector]


def attention(query, key, value):
    scores = [dot(query, k) for k in key]
    weights = softmax(scores)
    weighted_sum = [0.0] * len(value[0])
    for i in range(len(value)):
        weighted_sum = add(weighted_sum, scalar_mult(weights[i], value[i]))
    return weighted_sum

cross_attn_out = attention(self_attn_out, encoder_outputs, encoder_outputs)

def feed_forward(x):
    # First linear layer: W1 = identity * 2
    h = scalar_mult(2, x)
    # ReLU
    h_relu = [max(0, v) for v in h]
    # Second linear layer: W2 = identity * 0.5
    out = scalar_mult(0.5, h_relu)
    return out
    
ffn_out = feed_forward(cross_attn_out)
final_output = add(cross_attn_out, ffn_out)

print("Final decoder vector:", final_output)

def cosine_similarity(v1, v2):
    dot_prod = dot(v1, v2)
    norm1 = math.sqrt(dot(v1, v1))
    norm2 = math.sqrt(dot(v2, v2))
    return dot_prod / (norm1 * norm2)

def predict_next(output_vector, word_embeddings):
    best_word = None
    best_sim = -1
    for word, emb in word_embeddings.items():
        norm_emb = math.sqrt(dot(emb, emb))
        if norm_emb == 0:  # skip zero vectors like <start>
            continue
        sim = cosine_similarity(output_vector, emb)
        if sim > best_sim:
            best_sim = sim
            best_word = word
    return best_word



predicted_word = predict_next(final_output, word_embeddings)
print("Predicted next word:", predicted_word)
