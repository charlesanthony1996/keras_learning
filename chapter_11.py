# chapter 11 two approaches for representing groups of words: sets and sequences

# one time execution to setup the files for train, val , test

# import os, pathlib, shutil, random

# base_dir = pathlib.Path("aclImdb")
# val_dir = base_dir / "val"
# train_dir = base_dir / "train"

# for category in ("neg", "pos"):
#     os.makedirs(val_dir / category)
#     files = os.listdir(train_dir / category)
#     random.Random(1337).shuffle(files)
#     num_val_samples = int(0.2 * len(files))
#     val_files = files[-num_val_samples:]
#     for fname in val_files:
#         shutil.move(train_dir / category / fname, val_dir / category / fname)


from tensorflow import keras
from tensorflow.keras.layers import TextVectorization
batch_size = 32

train_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/train", batch_size=batch_size
)

val_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/val", batch_size=batch_size
)

test_ds = keras.utils.text_dataset_from_directory(
    "aclImdb/test", batch_size=batch_size
)

# listing 11.2 displaying the shapes and dtypes of the first batch

for inputs, targets in train_ds:
    print("inputs shape: ", inputs.shape)
    print("inputs dtype: ", inputs.dtype)
    print("targets shape: ", targets.shape)
    print("targets dtype: ", targets.dtype)
    print("inputs[0]: ", inputs[0])
    print("targets[0]: ", targets[0])
    break




# listing 11.3 preprocessing our datasets with a textvectorization layer

text_vectorization = TextVectorization(max_tokens=20000, output_mode = "multi_hot")
text_only_train_ds = train_ds.map(lambda x, y: x)
text_vectorization.adapt(text_only_train_ds)

binary_lgram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

binary_lgram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4
)

binary_lgram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y), num_parallel_calls=4 
)

# listing 11.4 inspecting the output of our binary unigram dataset

for inputs, targets in binary_lgram_train_ds:
    print("inputs shape: ", inputs.shape)
    print("inputs dtype: ", inputs.dtype)
    print("targets shape: ", targets.shape)
    print("targets dtype: ", targets.dtype)
    print("inputs[0]: ", inputs[0])
    print("targets[0]: ", targets[0])
    break


# listing 11.5 our model building utility

from tensorflow import keras
from tensorflow.keras import layers

def get_model(max_tokens=20000, hidden_dim=16):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation="relu")(inputs)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer="rmsprop", 
        loss="binary_crossentropy", 
        metrics=["accuracy"]
    )
    return model

# listing 11.6 training and testing the binary unigram model

model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_lgram.keras", save_best_only=True)
]
model.fit(
    binary_lgram_train_ds.cache(), 
    validation_data=binary_lgram_val_ds.cache(), 
    epochs=10, 
    callbacks=callbacks
)
model = keras.models.load_model("binary_lgram.keras")
print(f"Test acc: {model.evaluate(binary_lgram_test_ds)[1]:.3f}")

# listing 11.7 configuring the text vectorization layer to return bigrams

text_vectorization = TextVectorization(
    ngrams= 2,
    max_tokens=20000,
    output_mode="multi_hot"
)

# listing 11.8 training and testing the binary bigram model

text_vectorization.adapt(text_only_train_ds)
binary_2gram_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4
)

binary_2gram_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4
)

binary_2gram_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4
)


model = get_model()
model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("binary_2gram.keras", save_best_only=True)
]

model.fit(
    binary_2gram_train_ds.cache(), 
    validation_data=binary_2gram_val_ds.cache(),
    epochs=10,
    callbacks=callbacks
)

model = keras.models.load_model("binary_2gram.keras")
print(f"Test accuracy: {model.evaluate(binary_2gram_test_ds)[1]:.3f}")

# listing 11.9 configuring the text vectorization layer to return token counts

text_vectorization = TextVectorization(
    ngrams=2,
    max_tokens=20000,
    output_mode="count"
)



# listing 11.10 configuring text vectorization to return tf idf weighted outputs

# text_vectorization = TextVectorization(
#     ngrams=2,
#     max_tokens=20000,
#     output_mode="tf_idf"
# )

# listing 11.11 training and testing the tf idf bigram model

# import tensorflow as tf
# with tf.device('/CPU:0'):
#     text_vectorization.adapt(text_only_train_ds)
# text_vectorization.adapt(text_only_train_ds)

# tfidf_2gram_train_ds = train_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4
# )

# tfidf_2gram_val_ds = val_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4
# )

# tfidf_2gram_test_ds = test_ds.map(
#     lambda x, y: (text_vectorization(x), y),
#     num_parallel_calls=4
# )

# model = get_model()
# model.summary()

# callbacks = [
#     keras.callbacks.ModelCheckpoint("tfidf_2gram.keras", save_best_only=True)
# ]

# model.fit(
#     tfidf_2gram_train_ds.cache(),
#     validation_data=tfidf_2gram_val_ds.cache(),
#     epochs=10,
#     callbacks=callbacks
# )

# model = keras.models.load_model("tfidf_2gram.keras")
# print(f"Test acc: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}")

# listing 11.12 preparing integer sequence datasets

from tensorflow.keras import layers

max_length = 600
max_tokens = 20000

text_vectorization = layers.TextVectorization(
    max_tokens= max_tokens,
    output_mode="int",
    output_sequence_length=max_length
)

text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4
)

int_val_ds = val_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4
)

int_test_ds = test_ds.map(
    lambda x, y: (text_vectorization(x), y),
    num_parallel_calls=4
)


# listing 11.13 a sequence model built on one hot encoded sequences

# import tensorflow as tf

# class one_hot_layer(layers.Layer):
#     def __init__(self, depth):
#         super().__init__()
#         self.depth = depth

#     def call(self, inputs):
#         return tf.one_hot(inputs, depth=self.depth)




# inputs = keras.Input(shape=(None,), dtype="int64")
# # embedded = tf.one_hot(inputs, depth=max_tokens)
# embedded = one_hot_layer(depth=max_tokens)(inputs)

# x = layers.Bidirectional(layers.LSTM(32))(embedded)
# x = layers.Dropout(0.5)(x)

# outputs = layers.Dense(1, activation="sigmoid")(x)
# model = keras.Model(inputs, outputs)

# model.compile(
#     optimizer="rmsprop",
#     loss="binary_crossentropy",
#     metrics=["accuracy"]
# )

# model.summary()

# # listing 11.14 training a first basic sequence model

# callbacks = [
#     keras.callbacks.ModelCheckpoint("one_hot_bidir_lstm.keras", save_best_only=True)
# ]

# model.fit(
#     int_train_ds,
#     validation_data=int_val_ds,
#     epochs=10,
#     callbacks=callbacks
# )

# model = keras.models.load_model("one_hot_bidir_lstm.keras")

# print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

# listing 11.15 instatiating an embedding layer

embedding_layer = layers.Embedding(input_dim = max_tokens, output_dim=256)

# listing 11.16 model that uses an embedding layer trained from scratch

inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)

x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("embeddings_bidir_gru.keras", save_best_only=True)
]

model.fit(
    int_train_ds,
    validation_data=int_val_ds,
    epochs=10,
    callbacks=callbacks
)

model = keras.models.load_model("embeddings_bidir_gru.keras")

print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

# listing 11.17 using an embedding layer with masking method

inputs = keras.Input(shape=(None,), dtype="int64")
embedded = layers.Embedding(
    input_dim=max_tokens,
    output_dim=256,
    mask_zero=True
)(inputs)

x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
callbacks = [
    keras.callbacks.ModelCheckpoint("embeddings_bidir_gru_with_masking.keras", save_best_only=True)
]

model.fit(
    int_train_ds,
    validation_data=int_val_ds,
    epochs=10,
    callbacks=callbacks
)


model = keras.models.load_model("embeddings_bidir_gru_with_masking.keras")
print(f"Testing acc: {model.evaluate(int_test_ds)[1]:.3f}")

# listing 11.18 parsing the glove word embeddings file

import numpy as np
path_to_glove_file = "glove.6B.100d.txt"


embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index[word] = coefs

print(f"Found {len(embeddings_index)} word vectors")

# listing 11.19 preparing the glove word embeddings matrix

embedding_dim = 100

vocabulary = text_vectorization.get_vocabulary()
word_index = dict(zip(vocabulary, range(len(vocabulary))))

embedding_matrix = np.zeros((max_tokens, embedding_dim))
for word, i in word_index.items():
    if i < max_tokens:
        embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


embedding_layer = layers.Embedding(
    max_tokens,
    embedding_dim,
    embeddings_initializer=keras.initializers.Constant(embedding_matrix),
    trainable=False,
    mask_zero=True
)

# listing 11.20 model that uses a pretrained embedding layer

inputs = keras.Input(shape=(None,), dtype="int64")
embedded = embedding_layer(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("glove_embeddings_sequence_model.keras", save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=10, callbacks=callbacks)

model = keras.models.load_model("glove_embeddings_sequence_model.keras")
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")


def self_attention(input_sequence):
    output = np.zeros(shape=(input_sequence.shape))
    for i, pivot_vector in enumerate(input_sequence):
        scores = np.zeros(shape=(len(input_sequence),))
        for j, vector in enumerate(input_sequence):
            scores[j] = np.dot(pivot_vector, vector.T)
        scores /= np.sqrt(input_sequence.shape[1])
        scores = softmax(scores)
        new_pivot_representation = np.zeros(shape=pivot_vector.shape)
        for j, vector in enumerate(input_sequence):
            new_pivot_representation += vector * scores[j]
        output[j] = new_pivot_representation
    return output



# num_heads = 4
# embed_dim = 256
# mha_layer = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
# outputs = mha_layer(inputs, inputs, inputs)

# listing 11.21 Transformer encoder implemented as a subclassed layer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads= num_heads,
            key_dim= embed_dim
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim)]
        )

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim
        })
        return config


def layer_normalization(batch_of_sequences):
    mean = np.mean(batch_of_sequences, keepdims=True, axis=-1)
    variance = np.var(batch_of_sequences, keepdims=True, axis=-1)
    return (batch_of_sequences - mean) / variance

def batch_normalization(batch_of_images):
    mean = np.mean(batch_of_images, keepdims=True, axis=(0, 1, 2))
    variance = np.var(batch_of_images, keepdims=True, axis=(0, 1, 2))
    return (batch_of_images - mean) / variance

# listing 11.22 using the transformer encoder for text classification

vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)

outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# listing 11.23 training and evaluating the transformer encoder based model

callbacks = [
    keras.callbacks.ModelCheckpoint("transformer_encoder.keras", save_best_only=True)
]

model.fit(int_train_ds, validation_data=int_val_ds, epochs=20, callbacks=callbacks)

model = keras.models.load_model("transformer_encoder.keras", custom_objects={"TransformerEncoder": TransformerEncoder})
print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")


# listing 11.24 implementing positional embedding as a subclassed layer

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, ** kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim = input_dim,
            output_dim = output_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim = sequence_length,
            output_dim = output_dim
        )
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim


    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def compute_mask(self, inputs, mask=None):
        # return tf.math.not_equal(inputs, 0)
        return keras.ops.not_equal(inputs, 0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim
        })
        return config
    

# listing 11.25 combining the transformer decoder with positional embedding

vocab_size = 20000
sequence_length = 600
embed_dim = 256
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype="int64")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = keras.Model(inputs, outputs)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

callbacks = [
    keras.callbacks.ModelCheckpoint("full_transformer_encoder.keras", save_best_only=True)
]

model.fit(
    int_train_ds,
    validation_data=int_val_ds,
    epochs=20,
    callbacks=callbacks
)

model = keras.models.load_model(
    "full_transformer_encoder.keras",
    custom_objects={"TransformerEncoder": TransformerEncoder,
                    "PositionalEmbedding": PositionalEmbedding}
)

print(f"Test acc: {model.evaluate(int_test_ds)[1]:.3f}")

# examples with the dataset
# !wget http:/ /storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip
# !unzip -q spa-eng.zip

text_file = "spa-eng/spa.txt"
with open(text_file) as f:
    lines = f.read().split("\n")[:-1]
text_pairs = []

for line in lines:
    english, spanish = lines.split("\t")
    spanish = "[start] " + spanish + " [end]"
    text_pairs.append((english, spanish))

import random
print(random.choice(text_pairs))

# shuffling them and split them into the usual training, validation and test sets

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]


# listing 11.26 vectorizing the english and spanish text pairs

import tensorflow as tf
import string
import re

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

def custom_standarization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
        lowercase, f"[{re.escape(strip_chars)}]", ""
    )

vocab_size = 20000
sequence_length = 20

source_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
)

target_vectorization = layers.TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standarization
)

train_english_texts = [pair[0] for pair in train_pairs]
train_spanish_texts = [pair[1] for pair in train_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)

# listing 11.27 preparing datasets for the transition task

batch_size = 64

def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return ({
        "english": eng,
        "spanish": spa[:, :-1],
    }, spa[:, 1:])

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_ds)

# heres what our dataset looks like

for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: { inputs['english'].shape}")
    print(f"inputs['spanish'].shape: { inputs['spanish'].shape}")
    print(f"targets.shape: { targets.shape }")


inputs = keras.Input(shape=(sequence_length,), dtype="int64")
x = layers.Embedding(input_dim=vocab_size, output_dim=128)(inputs)
x = layers.LSTM(32, return_sequences=True)(x)
outputs = layers.Dense(vocab_size, activation="softmax")(x)
model = keras.Model(inputs, outputs)

# listing 11.28 gru based encoder

from tensorflow import keras
from tensorflow.keras import layers

embed_dim = 256
latent_dim = 1024

source = keras.Input(shape=(None,), dtype="int64", name="english")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
encoded_source = layers.Bidirectional(
    layers.GRU(latent_dim), merge_mode="sum"
)(x)


# listing 11.29 gru based decoder and the end to end model

past_target = keras.Input(shape=(None,), dtype="int64", name="spanish")
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
decoder_gru = layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source)
x = layers.Dropout(0.5)(x)
target_next_step = layers.Dense(vocab_size, activation="softmax")(x)
seq2seq_rnn = keras.Model([source, past_target], target_next_step)

# listing 11.30 training our recurrent sequence to sequence to model

seq2seq_rnn.compile(
    optimizer="rmsprop",
    loss="sparse_cagtegorical_crossentropy",
    metrics=["accuracy"]
)

seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)

# listing 11.31 translating new sentences with our rnn encoder and decoder

import numpy as np
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sequence):
    tokenized_input_sentence = source_vectorization([input_sequence])
    decoded_sequence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decode_sequence])
        next_token_predictions = seq2seq_rnn.predict(
            [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(next_token_predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sequence += " " + sampled_token
        if sampled_token == "[end]":
            break
    return decoded_sequence

test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(20):
    input_sequence = random.choice(test_eng_texts)
    print("-")
    print(input_sequence)
    print(decode_sequence(input_sequence))



# listing 11.32 some sample results from the current translation model

### some text here

# listing 11.33 the transformer decoder

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

        self.attention_1 = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads = num_heads, key_dim=embed_dim
        )

        self.dense_proj = keras.Sequential([
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        ])

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config(self)
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim
        })
        return config


# listing 11.34 Transformer decoder method that generates a casual mask

def get_casual_attention(self, inputs):
    input_shape = tf.shape(inputs)
    batch_size, sequence_length = input_shape[0], input_shape[1]
    i = tf.range(sequence_length)[:, tf.newaxis]
    j = tf.range(sequence_length)
    mask = tf.cast(i >= j, dtype="int32")
    mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1),
        tf.constant([1, 1], dtype=tf.int32)], axis=0
    )
    return tf.tile(mask, mult)

# listing 11.35 the forward pass of the transformer decoder

def call(self, inputs, encoder_outputs, mask=None):
    casual_mask = self.get_casual_attention_mask(inputs)
    if mask is not None:
        padding_mask = tf.cast(
            mask[:, tf.newaxis, :], dtype="int32"
        )
        padding_mask = tf.minimum(padding_mask, casual_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=casual_mask
        )
        attention_output_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )

        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2)
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)
    

# listing 11.36 end to end transformer

embed_dim = 256
dense_dim = 2048
num_heads = 2

encoder_inputs = keras.Input(Shape=(None,), dtype="int64", name="english")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_inputs = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)


