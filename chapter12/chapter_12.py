# chapter 12 generative deep learning

# listing 12.1 reweighting a probability distribution to a different temperature

import numpy as np

def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    return distribution / np.sum(distribution)


# listing 12.2 downloading and uncompressing the imdb movie reviews dataset

# !wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
# !tar -xf aclImdb_v1.tar.gz

# listing 12.3 creating a dataset from text files (one file = one sample)

import tensorflow as tf
from tensorflow import keras

dataset = keras.utils.text_dataset_from_directory(
    directory="aclImdb", label_mode=None, batch_size=256
)

dataset = dataset.map(lambda x: tf.strings.regex_replace(x, "<br />", " "))

# listing 12.4 preparing a text vectorization layer

from tensorflow.keras.layers import TextVectorization

sequence_length = 100
vocab_size = 15000

text_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length
)

text_vectorization.adapt(dataset)



# listing 12.5 setting up a language modelling dataset

def prepare_lm_dataset(text_batch):
    vectorized_sequences = text_vectorization(text_batch)
    x = vectorized_sequences[:, :-1]
    y = vectorized_sequences[:, 1:]
    return x, y

lm_dataset = dataset.map(prepare_lm_dataset, num_parallel_calls=4)

# listing 12.6 a simple transformed based language model

from tensorflow.keras import layers