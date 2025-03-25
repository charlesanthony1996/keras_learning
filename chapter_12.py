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

