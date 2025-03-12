# listing 5.1 adding white noise channels or all zeros channels to mnist

from tensorflow.keras.datasets import mnist
import numpy as np

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

train_images_with_noise_channels = np.concatenate(
    [train_images, np.random.random((len(train_images), 784)),], axis=1)

train_images_with_zeros_channels = np.concatenate(
    [train_images, np.zeros((len(train_images), 784))], axis=1)


# listing 5.2 training the same model on mnist data with noise channels or all zero channels

from tensorflow import keras
from tensorflow.keras import layers

def get_model():
    model = keras.Sequential([
        layers.Dense(512, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    return model

model = get_model()
history_noise = model.fit(
    train_images_with_noise_channels, 
    train_labels, 
    epochs=10, 
    batch_size=128, 
    validation_split=0.2
)

model = get_model()
history_zeros = model.fit(
    train_images_with_zeros_channels,
    train_labels,
    epochs = 10,
    batch_size = 128,
    validation_split=0.2
)


# listing 5.3 plotting a validation accuracy comparison

import matplotlib.pyplot as plt
val_acc_noise = history_noise.history["val_accuracy"]
val_acc_zeros = history_zeros.history["val_accuracy"]

epochs = range(1, 11)
plt.plot(epochs, val_acc_noise, "b-", label="Validation accuracy with noise channels")
plt.plot(epochs, val_acc_zeros, "b--", label="Validation accuracy with zeros channels")

plt.title("Effect")
plt.xlabel("Epochs")
plt.ylabel("Validation accuracy")
plt.legend()

# plt.show()

# listing 5.4 fitting an mnist model with randomly shuffled labels

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

random_train_labels = train_labels[:]
np.random.shuffle(random_train_labels)

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="rmsprop", 
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_images,
    random_train_labels,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)

# code in between is about templates, which is not that necessary

# listing 5.7 training an mnist model with an incorrectly high learning rate

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.RMSprop(1.),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=3, batch_size=128, validation_split=0.2)

# listing 5.8 the same model

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.RMSprop(1e-2),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_images, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# listing 5.9 a simple logistic regression on mnist

model = keras.Sequential([layers.Dense(10, activation="softmax")])
model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_small_model = model.fit(
    train_images,
    train_labels,
    epochs=20,
    batch_size=128,
    validation_split=0.2
)

import matplotlib.pyplot as plt
val_loss = history_small_model.history["val_loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b--", label="Validation loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
# plt.show()


# lets try training a bigger model, one with two intermediate layers with 96 units each:

model = keras.Sequential([
    layers.Dense(96, activation="relu"),
    layers.Dense(96, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="rmsprop",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_large_model = model.fit(train_images, train_labels, epochs=20, batch_size=128, validation_split=0.2)

# import matplotlib.pyplot as plt
# val_loss = history_large_model.history["val_loss"]
# epochs = range(1, 21)
# plt.plot(epochs, val_loss, "b--", label="Validation loss")
# plt.title("Effect of insufficient model capacity on validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.show()

# listing 5.10 original model

from tensorflow.keras.datasets import imdb
# from tensorflow import keras
# from tensorflow.keras import layers
(train_data, train_labels), _ = imdb.load_data(num_words=10000)

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

train_data = vectorize_sequences(train_data)

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])


model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

history_original = model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4)

# listing 5.11 Version of the model with lower capacity

model = keras.Sequential([
    layers.Dense(4, activation="relu", input_shape=(10000,)),
    layers.Dense(4, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_original = model.fit(train_data, train_labels, epochs=20, batch_size = 512, validation_split=0.4)

# listing 5.12 version of the model with higher capacity

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(512, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

hisotry_larger_model = model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_split=0.4)

# listing 5.13 adding l2 weight regularization to the model

from tensorflow.keras import regularizers

model = keras.Sequential([
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.002)),
    layers.Dense(16, kernel_regularizer=regularizers.l2(0.002)),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

history_l2_reg = model.fit(train_data, train_labels, epochs=20, batch_size=20, validation_split=0.4)

import matplotlib.pyplot as plt
val_loss = history_large_model.history["val_loss"]
epochs = range(1, 21)
plt.plot(epochs, val_loss, "b--", label="Validation loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
# plt.show()

# listing 5.14 different weight regularizers available in keras

from tensorflow.keras import regularizers
regularizers.l1(0.001)
regularizers.l1_l2(l1= 0.001,l2 = 0.002)

# studying dropout -> just an example

# layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
# layer_output *= 0.5


# listing 5.15 adding dropout to the imdb model

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history_dropout = model.fit(
    train_data,
    train_labels,
    epochs=20,
    batch_size = 512,
    validation_split=0.4
)


