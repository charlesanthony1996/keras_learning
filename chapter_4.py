# chapter 4 -> getting started: with neural networks classfication and regression

# listing 4.1 loading the imdb dataset
from tensorflow.keras.datasets import imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# print(train_data[0])
print(train_labels[0])


print("max: ", max([max(sequence) for sequence in train_data]))

# listing 4.2 decoding reviews back to text

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

decoded_review = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])

print(decoded_review)

# listing 4.3 encoding the integer sequences vie multi hot encoding

import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        for j in sequence:
            results[i, j] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# print(x_train[0])

y_train = np.asarray(train_labels).astype("float32")
y_test = np.asarray(test_labels).astype("float32")

# listing 4.4 model definition

from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# listing 4.5 compiling the model

model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# listing 4.6 setting aside a validation set

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


# listing 4.7 training your model

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs = 20,
    batch_size=512,
    validation_data=(x_val, y_val)
)


history_dict = history.history
history_dict.keys()

print(history_dict.keys())


# listing 4.8 plotting the training and validation loss

import matplotlib.pyplot as plt
# history_dict = history.history
# loss_values = history_dict["loss"]
# val_loss_values = history_dict["val_loss"]
# epochs = range(1, len(loss_values) + 1)
# plt.plot(epochs, loss_values, "bo", label="Training loss")
# plt.plot(epochs, val_loss_values, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.show()

# listing 4.9 plotting the training and validation accuracy

# plt.clf()
# acc = history_dict["accuracy"]
# val_acc = history_dict["val_accuracy"]
# plt.plot(epochs, acc, "bo", label="Training accuracy")
# plt.plot(epochs, val_acc, "b", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.show()

# listing 4.10 retraining a model from scratch

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])


model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=4, batch_size=512)

results = model.evaluate(x_test, y_test)
print("results: ", results)

# using a trained model to generate predictions on new data

print(model.predict(x_test))


# listing 4.11 loading the reuters dataset

from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000) 

# print(test_data[0])

train_data[10]

# listing 4.12 decoding newswires back to text

word_index = reuters.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])

print(train_labels[10])

# listing 4.13 encoding the input data

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# listing 4.14 encoding the labels

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))

    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(train_labels)
y_test = to_categorical(test_labels)


# listing 4.15 model definition

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

# listing 4.16 compiling the model

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# listing 4.17 setting aside a validation set

x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# listing 4.18 training the model

history = model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# listing 4.19 plotting the training and validation loss

# loss = history.history["loss"]
# val_loss = history.history["val_loss"]
# epochs = range(1, len(loss) + 1)
# plt.plot(epochs, loss, "bo", label="Training loss")
# plt.plot(epochs, val_loss, "b", label="Validation loss")
# plt.title("Training and validation loss")
# plt.legend()
# plt.show()



# listing 4.20 plotting the training and validation accuracy

# plt.clf()
# acc = history.history["accuracy"]
# val_acc = history.history["val_accuracy"]
# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, "bo", label="Training accuracy")
# plt.plot(epochs, val_acc, "b", label="Validation accuracy")
# plt.title("Training and validation accuracy")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()


# listing 4.21 retraining a model from scratch

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(46, activation="softmax")
])

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train,
    y_train,
    epochs=9,
    batch_size=512,
)

results = model.evaluate(x_test, y_test)
print(results)

import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
hits_array.mean()
print("hits array mean: ", hits_array.mean())

# generating predictions on new data

predictions = model.predict(x_test)

print(predictions[0].shape)

print(np.sum(predictions))

# the largest entry is the predicted class

print(np.max(predictions[0]))

# a different way to handle the labels and the loss

y_train = np.array(train_labels)
y_test = np.array(test_labels)

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# listing 4.22 a model with an information bottleneck

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(4, activation="relu"),
    layers.Dense(46, activation="softmax")
])

model.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    partial_x_train,
    partial_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)

# listing 4.23 loading the boston housing dataset

from tensorflow.keras.datasets import boston_housing
(train_data, train_targets), (test_data, test_targets) = (boston_housing.load_data())

# lets look at the data
print(train_data.shape)

print(test_data.shape)

print("train targets: ", train_targets.shape)

# listing 4.24 normalizing the data

mean = train_data.mean(axis=0)
train_data -= mean

std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

# listing 4.25 model definition

def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])

    model.compile(optimizer="rmsprop", loss="mse", metrics=["mse"])
    return model

# listing 4.26 k-fold definition

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print(f"Processing fold #{i}")
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
        train_data[(i + 1) * num_val_samples:]],
        axis=0
    )
    partial_train_targets = np.concatenate(
        []
    )
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets)