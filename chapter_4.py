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
history_dict = history.history
loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, "bo", label="Training loss")
plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
# plt.show()

# listing 4.9 plotting the training and validation accuracy

plt.clf()
acc = history_dict["accuracy"]
val_acc = history_dict["val_accuracy"]
plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
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
(train_data, train_labels), (test_data, train_data) = reuters.load_data(num_words=10000) 

# print(test_data[0])

train_data[10]

# listing 4.12 decoding newswires back to text

word_index = reuters.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()]
)

decoded_newswire = " ".join([reverse_word_index.get(i - 3, "?") for i in train_data[0]])

print(train_labels[10])



