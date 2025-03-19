# chapter 10 deep learning for timeseries

import os
fname = os.path.join("jena_climate_2009_2016.csv")
from tensorflow import keras

with open(fname) as f:
    data = f.read()


lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(header)
print(len(lines))

# listing 10.2 parsing the data

import numpy as np
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]


# listing 10.3 plotting the temperature timeseries

from matplotlib import pyplot as plt
# plt.plot(range(len(temperature)), temperature)
# plt.show()

# listing 10.4 plotting the first 10 days of the temperature timeseries

# plt.plot(range(1440), temperature[:1440])
# plt.show()

# listing 10.5 computing the number of samples we'll use for each data split
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples


print("num train samples: ", num_train_samples)
print("num val samples: ", num_val_samples)
print("num test samples: ", num_test_samples)

# listing 10.6 normalizing the data

mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std

# print("mean: ", mean)
# print("std: ", std)

# listing 10.7 instantiating datasets for training, validation and testing

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples,
    # drop_remainder=True,
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples,
)

test_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples
)

# listing 10.8 inspecting the output of one of our datasets

for samples, targets in train_dataset:
    print("samples shape: ", samples.shape)
    print("targets shape: ", targets.shape)
    break


# listing 10.9 computing the common sense baseline mae

def evaluate_naive_method(dataset):
    total_abs_err = 0.
    samples_seen = 0
    for samples, targets in dataset:
        preds = samples[:, -1, 1] * std[1] + mean[1]
        total_abs_err += np.sum(np.abs(preds - targets))
        samples_seen += samples.shape[0]
    return total_abs_err / samples_seen

print(f"Validation mae: {evaluate_naive_method(val_dataset):.2f}")
print(f"Test mae: {evaluate_naive_method(test_dataset):.2f}")




# listing 10.10 training and evaluating a densely connected model

from tensorflow import keras
from tensorflow.keras import layers

# this doesnt run on the latest tensorflow verison. maybe on tensorflow===2.6.0
# inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
# x = layers.Flatten()(inputs)
# x = layers.Dense(16, activation="relu")(x)
# outputs = layers.Dense(1)(x)
# model = keras.Model(inputs, outputs)

inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))  # (120, 14)
x = layers.Reshape((sequence_length * raw_data.shape[-1],))(inputs)  # (1680,)
x = layers.Dense(16, activation="relu")(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)


callbacks = [
    keras.callbacks.ModelCheckpoint("jena_dense.keras", save_best_only=True)
]

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset, epochs=10, validation_data=val_dataset, callbacks=callbacks)
model = keras.models.load_model("jena_dense.keras")
print(f"Test mae: {model.evaluate(test_dataset)[1]:.2f}")


