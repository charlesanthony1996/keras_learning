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
std = raw_data[:num_train_samples].mean(axis=0)
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
    end_index=num_train_samples
)

val_dataset = keras.utils.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples,
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