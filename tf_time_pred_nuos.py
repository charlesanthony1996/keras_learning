import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
# from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# this should be in real time or stored in a csv
data = np.array([
    [1, 2.6],
    [2, 1.1],
    [3, 1.07],
    [4, 2.68],
    [5, 1.17]
])

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# predict input, output
x = data_scaled[:-1, 0].reshape(-1, 1, 1)
y = data_scaled[1:, 1]

# define a lstm model
model = keras.Sequential([
    LSTM(50, activation="relu", return_sequences=True, input_shape=(1, 1)),
    LSTM(50, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

# model fitting
model.fit(x, y, epochs=1, verbose=0)

# predict next step
next_step = np.array([[6]])
next_step_scaled = scaler.transform(next_step.reshape(-1, 1))
predicted_time = model.predict(next_step_scaled.reshape(1, 1, 1))

print("Predicted duration for next step: ", scaler.inverse_transform(predicted_time.reshape(-1, 1))[0][0])