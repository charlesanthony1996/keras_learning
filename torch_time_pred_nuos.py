import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data = np.array([
    [1, 2.6], [2, 1.1], [3, 1.07], [4, 2.68], [5, 1.17]
])

# normalize data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

# data
data_x = data[:, 0].reshape(-1, 1)
data_y = data[:, 1].reshape(-1, 1)

data_x_scaled = scaler_x.fit_transform(data_x)
data_y_scaled = scaler_y.fit_transform(data_y)


x = torch.tensor(data_x_scaled[:-1]).float().view(-1, 1, 1)
y = torch.tensor(data_y_scaled[1:]).float().view(-1, 1)

class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(1, 50, batch_first=True)
        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1])
        return x
    
model = LSTMModel()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

# training the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(x)
    loss = criterion(outputs, y.view(-1, 1))
    loss.backward()
    optimizer.step()


# predict next step time
next_step = np.array([[6]])
next_step_scaled = torch.tensor(scaler_x.transform(next_step)).float().view(1, 1, 1)
predicted_time_scaled = model(next_step_scaled).detach().numpy()
preicted_time = scaler_y.inverse_transform(predicted_time_scaled)

print("Predicted time duration for next step: ", preicted_time[0][0])