import numpy as np
from tinygrad.tensor import Tensor

x_train = np.array([[1], [2], [3], [4], [5]], dtype=np.float32)
y_train = np.array([[2.6], [1.1], [1.07], [2.68], [1.17]], dtype=np.float32)

w = Tensor.uniform(1, 1)
b = Tensor.zeros(1)

def forward(x):
    return x @ w + b

# train model
for i in range(1000):
    w.grad, b.grad = None, None
    y_pred = forward(Tensor(x_train))
    loss = ((y_pred - Tensor(y_train)) ** 2).mean()
    loss.backward()

    if w.grad is not None and b.grad is not None:
        w -= w.grad * 0.01
        b -= b.grad * 0.01
    


# predict next step time
next_step = np.array([[6]], dtype=np.float32)
predicted_time = forward(Tensor(next_step)).numpy()

print("Predicted duration for the next step: ", predicted_time[0][0])