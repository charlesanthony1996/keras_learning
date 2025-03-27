import numpy as np

np.random.seed(42)

def generate_data(num_samples=1000, seq_len=3):
    x = np.random.randint(0, 10, size=(num_samples, seq_len))
    y = x * 2
    return x, y

# initializing weights for encoder and decoder (simple linear layer: Wx)
seq_len = 3
input_dim = seq_len
hidden_dim = seq_len

w_encoder = np.random.randn(hidden_dim, input_dim) * 0.1
w_decoder = np.random.randn(seq_len, hidden_dim) * 0.1

# params
learning_rate = 0.0001
epochs = 200

x_train, y_train = generate_data()

for epoch in range(epochs):
    total_loss = 0
    for x, y_true in zip(x_train, y_train):
        x = x.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)

        # forward pass
        encoded = w_encoder @ x
        y_pred = w_decoder @ encoded

        # loss mse
        loss = np.mean((y_pred - y_true) ** 2)
        total_loss += loss

        # backpropagation (gradient descent)
        dl_dy = 2 * (y_pred - y_true) / seq_len
        dl_dw_decoder = dl_dy @ encoded.T
        dl_dencoded = w_decoder.T @ dl_dy
        dl_dw_encoder = dl_dencoded @ x.T

        # update weights
        w_decoder -= learning_rate * dl_dw_decoder
        w_encoder -= learning_rate * dl_dw_encoder


    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss = {total_loss / len(x_train):.4f}")


# test 
test_input = np.array([7, 8, 9]).reshape(-1, 1)
encoded = w_encoder @ test_input
output = w_decoder @ encoded

print("Test input: ", test_input.ravel())
print("Expected: ", (test_input * 2).ravel())
print("Predicted: ", (output.ravel(), 2))
