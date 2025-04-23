import argparse
import numpy as np
from sklearn.model_selection import train_test_split


# Aktywacje
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

# Funkcja straty (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Inicjalizacja
def init_params(n_input, n_hidden, n_output):
    return {
        'W1': np.random.randn(n_input, n_hidden) * 0.1,
        'W2': np.random.randn(n_hidden, n_hidden) * 0.1,
        'W3': np.random.randn(n_hidden, n_output) * 0.1,
        'b1': np.zeros((1, n_hidden)),
        'b2': np.zeros((1, n_hidden)),
        'b3': np.zeros((1, n_output))
    }

# Forward
def forward(X, params):
    Z1 = X @ params['W1'] + params['b1']
    A1 = tanh(Z1)
    Z2 = A1 @ params['W2'] + params['b2']
    A2 = tanh(Z2)
    Z3 = A2 @ params['W3'] + params['b3']
    output = Z3
    cache = X, Z1, Z2, A1, A2, Z3, output
    return output, cache

# Backward
def backward(y, params, cache):
    X, Z1, Z2, A1, A2, Z3, output = cache
    m = y.shape[0]

    # Wyjściowa warstwa
    dZ3 = (output - y)  # MSE: ∂L/∂Z3
    dW3 = A2.T @ dZ3 / m
    db3 = np.sum(dZ3, axis=0, keepdims=True) / m

    # Warstwa ukryta 2
    dA2 = dZ3 @ params['W3'].T
    dZ2 = dA2 * tanh_deriv(Z2)
    dW2 = A1.T @ dZ2 / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Warstwa ukryta 1
    dA1 = dZ2 @ params['W2'].T
    dZ1 = dA1 * tanh_deriv(Z1)
    dW1 = X.T @ dZ1 / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return {
        'dW1': dW1, 'dW2': dW2, 'dW3': dW3,
        'db1': db1, 'db2': db2, 'db3': db3
    }

# Aktualizacja
def update_params(params, grads, lr):
    for key in grads:
        params[key[:-1]] -= lr * grads[key]
    return params

# Trening
def train(X_train, y_train, n_hidden=10, epochs=2000, lr=0.01):
    n_input = X_train.shape[1]
    n_output = y_train.shape[1]
    params = init_params(n_input, n_hidden, n_output)

    for epoch in range(epochs):
        y_pred, cache = forward(X_train, params)
        loss = mse(y_train, y_pred)
        grads = backward(y_train, params, cache)
        params = update_params(params, grads, lr)

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | MSE: {loss:.5f}")

    return params

def predict(X, params):
    output, _ = forward(X, params)
    return output

# --- Wczytywanie danych ---
parser = argparse.ArgumentParser(description='Wczytuje dane.',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-s", type=argparse.FileType('r'), help="np. -s Dane/daneXX.txt")
args = vars(parser.parse_args())

data = [line.strip().split() for line in args['s'] if line.strip()]
X = np.array([[float(x)] for x, _ in data])  # 2D: (n_samples, n_features)
y = np.array([[float(y)] for _, y in data])  # 2D: (n_samples, 1)

# --- Podział na trening/test ---
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Normalizacja na podstawie zbioru treningowego
X_mean, X_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

X_train_norm = (X_train - X_mean) / X_std
y_train_norm = (y_train - y_mean) / y_std

X_test_norm = (X_test - X_mean) / X_std
y_test_norm = (y_test - y_mean) / y_std

params = train(X_train_norm, y_train_norm, n_hidden=30, epochs=3000, lr=0.01)


y_pred_test_norm = predict(X_test_norm, params)
y_pred_test = y_pred_test_norm * y_std + y_mean

# MSE na zbiorze testowym
from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, y_pred_test)
print("MSE (test):", mse_test)