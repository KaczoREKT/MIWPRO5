import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Funkcje aktywacji i ich pochodne
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - tanh(x) ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Mapowanie nazw na funkcje
activation_functions = {
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative)
}

# Wczytywanie danych z pliku
def load_data():
    parser = argparse.ArgumentParser(description='Trenowanie sieci neuronowej.')
    parser.add_argument('-s', type=argparse.FileType('r'), required=True, help='Ścieżka do pliku np. -s Dane/daneXX.txt')
    parser.add_argument('--activation', choices=['relu', 'tanh', 'sigmoid'], default='relu', help='Funkcja aktywacji')
    parser.add_argument('--mode', choices=['batch', 'online'], default='batch', help='Tryb uczenia')
    args = parser.parse_args()

    data = [line.strip().split() for line in args.s if line.strip()]
    X = np.array([[float(x)] for x, _ in data])
    y = np.array([[float(y)] for _, y in data])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train.T, X_test.T, y_train.T, y_test.T, args.activation, args.mode

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, activation_name, mode = load_data()

    # Parametry sieci
    S1 = 50
    lr = 0.005
    epochs = 400

    # Inicjalizacja wag
    W1 = np.random.rand(S1, 1) - 0.5
    B1 = np.random.rand(S1, 1) - 0.5
    W2 = np.random.rand(1, S1) - 0.5
    B2 = np.random.rand(1, 1) - 0.5

    activation, activation_deriv = activation_functions[activation_name]

    for epoch in range(1, epochs + 1):
        if mode == 'batch':
            # Forward
            Z1 = W1 @ X_train + B1 @ np.ones((1, X_train.shape[1]))
            A1 = activation(Z1)
            A2 = W2 @ A1 + B2

            # Błąd i propagacja
            E2 = y_train - A2
            E1 = W2.T @ E2 * activation_deriv(Z1)

            # Aktualizacja wag
            W2 += lr * E2 @ A1.T
            B2 += lr * E2 @ np.ones((E2.shape[1], 1))
            W1 += lr * E1 @ X_train.T
            B1 += lr * E1 @ np.ones((X_train.shape[1], 1))
        elif mode == 'online':
            for i in range(X_train.shape[1]):
                x = X_train[:, i:i+1]
                y = y_train[:, i:i+1]

                z1 = W1 @ x + B1
                a1 = activation(z1)
                a2 = W2 @ a1 + B2

                e2 = y - a2
                e1 = W2.T @ e2 * activation_deriv(z1)

                W2 += lr * e2 @ a1.T
                B2 += lr * e2
                W1 += lr * e1 @ x.T
                B1 += lr * e1


        # Wizualizacja co 10 epok
        if epoch % 10 == 0:
            Z1_test = W1 @ X_test + B1 @ np.ones((1, X_test.shape[1]))
            A1_test = activation(Z1_test)
            A2_test = W2 @ A1_test + B2

            plt.clf()
            # Sortuj dane według X_test, by wykres miał sensowny kształt
            sorted_indices = np.argsort(X_test.flatten())
            X_sorted = X_test.flatten()[sorted_indices]
            y_sorted = y_test.flatten()[sorted_indices]
            A2_sorted = A2_test.flatten()[sorted_indices]

            plt.plot(X_sorted, y_sorted, 'ro', label='Target')
            plt.plot(X_sorted, A2_sorted, 'b-', label='Output')

            plt.title(f"Epoka {epoch} | Aktywacja: {activation_name} | Tryb: {mode}")
            plt.legend()
            plt.pause(0.05)

    plt.show()
