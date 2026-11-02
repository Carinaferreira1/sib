import numpy as np
from si.neural_networks.layers import DenseLayer


class NoUpdate:
    def update(self, w, grad):
        return w


mse = lambda Y, P: np.mean((Y - P) ** 2)
mse_derivative = lambda Y, P: 2 * (P - Y) / Y.size

np.random.seed(0)

X = np.random.rand(4, 3)
Y = np.random.rand(4, 2)

layer = DenseLayer(2, (3,)).initialize(NoUpdate())
out = layer.forward_propagation(X, training=True)

analytic = X.T @ mse_derivative(Y, out)

numeric = np.zeros_like(layer.weights)
eps = 1e-6

for i in range(3):
    for j in range(2):
        layer.weights[i, j] += eps
        lp = mse(Y, layer.forward_propagation(X, True))

        layer.weights[i, j] -= 2 * eps
        lm = mse(Y, layer.forward_propagation(X, True))

        layer.weights[i, j] += eps
        numeric[i, j] = (lp - lm) / (2 * eps)

print(np.abs(numeric - analytic).max())