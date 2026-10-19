import numpy as np

X = np.random.rand(5, 2)

W1 = np.random.rand(2, 3) - 0.5
b1 = np.zeros((1, 3))

W2 = np.random.rand(3, 1) - 0.5
b2 = np.zeros((1, 1))

H = np.dot(X, W1) + b1
Y = np.dot(H, W2) + b2

print("X:", X.shape)
print("W1:", W1.shape)
print("H:", H.shape)
print("W2:", W2.shape)
print("Y:", Y.shape)