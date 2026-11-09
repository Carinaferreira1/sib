from abc import abstractmethod

from si.neural_networks.layers import Layer


class ActivationLayer(Layer):

    def __init__(self, input_shape=None):
        self.input = None
        self.output = None

        if input_shape is not None:
            self.set_input_shape(input_shape)

    @abstractmethod
    def activation_function(self, x):
        pass

    @abstractmethod
    def derivative(self, x):
        pass

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = self.activation_function(input)
        return self.output

    def backward_propagation(self, output_error):
        return output_error * self.derivative(self.input)

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0


import numpy as np


class ReLUActivation(ActivationLayer):

    def activation_function(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)

class SigmoidActivation(ActivationLayer):

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        s = self.activation_function(x)
        return s * (1 - s)

class TanhActivation(ActivationLayer):

    def activation_function(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2

class SoftmaxActivation(ActivationLayer):

    def activation_function(self, x):
        x = np.asarray(x)
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def derivative(self, x):
        s = self.activation_function(x)
        return s * (1 - s)