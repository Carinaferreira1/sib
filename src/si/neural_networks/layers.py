from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):

    def set_input_shape(self, input_shape):
        self._input_shape = input_shape

    def input_shape(self):
        return self._input_shape

    def layer_name(self):
        return self.__class__.__name__

    @abstractmethod
    def forward_propagation(self, input, training):
        pass

    @abstractmethod
    def backward_propagation(self, output_error):
        pass

    @abstractmethod
    def output_shape(self):
        pass

    @abstractmethod
    def parameters(self):
        pass

import numpy as np


class DenseLayer(Layer):

    def __init__(self, n_units, input_shape=None):
        self.n_units = n_units
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def initialize(self, optimizer=None):
        n_features = self.input_shape()[0]

        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))

        self.w_opt = optimizer
        self.b_opt = optimizer

        return self

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        bias_error = np.sum(output_error, axis=0, keepdims=True)

        self.weights = self.w_opt.update(self.weights, weights_error)
        self.biases = self.b_opt.update(self.biases, bias_error)

        return input_error

    def output_shape(self):
        return (self.n_units,)

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

class Dropout(Layer):

    def __init__(self, probability, input_shape=None):
        if not 0 <= probability < 1:
            raise ValueError("Probability must be between 0 and 1")

        self.probability = probability
        self.mask = None
        self.input = None
        self.output = None

        if input_shape is not None:
            self.set_input_shape(input_shape)

    def forward_propagation(self, input, training=False):
        self.input = input

        if training:
            scaling_factor = 1 / (1 - self.probability)
            self.mask = np.random.binomial(
                1, 1 - self.probability, size=input.shape
            )
            self.output = input * self.mask * scaling_factor
        else:
            self.mask = None
            self.output = input

        return self.output

    def backward_propagation(self, output_error):
        if self.mask is None:
            return output_error

        scaling_factor = 1 / (1 - self.probability)
        return output_error * self.mask * scaling_factor

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0