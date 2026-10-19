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

    def initialize(self):
        n_features = self.input_shape()[0]

        self.weights = np.random.rand(n_features, self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))

        return self

    def forward_propagation(self, input, training=False):
        self.input = input
        self.output = np.dot(input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        return np.dot(output_error, self.weights.T)

    def output_shape(self):
        return (self.n_units,)

    def parameters(self):
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)