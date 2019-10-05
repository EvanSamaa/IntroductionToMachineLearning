import numpy as np

class ElementwiseMultiply():
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, vectorInput):
        if vectorInput.shape == self.weight.shape:
            return vectorInput * self.weight

class AddBias():
    def __init__(self, bias):
        self.bias = bias
    def __call__(self, input):
        return self.bias + input

class LeakyRelu():
    def __init__(self, alpha):
        self.alpha = alpha
    def __call__(self, input):
        return np.where(input >= 0, input, self.alpha * input)

class Compose():
    def __init__(self, layers):
        self.layers = layers
    def __call__(self, input):
        for item in self.layers:
            input = item(input)
        return input
