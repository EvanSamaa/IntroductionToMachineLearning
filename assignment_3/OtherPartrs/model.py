import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size, neuronCount = 20, activation = 0):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, neuronCount, bias=True)
        self.fc2 = nn.Linear(neuronCount, 1, bias=True)
        if activation == 0:
            self.activationFunc = F.relu
        elif activation == 1:
            self.activationFunc = F.tanh
        else:
            self.activationFunc = F.sigmoid
    def forward(self, features):
        x = self.fc1(features)
        x = self.activationFunc(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x
