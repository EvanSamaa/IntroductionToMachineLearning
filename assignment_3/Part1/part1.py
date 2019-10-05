import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot
import random

# Command Line Arguments
parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
parser.add_argument('trainingfile', help='name stub for training data and label output in csv format',default="train")
parser.add_argument('validationfile', help='name stub for validation data and label output in csv format',default="valid")
parser.add_argument('numtrain', help='number of training samples',type= int,default=200)
parser.add_argument('numvalid', help='number of validation samples',type= int,default=20)
parser.add_argument('-seed', help='random seed', type= int,default=1)
parser.add_argument('-learningrate', help='learning rate', type= float,default=0.1)
parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'],default='linear')
parser.add_argument('-numepoch', help='number of epochs', type= int,default=50)

args = parser.parse_args()

traindataname = args.trainingfile + "data.csv"
trainlabelname = args.trainingfile + "label.csv"

print("training data file name: ", traindataname)
print("training label file name: ", trainlabelname)

validdataname = args.validationfile + "data.csv"
validlabelname = args.validationfile + "label.csv"

print("validation data file name: ", validdataname)
print("validation label file name: ", validlabelname)

print("number of training samples = ", args.numtrain)
print("number of validation samples = ", args.numvalid)

print("learning rate = ", args.learningrate)
print("number of epoch = ", args.numepoch)

print("activation function is ",args.actfunction)
class SNC(nn.Module):
    def __init__(self, activationFunc):
        super(SNC, self).__init__()
        self.activationFunc = activationFunc
        self.fc1 = nn.Linear(9, 1)

    def forward(self, input):
        x = self.fc1(input)
        if self.activationFunc == "relu":
            x = F.relu(x)
            return x
        elif self.activationFunc == "sigmoid":
            return F.sigmoid(x)
        return x
def calculateAccuracy(result, labels):
    counter = 0
    for i in range (0, labels.size()[0]):
        if labels[i].item() == 0:
            if result[i] < 0.5:
                counter = counter + 1
        else:
            if result[i] >= 0.5:
                counter = counter + 1
    return counter/result.size()[0]



# load training data
trainingDataT = torch.FloatTensor(np.loadtxt(traindataname, dtype=np.single, delimiter=','))
trainingDataLabelT = torch.FloatTensor(np.loadtxt(trainlabelname, dtype=np.single, delimiter=','))
validationDataT = torch.FloatTensor(np.loadtxt(validdataname, dtype=np.single, delimiter=','))
validationDataLabelT = torch.FloatTensor(np.loadtxt(validlabelname, dtype=np.single, delimiter=','))
learningRate = args.learningrate
numEpoch = args.numepoch
# initializing the neuron with activation function and learning rate
torch.manual_seed(12)

miniSNC = SNC(args.actfunction)
# set up loss function and optimizer
lossFunction = torch.nn.MSELoss()
optimizer = torch.optim.SGD(miniSNC.parameters(), lr = learningRate)

# set up ways to graph the trend
lossRecord = []
vlossRecord = []
accuracyRecord = []
vaccuracyRecord = []
nRec = []
# training
for i in range (0, numEpoch):
    optimizer.zero_grad()
    predict = miniSNC(trainingDataT)
    predictV = miniSNC(validationDataT)
    lossV = lossFunction(input = predictV.squeeze(), target = validationDataLabelT.float())
    accV = calculateAccuracy(predictV, validationDataLabelT)
    loss = lossFunction(input = predict.squeeze(), target = trainingDataLabelT.float()) # this calculates the loss
    acc = calculateAccuracy(predict, trainingDataLabelT)
    # recording data for plotting
    lossRecord.append(loss/trainingDataLabelT.size()[0])
    vlossRecord.append(lossV/validationDataLabelT.size()[0])
    accuracyRecord.append(acc)
    vaccuracyRecord.append(accV)
    nRec.append(i)

    # finding gradients and updating weights
    loss.backward() # this calculates the gradients
    optimizer.step()
    print("the loss is " + str(loss) + " and the accuracy is " + str(acc))

predict = miniSNC(trainingDataT)
predictV = miniSNC(validationDataT)
lossV = lossFunction(input = predictV.squeeze(), target = validationDataLabelT.float())
accV = calculateAccuracy(predictV, validationDataLabelT)
loss = lossFunction(input = predict.squeeze(), target = trainingDataLabelT.float()) # this calculates the loss
acc = calculateAccuracy(predict, trainingDataLabelT)
accV = calculateAccuracy(predictV, validationDataLabelT)
print("The final accuracy of the SNC for the validation data is " + str(accV))
print("The final accuracy of the SNC for the training data is " + str(acc))
print("The final loss of the SNC for the validation data is " + str(lossV))
print("The final loss of the SNC for the training data is " + str(loss))
pyplot.subplot(2, 1, 1)
pyplot.plot(nRec, lossRecord, label="training set")
pyplot.plot(nRec, vlossRecord, label="validation set")
pyplot.title("Average loss vs Epoch")
pyplot.ylabel("Loss")
pyplot.legend(loc='lower right')
pyplot.subplot(2, 1, 2)
pyplot.plot(nRec, accuracyRecord, label="training set")
pyplot.plot(nRec, vaccuracyRecord, label="validation set")
pyplot.title("Average Accuracy vs Epoch")
pyplot.ylabel("Accuracy")
pyplot.legend(loc='lower right')
pyplot.xlabel("Epochs")
pyplot.show()

