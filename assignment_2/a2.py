# Assignment 2 skeleton code
# This code shows you how to use the 'argparse' library to read in parameters

import argparse
import numpy as np
from matplotlib import pyplot
import random
from dispkernel import dispKernel

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


class Neuron:
    learningRate = 0
    numOfEpoch = 0
    def __init__(self, activationFunction, activationFunctionDeriv):
        # will generate weight and bias and that's it
        self.weight = np.zeros([9,])
        for i in range (0, self.weight.shape[0]):
            self.weight[i] = random.uniform(0,1)
        self.bias = random.uniform(0,1)
        self.activationFunc = activationFunction
        self.activationFuncDeriv = activationFunctionDeriv
    # will call generateOutput
    def __call__(self, inputData):
        self.generateOutput(self, inputData)
    # take in one input (9 by 1 array) and generate the output that includes the activation function
    def generateOutput(self, inputData):
        npArrInputData = np.array(inputData)
        output = self.activationFunc(npArrInputData.dot(self.weight) + self.bias)
        return output
    # will calculate the error, the input is gonan be the z value
    def calculateLoss(self, inputData, inputLabel):
        z = self.generateOutput(inputData)
        error = np.square(z - inputLabel)
        return error
    # will calculate the total error of the entire epoch
    def calculateLossAndAccuracyPerEpoch(self, inputData, inputLabel):
        totalErorr = 0
        totalAccuracy = 0
        npArrInputData = np.array(inputData)
        npArrInputLabel = np.array(inputLabel)
        for i in range (0, npArrInputLabel.shape[0]):
            totalErorr += self.calculateLoss(npArrInputData[i], npArrInputLabel[i])
            if npArrInputLabel[i] == 1:
                if self.generateOutput(npArrInputData[i]) > 0.5:
                    totalAccuracy = totalAccuracy + 1
            else:
                if self.generateOutput(npArrInputData[i]) <= 0.5:
                    totalAccuracy = totalAccuracy + 1
        return [totalErorr, totalAccuracy/inputLabel.shape[0]]
    # will calculate the dloss/dwi function and return a narray, input a [9,] array and number respectively
    def calculatedLdw(self, inputData, inputLabel):
        z = inputData.dot(self.weight) + self.bias
        rtv = inputData * self.activationFuncDeriv(z) * 2 * (self.activationFunc(z) - inputLabel)
        return rtv
    # generate the dL/db for bias
    def calculatedLdb(self, inputData, inputLabel):
        z = inputData.dot(self.weight) + self.bias
        rtv = 1 * self.activationFuncDeriv(z) * 2 * (self.activationFunc(z) - inputLabel)
        return rtv
    # perform training on a dataset and adjust the parameters as needed
    def trainingPerEpoch (self, inputData, inputLabel):
        npArrInputData = np.array(inputData)
        npArrInputLabel = np.array(inputLabel)
        totaldLdw = np.zeros(npArrInputData[0].shape)
        totaldLdb = 0
        for i in range(0, npArrInputData.shape[0]):
            totaldLdw = totaldLdw + self.calculatedLdw(npArrInputData[i], npArrInputLabel[i])
            totaldLdb += self.calculatedLdb(npArrInputData[i], npArrInputLabel[i])
        totaldLdw = totaldLdw/npArrInputData.shape[0]
        totaldLdb = totaldLdb/npArrInputData.shape[0]
        self.weight -= totaldLdw*self.learningRate
        self.bias -= totaldLdb*self.learningRate
    # perform x epochs of training
    def training (self, trainingData, trainingLabel, validationData, validationLabel):
        finalResults = np.zeros([5,0])
        for i in range (0, self.numOfEpoch):
            lossAndAccuracy = self.calculateLossAndAccuracyPerEpoch(trainingData, trainingLabel)
            validationSetLossAndAccuracy = self.calculateLossAndAccuracyPerEpoch(validationData, validationLabel)
            results = np.array([[i, lossAndAccuracy[0], lossAndAccuracy[1], validationSetLossAndAccuracy[0], validationSetLossAndAccuracy[1]]])
            finalResults = np.concatenate((finalResults, results.T), axis = 1)
            # final = np.append(final, results, axis = 0)
            self.trainingPerEpoch(trainingData, trainingLabel)


        # print(finalResults)
        pyplot.plot(finalResults[0], finalResults[1]/200, label="training set")
        pyplot.plot(finalResults[0], finalResults[3]/20, label="validation set")
        pyplot.title("Average loss vs Epoch")
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Loss")
        pyplot.legend(loc='lower right')
        pyplot.show()
        pyplot.plot(finalResults[0], finalResults[2], label="training set")
        pyplot.plot(finalResults[0], finalResults[4], label="validation set")
        pyplot.title("Average Accuracy vs Epoch")
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Accuracy")
        pyplot.legend(loc='lower right')
        pyplot.show()
        dispKernel(self.weight, 3, 200)
        # print(self.weight)
        # print(self.bias)
        # print(self.calculateLossAndAccuracyPerEpoch(trainingData, trainingLabel))
        # print(self.calculateLossAndAccuracyPerEpoch(validationData, validationLabel))
        return [self.calculateLossAndAccuracyPerEpoch(trainingData, trainingLabel),
                self.calculateLossAndAccuracyPerEpoch(validationData, validationLabel), finalResults]




# write the activation functions and their derivatives for later uses
def activationFunctionLinear(x):
    return x
def activationFunctionLinearDeriv(x):
    return 1
def activationFunctionSigmoid(x):
    rtv = 1/(1+np.exp(-x))
    return rtv
def activationFunctionSigmoidDeriv(x):
    rtv = (1/(1+np.exp(-x)))*(1-(1/(1+np.exp(-x))))
    return rtv
def activationFunctionRelu(x):
    if x > 0 :
        return x
    else :
        return 0
def activationFunctionReluDeriv(x):
    if x > 0:
        return 1
    else:
        return 0
# load training data
trainingData = np.loadtxt(traindataname, delimiter=',')
trainingDataLabel = np.loadtxt(trainlabelname, delimiter=',')
validationData = np.loadtxt(validdataname, delimiter=',')
validationDataLabel = np.loadtxt(validlabelname, delimiter=',')
# initializing the neuron with activation function and learning rate
random.seed(0)
tim = Neuron(activationFunctionLinear, activationFunctionLinearDeriv)
if args.actfunction == 'sigmoid':
    tim = Neuron(activationFunctionSigmoid, activationFunctionSigmoidDeriv)
elif args.actfunction == 'relu':
    tim = Neuron(activationFunctionRelu, activationFunctionReluDeriv)
tim.learningRate = args.learningrate
tim.numOfEpoch = args.numepoch
tim.training(trainingData, trainingDataLabel, validationData, validationDataLabel)

def testRandomSeed():
    listOflearningRates = np.array([0, 1, 2, 3, 4])
    resultMatrix = np.zeros([listOflearningRates.shape[0], 1, 2])
    graphsToPLot = []
    for i in range(0, listOflearningRates.shape[0]):
        random.seed(listOflearningRates[i])
        tim = Neuron(activationFunctionLinear, activationFunctionLinearDeriv)
        if args.actfunction == 'sigmoid':
            tim = Neuron(activationFunctionSigmoid, activationFunctionSigmoidDeriv)
        elif args.actfunction == 'relu':
            tim = Neuron(activationFunctionRelu, activationFunctionReluDeriv)
        tim.numOfEpoch = 1000
        tim.learningRate = 0.01
        result = tim.training(trainingData, trainingDataLabel, validationData, validationDataLabel)
        resultMatrix[i, 0, 0] = result[0][1]
        resultMatrix[i, 0, 1] = result[1][1]
        graphsToPLot.append(result[2])
    np.savetxt("seeEffetOfRandomSeed_training.csv", resultMatrix[:, :, 0])
    np.savetxt("seeEffetOfRandomSeed_validation.csv", resultMatrix[:, :, 1])
    for i in range(0, len(graphsToPLot)):
        # pyplot.plot(item[0], item[1])
        pyplot.title(
            "Accuracy of training and validation set for " + "Activation Function = " + str(listOflearningRates[i]))
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Accuracy")
        pyplot.plot(graphsToPLot[i][0], graphsToPLot[i][2], label="training set")
        pyplot.plot(graphsToPLot[i][0], graphsToPLot[i][4], label="validation set")
        pyplot.legend(loc='upper left')
        pyplot.show()
    print(resultMatrix[:, :, 1])
def testActivationFunction():
    listOflearningRates = np.array(['relu', 'sigmoid', 'linear'])
    resultMatrix = np.zeros([listOflearningRates.shape[0], 1, 2])
    graphsToPLot = []
    for i in range(0, listOflearningRates.shape[0]):
        tim = Neuron(activationFunctionLinear, activationFunctionLinearDeriv)
        if listOflearningRates[i] == 'sigmoid':
            tim = Neuron(activationFunctionSigmoid, activationFunctionSigmoidDeriv)
        elif listOflearningRates[i] == 'relu':
            tim = Neuron(activationFunctionRelu, activationFunctionReluDeriv)
        tim.numOfEpoch = 1000
        tim.learningRate = 0.01
        result = tim.training(trainingData, trainingDataLabel, validationData, validationDataLabel)
        resultMatrix[i, 0, 0] = result[0][1]
        resultMatrix[i, 0, 1] = result[1][1]
        graphsToPLot.append(result[2])
    np.savetxt("seeEffetOfActivationFunc_training.csv", resultMatrix[:, :, 0])
    np.savetxt("seeEffetOfActivationFunc_validation.csv", resultMatrix[:, :, 1])
    for i in range(0, len(graphsToPLot)):
        # pyplot.plot(item[0], item[1])
        pyplot.title("Accuracy of training and validation set for " + "Activation Function = " + str(listOflearningRates[i]))
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Accuracy")
        pyplot.plot(graphsToPLot[i][0], graphsToPLot[i][2], label="training set")
        pyplot.plot(graphsToPLot[i][0], graphsToPLot[i][4], label="validation set")
        pyplot.legend(loc='upper left')
        pyplot.show()
    print(resultMatrix[:, :, 1])
def testEpochs():
    random.seed(0)
    listOflearningRates = np.array([10, 100, 1000, 5000, 10000])
    # listOflearningRates = np.array([0.001, 0.1, 0.3, 0.5])
    resultMatrix = np.zeros([listOflearningRates.shape[0], 1, 2])
    graphsToPLot = []
    for i in range(0, listOflearningRates.shape[0]):
        tim = Neuron(activationFunctionLinear, activationFunctionLinearDeriv)
        if args.actfunction == 'sigmoid':
            tim = Neuron(activationFunctionSigmoid, activationFunctionSigmoidDeriv)
        elif args.actfunction == 'relu':
            tim = Neuron(activationFunctionRelu, activationFunctionReluDeriv)
        tim.numOfEpoch = listOflearningRates[i]
        tim.learningRate = 0.1
        result = tim.training(trainingData, trainingDataLabel, validationData, validationDataLabel)
        resultMatrix[i, 0, 0] = result[0][1]
        resultMatrix[i, 0, 1] = result[1][1]
        graphsToPLot.append(result[2])
    np.savetxt("seeEffetOfNumEpoch_training.csv", resultMatrix[:,:,0])
    np.savetxt("seeEffetOfNumEpoch_validation.csv", resultMatrix[:, :, 1])
    for i in range(0, len(graphsToPLot)):
        # pyplot.plot(item[0], item[1])
        pyplot.title("Accuracy of training and validation set for " + "NumOfEpoch = " + str(listOflearningRates[i]))
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Accuracy")
        pyplot.plot(graphsToPLot[i][0], graphsToPLot[i][2], label="training set")
        pyplot.plot(graphsToPLot[i][0], graphsToPLot[i][4], label="validation set")
        pyplot.legend(loc='upper left')
        pyplot.show()
    print(resultMatrix[:,:,1])
def testingLearninRates():
    random.seed(0)
    listOflearningRates = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1])
    # listOflearningRates = np.array([0.001, 0.1, 0.3, 0.5])
    resultMatrix = np.zeros([listOflearningRates.shape[0], 1, 2])
    graphsToPLot = []
    for i in range(0, listOflearningRates.shape[0]):
        tim = Neuron(activationFunctionLinear, activationFunctionLinearDeriv)
        if args.actfunction == 'sigmoid':
            tim = Neuron(activationFunctionSigmoid, activationFunctionSigmoidDeriv)
        elif args.actfunction == 'relu':
            tim = Neuron(activationFunctionRelu, activationFunctionReluDeriv)
        tim.learningRate = listOflearningRates[i]
        tim.numOfEpoch = 2000
        result = tim.training(trainingData, trainingDataLabel, validationData, validationDataLabel)
        resultMatrix[i, 0, 0] = result[0][1]
        resultMatrix[i, 0, 1] = result[1][1]
        graphsToPLot.append(result[2])
    print(graphsToPLot)
    np.savetxt("seeEffetOfLearningRate_training.csv", resultMatrix[:,:,0])
    np.savetxt("seeEffetOfLearningRate_validation.csv", resultMatrix[:, :, 1])
    for i in range(0, len(graphsToPLot)):
        # pyplot.plot(item[0], item[1])
        pyplot.title("Accuracy of training and validation set for " + "LR = " + str(listOflearningRates[i]))
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Accuracy")
        pyplot.plot(graphsToPLot[i][0], graphsToPLot[i][2], label="training set")
        pyplot.plot(graphsToPLot[i][0], graphsToPLot[i][4], label="validation set")
        pyplot.legend(loc='upper left')
        pyplot.show()
    print(resultMatrix)
def testingHyperParam():
    # listOflearningRates = np.arange(0, 0.5, 0.01)
    listOflearningRates = np.array([0.05, 0.06, 0.07, 0.08, 0.09])
    listOfEpoch = np.arange(10, 150, 10)
    # listOfEpoch = np.array([100, 500, 1000, 5000, 10000])
    resultMatrix = np.zeros([listOflearningRates.shape[0], listOfEpoch.shape[0], 2])
    for i in range (0, listOflearningRates.shape[0]):
        for j in range (0, listOfEpoch.shape[0]):
            tim = Neuron(activationFunctionLinear, activationFunctionLinearDeriv)
            if args.actfunction == 'sigmoid':
                tim = Neuron(activationFunctionSigmoid, activationFunctionSigmoidDeriv)
            elif args.actfunction == 'relu':
                tim = Neuron(activationFunctionRelu, activationFunctionReluDeriv)
            tim.learningRate = listOflearningRates[i]
            tim.numOfEpoch = listOfEpoch[j]
            result = tim.training(trainingData, trainingDataLabel, validationData, validationDataLabel)
            print(result)
            resultMatrix[i, j, 0] = result[0][1]
            resultMatrix[i, j, 1] = result[1][1]
    np.savetxt("optimizeResult.csv", resultMatrix[:, :, 1])

# testingHyperParam()