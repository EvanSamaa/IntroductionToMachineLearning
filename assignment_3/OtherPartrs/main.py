import argparse
from time import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from matplotlib import pyplot

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *
import time


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# =================================== LOAD DATASET =========================================== #

data = pd.read_csv(filepath_or_buffer = "adult.csv")
# =================================== DATA VISUALIZATION =========================================== #


# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

# print("-------------------- 3.2 ---------------------")
# print(data.shape) # returns data in [#row, #column],
# print(data.columns)
# verbose_print(data.head())
# print(data["income"].value_counts())

# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    pass
# print("-------------------- 3.3 ---------------------")
# print(data.isin(["?"]).sum())
# print(data.shape)
for column in data.columns:
    data = data[data[column] != "?"]
# print(data.shape)
# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

# =================================== BALANCE DATASET =========================================== #

# print("-------------------- 3.4 ---------------------")
# print(data["income"].value_counts())
minVal = data["income"].value_counts()[0]
if data["income"].value_counts()[1] <= minVal:
    minVal = data["income"].value_counts()[1]  # find which one is smaller
highIncomeData = data[data["income"] == ">50K"]
lowIncomeData = data[data["income"] == "<=50K"]
lowIncomeData = lowIncomeData.sample(n=minVal, random_state=1)
highIncomeData = highIncomeData.sample(n=minVal, random_state=1)
data = pd.concat([lowIncomeData, highIncomeData])
# print(data["income"].value_counts())


# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

# print("-------------------- 3.5 ---------------------")
# verbose_print(data["age"].describe())
# verbose_print(data["hours-per-week"].describe())


# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

# visualize the first 3 features using pie and bar graphs
categorical_3_feats = categorical_feats
# for feature in categorical_3_feats:
    # pie_chart(data, feature)
    # binary_bar_chart(data, feature)


# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# NORMALIZE CONTINUOUS FEATURES
# print(data.columns)
ctsFeatures = ["age", "educational-num", "capital-gain", "capital-loss", "hours-per-week", "fnlwgt"]
ctsData = data[ctsFeatures]  # currently in numpy array (22416, 6)
discreteData = data[categorical_feats]
processedData = []
for column in ctsFeatures:
    temp = (ctsData[column].to_numpy() - ctsData[column].mean())/ctsData[column].std()
    processedData.append(temp)
processedData = np.array(processedData) # currently in numpy array (6, 22416)
# ENCODE CATEGORICAL FEATURES
label_encoder = LabelEncoder()
discreteData = discreteData.to_numpy().T
nplabelSet = discreteData[discreteData.shape[0]-1].copy()
npDiscreteData = np.empty([discreteData.shape[0]-1, discreteData.shape[1]])# shape = (feature, samples)
for i in range(0, discreteData.shape[0]-1):
    npDiscreteData[i] = label_encoder.fit_transform(discreteData[i])
# npDiscreteDate is now in shape = (feature, samples)
oneh_encoder = OneHotEncoder(categories='auto')
npDiscreteData = oneh_encoder.fit_transform(npDiscreteData.T).T # i want it back in shape=(feature, samples)
npDiscreteData = npDiscreteData.todense()
processedData = np.array(np.concatenate((processedData, npDiscreteData), axis=0).T)
# now deal with label
processedLabels = np.array(label_encoder.fit_transform(nplabelSet))
# Hint: .toarray() converts the DataFrame to a numpy array


# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter
trainingData, validationData, trainingLabel, validationLabel = train_test_split(processedData, processedLabels, test_size = 0.1, random_state=seed)
# print()

# =================================== LOAD DATA AND MODEL =========================================== #
def load_data(batch_size):
    trainDataSet = AdultDataset(trainingData, trainingLabel)
    validationDataSet = AdultDataset(validationData, validationLabel)
    train_loader = DataLoader(trainDataSet, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validationDataSet, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader
def load_model(lr, neuronNum = 20, activationFunc = 0):
    model = MultiLayerPerceptron(processedData.shape[1], neuronNum, activationFunc)
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, loss_fnc, optimizer
def evaluate(model, val_loader):
    total_corr = 0
    for i, batch in enumerate(val_loader):
        data, label = batch
        data = data.type(torch.FloatTensor)
        prediction = model(data)
        for j in range(0, len(prediction)):
            if label[j] == 1:
                if prediction[j] > 0.5:
                    total_corr = total_corr + 1
            elif label[j] == 0:
                if prediction[j] < 0.5:
                    total_corr = total_corr + 1
    # print(float(total_corr)/len(val_loader.dataset))
    return float(total_corr)/len(val_loader.dataset)
def activationFunctions():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()
    batch_size = args.batch_size
    numOfEpochs = args.epochs
    eval_every = args.eval_every
    lr = args.lr
    timeSpend = []
    nArrList = []
    nArrValidationList = []
    accTrainingList = []
    accValidationList = []
    for i in range (0, 3):
        torch.manual_seed(seed)
        training_loader, validation_loader = load_data(batch_size)
        model, loss_fnc, optimizer = load_model(lr=lr, neuronNum=20)
        nArr = []
        nArrValidation = []
        accTraining = []
        accValidation = []
        counter = 0
        training_loader, validation_loader = load_data(batch_size)
        model, loss_fnc, optimizer = load_model(lr=lr, activationFunc=i)
        torch.manual_seed(seed=seed)
        startTime = time.time()
        for epoch in range(0, numOfEpochs):
            for i, batch in enumerate(training_loader):
                accumCorr = 0
                feat, label = batch
                optimizer.zero_grad()
                feat = feat.type(torch.FloatTensor)
                prediction = model(feat)
                loss = loss_fnc(input=prediction.squeeze(), target=label.float())
                loss.backward()
                for j in range(0, len(prediction)):
                    if label[j] == 1:
                        if prediction[j] > 0.5:
                            accumCorr = accumCorr + 1
                    elif label[j] == 0:
                        if prediction[j] < 0.5:
                            accumCorr = accumCorr + 1
                accTraining.append(accumCorr / batch_size)
                nArr.append(time.time() - startTime)
                counter = counter + 1
                optimizer.step()
            if (epoch) % eval_every == 0:
                nArrValidation.append(time.time() - startTime)
                accuracy = evaluate(model, validation_loader)
                accValidation.append(accuracy)
        timeSpend.append(time.time()-startTime)
        nArrList.append(nArr)
        nArrValidationList.append(nArrValidation)
        accValidationList.append(accValidation)
        accTrainingList.append(accTraining)
    timeSpend = np.array(timeSpend)
    df = pd.DataFrame(timeSpend)
    df.to_excel("timeSpent.xlsx", index=False)
    # pyplot.subplot(2, 1, 1)
    # pyplot.plot(np.array(nArrList[0]), savgol_filter(np.array(accTrainingList[0]), 25, 2), label="relu")
    # pyplot.plot(np.array(nArrList[0]), savgol_filter(np.array(accTrainingList[1]), 25, 2), label="tanh")
    # pyplot.plot(np.array(nArrList[0]), savgol_filter(np.array(accTrainingList[2]), 25, 2), label="sigmoid")
    # pyplot.title("Training accuracy vs Step for different activation Functions")
    # pyplot.ylabel("Accuracy")
    # pyplot.legend(loc='lower right')
    # pyplot.subplot(2, 1, 2)
    # pyplot.plot(np.array(nArrValidationList[0]), np.array(accValidationList[0]), label="relu")
    # pyplot.plot(np.array(nArrValidationList[0]), np.array(accValidationList[1]), label="tanh")
    # pyplot.plot(np.array(nArrValidationList[0]), np.array(accValidationList[2]), label="sigmoid")
    # pyplot.title("Validation accuracy vs Step for different activation Functions")
    # pyplot.ylabel("Accuracy")
    # pyplot.legend(loc='lower right')
    # pyplot.xlabel("Step")
    # pyplot.show()

def gridSearch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()
    batch_size = args.batch_size
    numOfEpochs = args.epochs
    lr = args.lr
    eval_every = args.eval_every
    batch_sizeList = [10, 20, 100, 200, 500, 1000, 2000]
    numOfEpochsList = [1,3,5,10,20,30]
    accuracyList = []
    for batch_size in batch_sizeList:
        eachRow = []
        for numOfEpochs in numOfEpochsList:
            startTime = time.time()
            neuronNum = 20
            torch.manual_seed(seed)
            training_loader, validation_loader = load_data(batch_size)
            model, loss_fnc, optimizer = load_model(lr=lr, neuronNum=neuronNum)
            for epoch in range(0, numOfEpochs):
                for i, batch in enumerate(training_loader):
                    feat, label = batch
                    optimizer.zero_grad()
                    feat = feat.type(torch.FloatTensor)
                    prediction = model(feat)
                    loss = loss_fnc(input=prediction.squeeze(), target=label.float())
                    loss.backward()
                    optimizer.step()
            timeSpent = time.time()-startTime
            eachRow.append(timeSpent)
        accuracyList.append(eachRow)
    accuracyList = np.array(accuracyList)
    df = pd.DataFrame(accuracyList)
    df.to_excel("gridSearch_epochs_timeSpent.xlsx", index=False)

def searchLearningRate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--lr', type=float)
    args = parser.parse_args()
    batch_size = args.batch_size
    numOfEpochs = args.epochs
    lr = args.lr
    eval_every = args.eval_every
    batch_sizeList = [1, 100, 17932]
    numOfEpochsList = [5, 20, 40]
    nArrList = []
    nArrValidationList = []
    accTrainingList = []
    accValidationList = []
    for batch_size in batch_sizeList:
        if batch_size == 1:
            numOfEpochs = 5
        elif batch_size == 100:
            numOfEpochs = 20
        elif batch_size == 17932:
            numOfEpochs = 30
        nArr = []
        nArrValidation = []
        accTraining = []
        accValidation = []
        counter = 0
        training_loader, validation_loader = load_data(batch_size)
        model, loss_fnc, optimizer = load_model(lr=lr)
        torch.manual_seed(seed=seed)
        startTime = time.time()
        for epoch in range(0, numOfEpochs):
            for i, batch in enumerate(training_loader):
                accumCorr = 0
                feat, label = batch
                optimizer.zero_grad()
                feat = feat.type(torch.FloatTensor)
                prediction = model(feat)
                loss = loss_fnc(input=prediction.squeeze(), target=label.float())
                loss.backward()
                for j in range(0, len(prediction)):
                    if label[j] == 1:
                        if prediction[j] > 0.5:
                            accumCorr = accumCorr + 1
                    elif label[j] == 0:
                        if prediction[j] < 0.5:
                            accumCorr = accumCorr + 1
                accTraining.append(accumCorr/ batch_size)
                nArr.append(time.time()-startTime)
                counter = counter + 1
                optimizer.step()
            if (epoch) % eval_every == 0:
                nArrValidation.append(time.time()-startTime)
                accuracy = evaluate(model, validation_loader)
                accValidation.append(accuracy)

        accuracy_train = evaluate(model=model,val_loader=training_loader)
        accuracy_validation = evaluate(model=model,val_loader=validation_loader)
        nArrList.append(nArr)
        nArrValidationList.append(nArrValidation)
        accTrainingList.append(accTraining)
        accValidationList.append(accValidation)
    print(accValidationList[0][len(accValidationList[0]) - 1])
    print(accValidationList[1][len(accValidationList[1]) - 1])
    print(accValidationList[2][len(accValidationList[2]) - 1])
    pyplot.subplot(3, 1, 1)
    pyplot.plot(np.array(nArrList[0]), np.array(accTrainingList[0]), label="training")
    pyplot.plot(np.array(nArrValidationList[0]), np.array(accValidationList[0]), 'bo', label="validation")
    pyplot.title("Batch Size = 1")
    pyplot.ylabel("Accuracy")
    pyplot.legend(loc='lower right')
    pyplot.subplot(3, 1, 2)
    pyplot.plot(np.array(nArrList[1]), np.array(accTrainingList[1]), label="training")
    pyplot.plot(np.array(nArrValidationList[1]), np.array(accValidationList[1]), 'bo', label="validation")
    pyplot.title("Batch Size = 100")
    pyplot.ylabel("Accuracy")
    pyplot.legend(loc='lower right')
    pyplot.subplot(3, 1, 3)
    pyplot.plot(np.array(nArrList[2]), np.array(accTrainingList[2]), label="training")
    pyplot.plot(np.array(nArrValidationList[2]), np.array(accValidationList[2]), 'bo', label="validation")
    pyplot.title("Batch Size = 17932")
    pyplot.ylabel("Accuracy")
    pyplot.legend(loc='lower right')
    pyplot.xlabel("Time(sec)")
    pyplot.show()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--NumOfNeuronInLayer1', type=int, default=20)
    args = parser.parse_args()
    batch_size = args.batch_size
    numOfEpochs = args.epochs
    eval_every = args.eval_every
    numNeuron = args.NumOfNeuronInLayer1
    lr = args.lr
    torch.manual_seed(seed)
    training_loader, validation_loader = load_data(batch_size)
    model, loss_fnc, optimizer = load_model(lr=lr, neuronNum=numNeuron)
    nArr = []
    nArrValidation = []
    accTraining = []
    accValidation = []
    lossTraining = []
    lossValidation = []
    counter = 0
    typeOfError = [0, 0, 0, 0] # 0 is true positive, 1 is true negative, 2 is false positive, 3 is false negative
    for epoch in range (0, numOfEpochs):
        for i, batch in enumerate(training_loader):
            accumLoss = 0
            accumCorr = 0
            feat, label = batch
            optimizer.zero_grad()
            feat = feat.type(torch.FloatTensor)
            prediction = model(feat)
            loss = loss_fnc(input=prediction.squeeze(), target=label.float())
            loss.backward()
            optimizer.step()
            for j in range(0, len(prediction)):
                if label[j] == 1:
                    if prediction[j] > 0.5:
                        accumCorr = accumCorr + 1
                elif label[j] == 0:
                    if prediction[j] < 0.5:
                        accumCorr = accumCorr + 1
            accTraining.append(accumCorr/batch_size)
            lossTraining.append(loss.item())
            nArr.append(counter)
            counter = counter + 1
        if (epoch) % eval_every == 0:
            nArrValidation.append(counter)
            accuracy = evaluate(model, validation_loader)
            accValidation.append(accuracy)
            print(accuracy)
    print("Learning rate: " + str(lr))
    print("Number of Epoch: " + str(numOfEpochs))
    print("Batch size: " + str(batch_size))
    print("Accuracy Measured Every: " + str(eval_every) + " epochs")
    print("Final Validation Accuracy: " + str(evaluate(model, validation_loader)))
    print("Final Training Accuracy: " + str(evaluate(model, training_loader)))
    accTraining = savgol_filter(accTraining, 25, 2)
    lossTraining = savgol_filter(lossTraining, 25, 2)
    pyplot.legend(loc='lower right')
    pyplot.subplot(2, 1, 1)
    pyplot.plot(np.array(nArr), np.array(lossTraining), label="training set")
    pyplot.title("Loss vs Steps")
    pyplot.ylabel("Loss")
    pyplot.legend(loc='lower right')
    pyplot.subplot(2, 1, 2)
    pyplot.plot(np.array(nArr), np.array(accTraining), label="training set")
    pyplot.plot(np.array(nArrValidation), np.array(accValidation), 'bo', label="validation set")
    pyplot.title("Accuracy vs Steps")
    pyplot.ylabel("Accuracy")
    pyplot.legend(loc='lower right')
    pyplot.xlabel("Steps")
    pyplot.show()

#
if __name__ == "__main__":
    # searchLearningRate()()
    # activationFunctions()
    main()