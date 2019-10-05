import numpy as np


linearTable = np.fromfile("AccuracyMatrix_linear.dat")
reluTable = np.fromfile("AccuracyMatrix_Relu.dat")
sigmoidTable = np.fromfile("AccuracyMatrix_Sigmoid.dat")

print(linearTable)
np.savetxt("linearTable.csv", linearTable, delimiter=",")