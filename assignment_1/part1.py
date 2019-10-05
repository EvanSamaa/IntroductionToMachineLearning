import numpy as np

# 1. coding and numpy exercises
filename = "matrix.csv"
matrix = np.loadtxt(fname = filename, delimiter = ",")
vector = np.load("vector.npy")

# dot project using for loop
resultVector = np.zeros([3,1])
for i in range(0,matrix.shape[0]):
    for k in range (0, matrix.shape[1]):
        resultVector[i] += matrix[i][k] * vector[k]
np.savetxt(fname = "output_forloop.csv", X = resultVector)
# dot project using np.dot
resultVector2 = np.dot(matrix, vector)
np.save(file="output_dot.npy", arr = resultVector2)
# find output difference
difference = resultVector2 - resultVector
np.savetxt("output_difference.csv", difference)

# If the two files you compared above are the same, does it prove that your code is correct? Explain your answer.
# Ans: If the two files contains the same vector and produce a zero vector as the output_difference, then it means my code produce the same output as the numpy.dot() function. Which means my code is correct.