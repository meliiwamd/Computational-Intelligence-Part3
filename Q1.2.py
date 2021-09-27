# Q1.2_graded
# Do not change the above line.

# This cell is for your imports.

import numpy as np
import matplotlib.pyplot as plt

# First of all we should create the weight matrix.
# We'll do this using Hebbian weight change.
# In this question we have two patterns to learn.
# Patterns: (1, 1, 1, -1, -1, -1) & (1, -1, 1, -1, 1, -1)
# We have 6 neurons.


def WeightMatrix(Patterns, Neurons):

    Weights = np.zeros([Neurons, Neurons])
    for k in range(len(Patterns)):

        for i in range(Neurons):
            for j in range(Neurons):

                if i is not j:
                    Weights[i][j] += Patterns[k][i] * Patterns[k][j]
        
    return Weights

# Now that we've trained the network, we should test it's result.
def Activation(Weights, Pattern, Neurons):
    Result = np.zeros(Neurons)
    for i in range(Neurons):
        Sum = 0
        for j in range(Neurons):
            Sum += Weights[i][j] * Pattern[j]
        if Sum >= 0 and Pattern[i] > 0:
            Result[i] = 1
        elif Sum < 0 and Pattern[i] < 0:
            Result[i] = 1
        else:
            Result[i] = 0

    return Result
    

def UpdatePattern(Weights, Pattern, Neurons):

    Result = Activation(Weights, Pattern, Neurons)

    if np.all(Result == 1):
        print("Pattern is stable: ", Pattern)
        return
        
    print("This is initial pattern: ", Pattern, "\nStart finding nearest pattern.")
    while not np.all(Result == 1):
        # Start updating the pattern.
        for i in range(Neurons):
            Sum = 0
            for j in range(Neurons):
                Sum += Weights[i][j] * Pattern[j]
            if Sum >= 0:
                Pattern[i] = 1
            elif Sum < 0:
                Pattern[i] = -1
        Result = Activation(Weights, Pattern, Neurons) 

    
    if np.all(Result == 1):
        print("Nearest pattern: ", Pattern)
        

# Q1.2_graded
# Do not change the above line.

# This cell is for your codes.

Neurons = 6

Patterns = [[1, 1, 1, -1, -1, -1], [1, -1, 1, -1, 1, -1]]
Weights = WeightMatrix(Patterns, Neurons)

TestPattern = [1, 1, 1, -1, -1, -1]
UpdatePattern(Weights, TestPattern, Neurons)

TestPattern = [-1, 1, 1, -1, -1, -1]
UpdatePattern(Weights, TestPattern, Neurons)


