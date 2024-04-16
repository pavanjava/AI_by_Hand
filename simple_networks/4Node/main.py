import math

import numpy as np


def relu(i):
    try:
        return np.maximum(0, i)
    except Exception as e:
        print(e.__cause__)


def sigmoid(i):
    try:
        return round(1 / (1 + pow(math.e, i)), 8)
    except Exception as e:
        print(e.__cause__)


def tanh(i):
    try:
        return round((pow(math.e, i) - pow(math.e, -i)) / (pow(math.e, i) + pow(math.e, -i)), 8)
    except Exception as e:
        print(e.__cause__)


def four_node(weights, bias, inputs):
    linear_combination_output = []
    output = []
    try:
        for i in range(len(weights)):
            linear_combination_output.append(np.dot(np.array(weights[i]), inputs) + bias[i])
            output.append(sigmoid(linear_combination_output[i]))
    except Exception as e:
        print(e.__cause__)

    return linear_combination_output, output


# Given values for 4 single node network
inputs = np.array([2, 1, 3])
weights = np.array([[1, -1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
bias = [-5, 0, 1, -2]  # bias should be a vector or an array

# Perform the operation
output_before_activation, activated_output = four_node(weights, bias, inputs)
print(f"Wx + b = {output_before_activation}")
print(f"ReLU(Wx + b) = {activated_output}")
