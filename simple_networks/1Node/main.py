import numpy as np


def relu(i):
    return np.max(0, i)


def single_node(weights, bias, inputs):
    linear_combination_output = np.dot(weights, inputs) + bias
    output = relu(linear_combination_output)
    return linear_combination_output, output


# Given values for single node network
inputs = np.array([2, 1, 3])
weights = np.array([1, -1, 1])
bias = -5  # bias should be a scalar, not an array

# Perform the operation
output_before_activation, activated_output = single_node(weights, bias, inputs)
print(f"Wx + b = {output_before_activation}")
print(f"ReLU(Wx + b) = {activated_output}")
