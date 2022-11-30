import numpy as np
def relu(x):
    return np.maximum(x, 0)

x = np.array([1,2,3,4,5,-2,-3])

print(relu(x))