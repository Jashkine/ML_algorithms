import numpy as np

def CalculateSigmoid(x):
    return 1/(1+np.exp(-x))