import numpy as np

def CalculateMSE(y1,y2):
    return np.mean((y1-y2)**2)