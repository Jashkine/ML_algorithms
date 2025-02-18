import numpy as np

def euclidean_distances(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))