import numpy as np

def calculate_euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))