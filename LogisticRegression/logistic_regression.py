import numpy as np
from utility.sigmoid import CalculateSigmoid
class LogisticRegression:
    def __init__(self, lr= 0.001, n_iters=1000):
        self.lr= lr
        self.n_iters= n_iters
        self.weights = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]
        n, d = X.shape
        self.weights = [0]*d

        for i in range(self.n_iters):
            y_pred = CalculateSigmoid(X.dot(self.weights))
            grad = X.T.dot(y_pred-y)/n
            self.weights-= self.lr * grad

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        y_pred = CalculateSigmoid(X.dot(self.weights))
        return (y_pred>+0.5).astype(int)