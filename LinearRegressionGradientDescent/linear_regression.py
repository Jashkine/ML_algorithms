import numpy as np
class LinearRegression():
    def __init__(self, lr=0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]),X] #concatinate vector of ones as first feature of train dataset
        n, d = X.shape
        self.weights = np.zeros(d)

        for itrs in range(self.n_iters):
            grad = 2*(X.T.dot(X.dot(self.weights)-y))/n
            self.weights -= self.lr* grad

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        y_predict = X.dot(self.weights)
        return y_predict
