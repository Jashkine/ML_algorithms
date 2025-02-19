import numpy as np

class LinearRegression:
    def __init__(self):
        self.weights = None
    
    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]),X]   #add bias term in train data
        self.weights = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)  #closed form solution

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]),X]  #add bias term in test data
        y_predict = X.dot(self.weights)     #Predict 
        return y_predict