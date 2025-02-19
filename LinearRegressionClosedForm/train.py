import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets

from linear_regression import LinearRegression
from utility.mse import CalculateMSE

#Create synthetic dataset
X, y = datasets.make_regression(n_samples=100, n_features=1, noise = 20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=1234)

#Train
model = LinearRegression()
model.fit(X_train, y_train)

#Predict
y_predict = model.predict(X_test)

#MSE
mse = CalculateMSE(y_test,y_predict)
print(mse)

