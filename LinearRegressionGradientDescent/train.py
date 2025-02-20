from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from linear_regression import LinearRegression
from utility.mse import CalculateMSE

X, y = datasets.make_regression(n_samples = 1000, n_features = 1, noise = 20, random_state=1234)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

model = LinearRegression(lr = 0.01, n_iters=1000)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)

mse = CalculateMSE(y_predict, y_test)
print(mse)
