# Classify iris data using KNN

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

model = KNN(k=3)

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

accuracy = np.sum(y_predict==y_test)/len(y_test)
print(f"KNN model accuracy is {accuracy*100}%" )
