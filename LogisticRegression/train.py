import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.datasets as datasets
from logistic_regression import  LogisticRegression

X, y = datasets.make_classification(n_samples=1000, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1234)

model = LogisticRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
# print(y_predict)
accuracy = ((np.sum(y_predict==y_test)/len(y_test))*100)
print(f'Model accuracy = {accuracy}%')
