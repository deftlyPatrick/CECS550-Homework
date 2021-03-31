import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import random

def init():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    iris_np = df.to_numpy()
    X, y = df, iris.target
    k = random.randint(1,4)

    return X, y, k

def train_test_split_temp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    return X_train, X_test, y_train, y_test


def KNeighborsClassifier_predict(X, y, k, predict_number=None):

    X = pd.DataFrame(X).to_numpy()
    X_list = X.tolist()
    y_list = y.tolist()
    # X.reshape(1, -1)
    neigh = KNeighborsClassifier(n_neighbors=k)
    fit = neigh.fit(X_list, y_list)
    predicted = fit.predict([[predict_number]])

    return predicted

def KNeighborsClassifier_predict_prob(X, y, k, predict_proba=None):

    X = pd.DataFrame(X).to_numpy()
    X_list = X.tolist()
    y_list = y.tolist()
    # X.reshape(1, -1)
    neigh = KNeighborsClassifier(n_neighbors=k)
    fit = neigh.fit(X_list, y_list)
    predicted_prob = fit.predict([[predict_proba]])

    return predicted_prob

X, y, k = init()

X_train, X_test, y_train, y_test = train_test_split_temp(X, y)

print("X_train: ", X_train, "\n\n")
print("X_test: ", X_test, "\n\n")
print("y_train: ", y_train, "\n\n")
print("y_test: ", y_test, "\n\n")

predictOne = pd.DataFrame(X_test).to_numpy()

print(predictOne[0])
fit_Train = KNeighborsClassifier_predict(X_train, y_train, k, predict_number=predictOne[0])
# fit_Test = KNeighborsClassifier_temp(X_test, y_test, k, predict_number=1.1)

# train = KNeighborsClassifier_temp(X_train, y_train, k, predict_number=1.1)

# print("1.1 Predict - Train: ", train)
