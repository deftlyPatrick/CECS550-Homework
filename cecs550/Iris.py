import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def init():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    iris_np = df.to_numpy()
    X, y = df, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    return X_train, X_test, y_train, y_test



X_train, X_test, y_train, y_test = init()

print("X_train: ", X_train)
print("X_test: ", X_test)
print("y_train: ", y_train)
print("y_test: ", y_test)
