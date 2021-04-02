import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics, svm
import random

def init():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns = iris.feature_names)
    iris_np = df.to_numpy()
    X, y = df, iris.target
    k = range(1,25)

    return X, y, k

def train_test_split_temp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    return X_train, X_test, y_train, y_test
6

def KNeighborsClassifier_predict(X_train, y_train, X_test, y_test, k):

    # X = pd.DataFrame(X).to_numpy()
    # X_list = X.tolist()
    # y_list = y.tolist()
    #
    # # X.reshape(1, -1)

    scores = {}
    score_list = []

    for k_range in k:
        knn= KNeighborsClassifier(n_neighbors=k_range)
        knn.fit(X, y)
        y_pred = knn.predict(X_test)
        scores[k] = metrics.accuracy_score(y_test, y_pred)
        score_list.append(metrics.accuracy_score(y_test,y_pred))


    return score_list

def KNeighborsClassifier_predict_prob(X, y, k, predict_proba=None):

    X = pd.DataFrame(X).to_numpy()
    # X.reshape(1, -1)
    neigh = KNeighborsClassifier(n_neighbors=3)
    fit = neigh.fit(X, y)
    print(X)
    predicted_prob = fit.predict([[predict_proba]])

    return predicted_prob

def CrossValidation(X, y):
    clf = svm.SVC(kernel='linear', C=100)
    score = cross_val_score(clf, X, y)

    return score

def Optimizing(X, y):
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 25]}
    svc = svm.SVC()
    clf = GridSearchCV(svc,parameters)

    return clf.fit(X, y)

X, y, k = init()

X_train, X_test, y_train, y_test = train_test_split_temp(X, y)

print("X_train: ", X_train, "\n\n")
print("X_test: ", X_test, "\n\n")
print("y_train: ", y_train, "\n\n")
print("y_test: ", y_test, "\n\n")

fit_Train = KNeighborsClassifier_predict(X_train, y_train, X_test, y_test, k)
print(fit_Train)

score = CrossValidation(X_train, y_train)
print(score)

optimize = Optimizing(X_train, y_train)
print(optimize)


# train = KNeighborsClassifier_temp(X_train, y_train, k, predict_number=1.1)

# print("1.1 Predict - Train: ", train)
