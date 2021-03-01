import pandas as pd
import os
import numpy as np


def load_data(file_name):
    pwd = os.path.abspath(os.getcwd())
    os.chdir("..")

    # gets the csv file path
    csv_path = os.path.abspath(os.getcwd()) + "/" + file_name

    # print(csv_path)

    # stores the dataframe of the csv file
    df = pd.read_csv(csv_path)

    # returns back to original directory
    os.chdir(pwd)
    return df


def initiate_data(df):
    df_firstCol = df.iloc[:, 0].values
    df_secondCol = df.iloc[:, 1].values
    df_third_col = df.iloc[:, 2].values
    df_colAssignment_values = df.iloc[:, 3].values

    w1_col1_idx = []
    w2_col1_idx = []
    w3_col1_idx = []

    w1_col2_idx = []
    w2_col2_idx = []
    w3_col2_idx = []

    w1_col3_idx = []
    w2_col3_idx = []
    w3_col3_idx = []

    for i in range(len(df_colAssignment_values)):
        if df_colAssignment_values[i] == "w1":
            w1_col1_idx.append(df_firstCol[i])
            w1_col2_idx.append(df_secondCol[i])
            w1_col3_idx.append(df_third_col[i])

        if df_colAssignment_values[i] == "w2":
            w2_col1_idx.append(df_firstCol[i])
            w2_col2_idx.append(df_secondCol[i])
            w2_col3_idx.append(df_third_col[i])

        if df_colAssignment_values[i] == "w3":
            w3_col1_idx.append(df_firstCol[i])
            w3_col2_idx.append(df_secondCol[i])
            w3_col3_idx.append(df_third_col[i])

    colDict = {"w1": [w1_col1_idx, w1_col2_idx, w1_col3_idx],
               "w2": [w2_col1_idx, w2_col2_idx, w2_col3_idx], "w3": [w3_col1_idx, w3_col2_idx, w3_col3_idx]}

    for k, v in colDict.items():
        colDict[k] = np.array(v)

    return colDict


# mu
def calc_mean(colDict: dict):
    dim = (len(colDict), 1)

    w1 = np.zeros(dim)
    w2 = np.zeros(dim)
    w3 = np.zeros(dim)

    for k, v in colDict.items():
        if k == "w1":
            for i in range(len(v)):
                temp = np.array(sum(v[i]) / len(v[i]))
                for j in range(len(w1)):
                    if w1[j] == 0:
                        w1[j] = temp
                        break
        if k == "w2":
            for i in range(len(v)):
                temp = np.array(sum(v[i]) / len(v[i]))
                for j in range(len(w2)):
                    if w2[j] == 0:
                        w2[j] = temp
                        break
        if k == "w3":
            for i in range((len(v))):
                temp = np.array(sum(v[i]) / len(v[i]))
                for j in range(len(w3)):
                    if w3[j] == 0:
                        w3[j] = temp
                        break

    mu = {"w1": w1, "w2": w2, "w3": w3}

    return mu


def calc_variance(colDict: dict, df):
    data = initiate_data(df)
    mean = calc_mean(colDict)

    dim = (len(colDict), 1)

    n = 0

    for k, v in data.items():
        n = len(v[0])
        break

    varianceDict = {}

    for k, v in data.items():
        counter = 0
        countingVariance = 0
        varianceMatrix = np.zeros(dim)
        for i in v:
            nums = 0
            for j in range(len(i)):
                nums += ((i[j] - mean[k][counter][0]) ** 2)
            counter += 1
            variance = nums / (n - 1)
            for m in range(len(varianceMatrix)):
                if varianceMatrix[m] == 0 and countingVariance < 3:
                    varianceMatrix[m] = variance
                    countingVariance += 1
                    if countingVariance == 3:
                        varianceDict[k] = varianceMatrix
                        break
                    break

    return varianceDict


# sigma
def calc_covariance(colDict: dict, df):
    data = initiate_data(df)
    mean = calc_mean(colDict)

    dim = (len(colDict), 1)

    n = 0

    for k, v in data.items():
        n = len(v[0])
        break

    covarianceDict = {}

    for k, v in data.items():
        counter = 0
        countingCovariance = 0
        covarianceMatrix = np.zeros(dim)
        for i in v:
            nums = 0
            for j in range(len(i)):
                nums += ((i[j] - mean[k][counter][0]) ** 2)
            counter += 1
            variance = nums / (n - 1)
            covariance = np.sqrt(variance)
            for m in range(len(covarianceMatrix)):
                if covarianceMatrix[m] == 0 and countingCovariance < 3:
                    covarianceMatrix[m] = covariance
                    countingCovariance += 1
                    if countingCovariance == 3:
                        covarianceDict[k] = covarianceMatrix
                        break
                    break

    return covarianceDict


def createCovarianceMatrix(varianceDict: dict, d):
    covarianceMatrixDict = {}

    for k, v in varianceDict.items():
        tempList = []
        dim = (d, d)
        covarianceMatrix = np.zeros(dim)
        counter = 0
        for a in v:
            if len(tempList) != d:
                for i in range(len(covarianceMatrix)):
                    if covarianceMatrix[counter][counter] == 0:
                        covarianceMatrix[counter][counter] = a
                        counter += 1
                        # print(covarianceMatrix)
                        break
                tempList.append(a)
            covarianceMatrixDict[k] = covarianceMatrix

    return covarianceMatrixDict

def createCovarianceMatrixDeriv(varianceDict: dict, d):
    covarianceMatrixDict = {}

    for k, v in varianceDict.items():
        tempList = []
        dim = (d, d)
        covarianceMatrix = np.zeros(dim)
        counter = 0
        for a in v:
            if len(tempList) != d:
                for i in range(len(covarianceMatrix)):
                    if covarianceMatrix[counter][counter] == 0:
                        covarianceMatrix[counter][counter] = 1/a
                        counter += 1
                        # print(covarianceMatrix)
                        break
                tempList.append(a)
            covarianceMatrixDict[k] = covarianceMatrix

    return covarianceMatrixDict

# X = test data set - column
# mu = mean
# sigma = covariance
# d = columns
def calc_MultivariateDistribution(X, mu, covarianceMatrix, covarianceMatrixDeriv, d):
    x_m = {}

    for k, v in X.items():
        x_m[k] = np.subtract(X[k], mu[k])

    multiDict = {}

    for k, v in x_m.items():
        multivarDist = (1. / np.sqrt((2 * np.pi) ** d * np.linalg.det(covarianceMatrix[k]))) * np.exp(
            -0.5*np.linalg.solve(covarianceMatrix[k], x_m[k]).T.dot(x_m[k]))
        multiDict[k] = multivarDist

    return multiDict


# likelihood ratio =
# p(x | w1) / p (x | w2) > (lb12 - lb22 / lb21 - lb11) (P(w2) / P(w1)

# p(A | B) = P(B | A) * P(A) / P(B)

def calc_likelihood(multiDict, omega):
    likelihood = {}

    for k, v in multiDict.items():
        likelihood[k] = np.multiply(multiDict[k], omega)

    return likelihood


# class priors =
# (lb21 - lb11) * p(x | w1) * P(w1) > (lb12 - lb22) p(x | w2) * P(w2)
# def calc_prior():
#     prior = 0
#     return prior


# bayes = [ posterior =  likelihood * prior / evidence ]

# P(w | X) = P(X | w) P(w) / P(X)
def calc_bayes(multiDict: dict, X: dict, omega):
    bayes = {}

    for k, v in multiDict.items():
        bayes[k] = (np.multiply(multiDict[k], omega))

    return bayes


##################################################################
dfTrain = load_data("HW2-TrainData.csv")
dfTest = load_data("HW2-TestData.csv")
# print(dfTest.columns)
# print(dfTest.iloc[:,3])
# print(type(dfTest.iloc[:, 3].values[0]))

colDict_train = initiate_data(dfTrain)
colDict_test = initiate_data(dfTest)

print("Training: ", colDict_train, "\n")
print("Test: ", colDict_test, "\n")

d = len(colDict_train)

mean = calc_mean(colDict_train)
covariance = calc_covariance(colDict_train, dfTrain)
variance = calc_variance(colDict_train, dfTrain)
print("Mean: ", mean, "\n")
print("Variance: ", variance, "\n")
print("Covariance: ", covariance, "\n")

covarianceMatrix = createCovarianceMatrix(covariance, d)
covarianceMatrixDeriv = createCovarianceMatrixDeriv(covariance, d)

# print(covarianceMatrix)
# print(covarianceMatrixDeriv)

omega = np.array([0.5, 0.5, 0])

multivariate = calc_MultivariateDistribution(colDict_test, mean, covarianceMatrix, covarianceMatrixDeriv, d)
print("Multivariate: ", multivariate, "\n")
#
# omega = np.array([0.5, 0.5, 0])
# likelihood = calc_likelihood(multivariate, omega)
# print("Likelihood: ", likelihood, "\n")
#
# bayes = calc_bayes(likelihood, colDict_test, omega)
# print("Bayes: ", bayes, "\n")
