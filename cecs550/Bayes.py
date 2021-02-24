import pandas as pd
import csv
import os
from collections import defaultdict
import math


# gi(X) = ln(p(X | wi)) + ln(P(wi))
# if gi(X) > gj(x) for all i != j then X is in the region


# P(w1) = P(w2) = 1/2
# P(w3) = 0

def load_data(file_name):
    os.chdir("..")
    csv_path = os.path.abspath(os.getcwd()) + "/" + file_name
    # print(csv_path)
    df = pd.read_csv(csv_path)
    return df


#df = data_frame
#column_data = 0, 1, 2
# def initate_data(df, column_data: int):
#
#     df_colSearch = df.iloc[:, column_data]
#     df_colAssignment_values = df.iloc[:, 3]
#
#     print(df_colSearch)
#
#     w1Col_data = []
#     w2Col_data = []
#     w3Col_data = []
#
#     x1 = []
#     x2 = []
#     x3 = []
#
#
#     for i in range(len(dfTest)):
#         if df_colAssignment_values[i] == "w1":
#             w1Col_data.append(df_colSearch[i])
#
#         if df_colAssignment_values[i] == "w2":
#             w2Col_data.append(df_colSearch[i])
#
#         if df_colAssignment_values[i] == "w3":
#             w3Col_data.append(df_colSearch[i])
#
#     colDict = {"w1": w1Col_data, "w2": w2Col_data, "w3": w3Col_data}
#
#     return colDict

def initate_data(df):
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
               "w2": [w2_col1_idx, w2_col2_idx, w2_col3_idx], "w3": [w3_col1_idx, w3_col2_idx, w3_col3_idx ]}

    return colDict

#mu
def calc_mean(colDict: dict):

    w1 = []
    w2 = []
    w3 = []

    for k, v in colDict.items():
        if k == "w1":
            for i in range(len(v)):
                w1.append(sum(v[i])/len(v[i]))
        if k == "w2":
            for i in range(len(v)):
                w2.append(sum(v[i])/len(v[i]))
        if k == "w3":
            for i in range((len(v))):
                w3.append(sum(v[i])/len(v[i]))

    omega = {"w1" : w1, "w2": w2, "w3": w3}

    return omega

#sigma
def calc_standardDev(colDict: dict, df):

    data = initate_data(df)
    mean = calc_mean(colDict)

    n = 0

    for k, v in data.items():
        n = len(v[0])
        break

    standardDev = defaultdict(list)

    for k, v in data.items():
        counter = 0
        for i in v:
            nums = 0
            for j in range(len(i)):
                nums += ((i[j] - mean[k][counter]) ** 2)
            counter += 1
            variance = nums / (n-1)
            standardDev[k].append(math.sqrt(variance))

    #converting back to normal dict
    standardDev = dict(standardDev)
    return standardDev


# likelihood ratio =
# p(x | w1) / p (x | w2) > (lb12 - lb22 / lb21 - lb11) (P(w2) / P(w1)

# p(A | B) = P(B | A) * P(A) / P(B)

def calc_likelihood(w1, w2, w3, x1, x2):
    likelihood = 0
    return likelihood


# class priors =
# (lb21 - lb11) * p(x | w1) * P(w1) > (lb12 - lb22) p(x | w2) * P(w2)
def calc_prior():
    prior = 0
    return prior


def calc_MultivarianeDistribution(X, mu, sigma):
    sigma = 0
    mu = 0
    return mu


# bayes = [ posterior =  likelihood * prior / evidence ]
def calc_bayes():
    likelihood = calc_likelihood()
    prior = calc_prior()
    bayes = likelihood
    return bayes


dfTest = load_data("HW2-TrainData.csv")
# print(dfTest.columns)
# print(dfTest.iloc[:,3])
# print(type(dfTest.iloc[:, 3].values[0]))
#

colDict = initate_data(dfTest)
# print(colDict)
# print(calc_mean(colDict))
print(calc_standardDev(colDict, dfTest))


# def initate_data(file_name):
#     df = load_data(file_name)
#     df_firstCol = df.iloc[:, 0].values
#     df_secondCol = df.iloc[:, 1].values
#     df_third_col = df.iloc[:, 2].values
#     df_colAssignment_values = df.iloc[:, 3].values
#
#     for i in range(len(df_colAssignment_values)):
#         w1_col1_idx = []
#         w2_col1_idx = []
#         w3_col1_idx = []
#
#         w1_col2_idx = []
#         w2_col2_idx = []
#         w3_col2_idx = []
#
#         w1_col3_idx = []
#         w2_col3_idx = []
#         w3_col3_idx = []
#
#         if df_colAssignment_values[i] == "w1":
#             w1_col1_idx.append(df_firstCol[i])
#             w1_col2_idx.append(df_secondCol[i])
#             w1_col3_idx.append(df_third_col[i])
#
#         if df_colAssignment_values[i] is "w2":
#             w2_col1_idx.append(df_firstCol[i])
#             w2_col2_idx.append(df_secondCol[i])
#             w2_col3_idx.append(df_third_col[i])
#
#         if df_colAssignment_values[i] is "w3":
#             w3_col1_idx.append(df_firstCol[i])
#             w3_col2_idx.append(df_secondCol[i])
#             w3_col3_idx.append(df_third_col[i])
