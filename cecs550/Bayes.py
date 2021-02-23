import pandas as pd
import csv
import os


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
def initate_data(df, column_data: int):

    df_colSearch = df.iloc[:, column_data]
    df_colAssignment_values = df.iloc[:, 3]

    w1Col_data = []
    w2Col_data = []
    w3Col_data = []

    for i in range(len(df_colAssignment_values)):
        if df_colAssignment_values[i] == "w1":
            w1Col_data.append(df_colSearch[i])

        if df_colAssignment_values[i] == "w2":
            w2Col_data.append(df_colSearch[i])

        if df_colAssignment_values[i] == "w3":
            w3Col_data.append(df_colSearch[i])

    colDict = {"w1": w1Col_data, "w2": w2Col_data, "w3": w3Col_data}

    return colDict

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
# print(dfTest.iloc[:,3])
# print(type(dfTest.iloc[:, 3].values[0]))
#

print(initate_data(dfTest, 1))



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
