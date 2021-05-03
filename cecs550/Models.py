import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectKBest, chi2
import os
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
# print(len(data.target))

breast_cancer = pd.read_csv("breast-cancer-wisconsin.csv")
# print(breast_cancer)

df = pd.DataFrame(breast_cancer)
new = df.replace({'?':np.nan}).dropna()
new['BN'] = new['BN'].astype(str).astype('int64')
new = new.drop('id', axis=1)
new = new.dropna(axis='columns')

# print(new)

# print(new.columns)
# print(len(new.columns))

labels = []

for i in new.columns:
    labels.append(i)


# print(labels)

new.index = pd.RangeIndex(len(new.index))
new.index = range(len(new.index))

# for i in range(len(new)):
#     print(new.iloc[i])

print(new['BN'])
print(new['CT'])
print(new['UCSize'])

counter = 0
for i in range(len(new['CT'])):
    # print("i:", i, " ", new['BN'][i])
    if counter != i:
        print("NO")
    counter += 1
# print(list(new.columns))

# print(new.iloc[[]])
# boxplot = new.boxplot(['CT', 'UCSize', 'UCShape', 'MA', 'SECS','BN', 'BC', 'NN', 'M', 'C'], showbox=True)
plt.show()



# fig, ax = plt.subplots(figsize=(16,8))
# ax.scatter(new['CT'], new['BN'])
# plt.show()
# removalOutlier = []
# for i in range(len(df)):
#     counter = 0
#     counterTwo = 0
#     for k in range(len(df[i])):
#         if df[i][k] =