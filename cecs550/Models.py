import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectKBest, chi2
import os
from sklearn import preprocessing
from collections import defaultdict
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

counter = 0
for i in range(len(new['CT'])):
    # print("i:", i, " ", new['BN'][i])
    if counter != i:
        print("NO")
    counter += 1
print(len(new))

# print(new.iloc[[]])
boxplot = new.boxplot(['CT', 'UCSize', 'UCShape', 'MA', 'SECS','BN', 'BC', 'NN', 'M', 'C'], showbox=True)
plt.show()



# removalOutlier = defaultdict(list)
#
# for i in range(len(new)):
#
#     if new['MA'][i] > 8:
#         removalOutlier[i].append(i)
#
#     if new['SECS'][i] > 7:
#         removalOutlier[i].append(i)
#
#     if new['BC'][i] > 9:
#         removalOutlier[i].append(i)
#
#     if new['NN'][i] > 8:
#         removalOutlier[i].append(i)
#
#     if new['M'][i] > 1:
#         removalOutlier[i].append(i)
#

removalOutlier = {}

for i in range(len(new)):
    if new['M'][i] > 9:
        removalOutlier[i] = i

    # if newRemoved['SECS'][i] > 3:
    #     removalOutlier2[i] = i


print(len(removalOutlier))

newRemoved = new.drop(index=removalOutlier)
newRemoved = newRemoved.reset_index(drop=True)
print(newRemoved)

boxplot = newRemoved.boxplot(['CT', 'UCSize', 'UCShape', 'MA', 'SECS','BN', 'BC', 'NN', 'M', 'C'], showbox=True)
plt.show()