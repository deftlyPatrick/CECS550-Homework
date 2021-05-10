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
from sklearn.neighbors import KernelDensity


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

# print(new)

boxplot = new.boxplot(['CT', 'UCSize', 'UCShape', 'MA','BN', 'BC', 'NN', 'C'], showbox=True)
plt.show()


benign = []
malignant = []
for i in new.itertuples():
    temp = []
    counter = 0
    num = 0
    for k in i:
        if counter == 0:
            counter += 1
        elif counter == 10:
            if k == 2:
                num = 2
                temp.append(k)
            else:
                num = 4
                temp.append(k)
        else:
            temp.append(k)
            counter += 1
    if num == 2:
        benign.append(temp)
    else:
        malignant.append(temp)
# print(X)

# print(len(benign))
# print(len(malignant))

X_benign = np.asarray(benign, dtype=np.float32)
# print(X_benign)

kde = KernelDensity(kernel='gaussian',bandwidth=0.2).fit(X_benign)
scores_benign = kde.score_samples(X_benign)

X_malignant = np.asarray(malignant, dtype=np.float32)
# print(X_malignant)

kde = KernelDensity(kernel='gaussian',bandwidth=0.2).fit(X_malignant)
scores_malignant = kde.score_samples(X_malignant)


new_series = pd.Series(scores_malignant)
ax = new_series.plot.kde()
new_series = pd.Series(scores_benign)
ax = new_series.plot.kde()
plt.show()

cols = list(new.columns)
cols = [cols[-1]] + cols[:-1]
new = new[cols]

# print(new.columns)
# print(new.describe())
#
# print(new['C'])
datas = pd.DataFrame(preprocessing.minmax_scale(new.iloc[:,1:len(new.columns)]))
datas.columns = list(new.iloc[:,1:len(new.columns)].columns)
datas['C'] = new['C']
# print(datas.shape)

# print(datas.columns)

features_mean = ['CT', 'UCSize', 'UCShape', 'MA', 'SECS', 'BN', 'BC', 'NN', 'M', 'C']
plt.figure(figsize=(15,15))
heat = sns.heatmap(datas[features_mean].corr(), vmax=1, square=True, annot=True)
plt.show()

X, y = datas, datas['C'].values

print(X)

X_new = SelectKBest(chi2).fit_transform(X, y)

print(X_new)