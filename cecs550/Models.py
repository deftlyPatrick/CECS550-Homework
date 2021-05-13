import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.feature_selection import SelectKBest, chi2
import os
from sklearn import preprocessing
from collections import defaultdict
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import tensorflow as tf
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree


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

# print(X)

X_new = SelectKBest(chi2, k=1).fit_transform(X, y)

# print(X_new)

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(X,y)

principalDf = pd.DataFrame(data=principalComponents, columns=['1', '2'])

finalDf = pd.concat([principalDf, new[['C']]], axis=1)

# print(finalDf)

plt.xlabel('Category 1', fontsize=15)
plt.ylabel('Category 2', fontsize=15)
ax.set_title('PCA', fontsize=20)
counterMalignant = 0
counterBenign = 0
for i in finalDf.itertuples():

    if i[3] == 2:
        color = 'r'
        if counterMalignant == 0:
            plt.scatter(i[1],i[2],c=color,label='malignant')
            counterMalignant += 1
        plt.scatter(i[1], i[2], c=color)
    else:
        color = 'b'
        if counterBenign == 0:
            plt.scatter(i[1],i[2],c=color,label='benign')
            counterBenign += 1
        plt.scatter(i[1], i[2], c=color)
plt.legend(loc='upper center')
plt.show()


new_v = new.drop('C', axis=1)
X, y = new_v, new['C'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
y_train = y_train.astype(int)
y_test = y_test.astype(int)
batch_size = len(X_train)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
X_test_scaled = scaler.fit_transform(X_test.astype(np.float64))

feature_columns = [tf.feature_column.numeric_column('x', shape=X_train_scaled.shape[1:])]

print(tf.__version__)

estimator = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[300, 100],
    n_classes=10)

train_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": X_train_scaled},
    y=y_train,
    batch_size=50,
    shuffle=False,
    num_epochs=None)

estimator.train(input_fn=train_input,steps=1000)

# print(type(X_test))
# print(type(y_test))

eval_input = tf.compat.v1.estimator.inputs.numpy_input_fn(
    x={"x": X_test_scaled},
    y=y_test,
    shuffle=False,
    batch_size=X_test_scaled.shape[0],
    num_epochs=1)

eval = estimator.evaluate(eval_input,steps=None)

print(eval)

clf = tree.DecisionTreeClassifier()

# Fit regression model
regr_1 = tree.DecisionTreeRegressor(max_depth=2)
regr_2 = tree.DecisionTreeRegressor(max_depth=5)

clf_train_1 = regr_1.fit(X_train, y_train)
clf_train_2 = regr_2.fit(X_train, y_train)

clf_test_1 = regr_1.fit(X_test, y_test)
clf_test_2 = regr_2.fit(X_test, y_test)


clf_train_1_fit = regr_1.fit(X, y)
clf_train_2_fit = regr_2.fit(X, y)

clf_test_1_fit = regr_1.fit(X, y)
clf_test = regr_2.fit(X, y)

# Predict
y_1_train = regr_1.predict(X_train)
y_2_train = regr_2.predict(X_train)

y_1_test = regr_1.predict(X_test)
y_2_test = regr_2.predict(X_test)

print(y_1_train)
print(y_2_train)

print(y_1_test)
print(y_2_test)