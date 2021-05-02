# print(breast_cancer)

df = pd.DataFrame(breast_cancer)

size = 569
x = pd.Series(np.random.normal(size=size))
x = x[x.between(x.quantile(.15), x.quantile(.85))]
print(x)
# print(df)

# print(df["BN"])
# print("df: ", len(df))
#
#
# remove = []
# temp = set()
# tempB = []
#
# counterBN = 0
# counterID = 0
#
#
# for i in df["id"]:
#     temp.add(i)
#
# print(len(temp))
#
# for i in df["BN"]:
#     if i == "?":
#         remove.append(counterBN)
#         counterBN += 1
#     counterBN += 1
#
#
# print(remove)
#
# new = df.drop(index=remove, axis=0)
# new = new.drop('id', axis=1)
# # print(new)
# # print(new.shape)
#
# newcor = new.corr()

# plt.figure(figsize=(8, 8))
#
# sns.heatmap(newcor, cbar=True, annot=False, yticklabels=new.columns,
#             cmap=ListedColormap(['#C71585', '#DB7093', '#FF00FF', '#FF69B4', '#FFB6C1', '#FFC0CB']),
#             xticklabels=new.columns)
# plt.show()
#
# X, y = new,