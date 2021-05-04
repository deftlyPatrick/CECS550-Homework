
# for i in range(len(df)):
#     if df['CT'][i] > 7:
#         removalOutlier[i] = i
#
#     if df['UZSize'][i] > 6:
#         removalOutlier[i] = i
#
#     if df['UCShape'][i] > 6:
#         removalOutlier[i] = i
#
#     if df['MA'][i] > 5:
#         removalOutlier[i] = i
#
#     if df['SECS'][i] > 5:
#         removalOutlier[i] = i
#
#     if df['BN'][i] > 7:
#         removalOutlier[i] = i
#
#     if df['BC'][i] > 6:
#         removalOutlier[i] = i
#
#     if df['NM'][i] > 5:
#         removalOutlier[i] = i
#
#     if df['M'][i] > 1:
#         removalOutlier[i] = i
#
#     if df['C'][i] > 5:
#         removalOutlier[i] = i

removalOutlier = dict(removalOutlier)

print(removalOutlier)

indRemove = []

for k, v in removalOutlier.items():
    if (len(v)) > 1:
        indRemove.append(v[0])

print(len(indRemove))

newRemoved = new.drop(index=indRemove)
newRemoved = newRemoved.reset_index(drop=True)
print(newRemoved)

boxplot = newRemoved.boxplot(['CT', 'UCSize', 'UCShape', 'MA', 'SECS','BN', 'BC', 'NN', 'M', 'C'], showbox=True)
plt.show()

removalOutlier2 ={}

#Cannot clean SECS, NN

for i in range(len(newRemoved)):
    if newRemoved['M'][i] > 8:
        removalOutlier2[i] = i

    # if newRemoved['SECS'][i] > 3:
    #     removalOutlier2[i] = i


print(len(removalOutlier2))

newRemoved = newRemoved.drop(index=removalOutlier2)
newRemoved = newRemoved.reset_index(drop=True)
print(newRemoved)

boxplot = newRemoved.boxplot(['CT', 'UCSize', 'UCShape', 'MA', 'SECS','BN', 'BC', 'NN', 'M', 'C'], showbox=True)
plt.show()
