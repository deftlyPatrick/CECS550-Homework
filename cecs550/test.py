#

# print(len(scores))
#
# plt.hist(scores, bins=len(scores))
# plt.ylim(-0.5, 60)
# plt.show()
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

# removalOutlier = {}
#
# for i in range(len(new)):
#     if new['M'][i] > 9:
#         removalOutlier[i] = i
#
#     # if newRemoved['SECS'][i] > 3:
#     #     removalOutlier2[i] = i
#
# print(len(removalOutlier))
#
# newRemoved = new.drop(index=removalOutlier)
# newRemoved = newRemoved.reset_index(drop=True)
# print(newRemoved)
#
# boxplot = newRemoved.boxplot(['CT', 'UCSize', 'UCShape', 'MA','BN', 'BC', 'NN', 'C'], showbox=True)
# plt.show()