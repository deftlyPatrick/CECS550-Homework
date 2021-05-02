df = pd.DataFrame(breast_cancer)
# print(df.iloc[0][5])
remove = []
counterBN = 0

for i in df["BN"]:
    if i == "?":
        # print(i)
        remove.append(counterBN)
        counterBN += 1
    counterBN += 1


# print(remove)

new = df.drop(index=remove, axis=0)
new = new.drop('id', axis=1)
new = new.dropna(axis='columns')
# print(new)

for i in range(len(new)):
    print(new.iloc[i])