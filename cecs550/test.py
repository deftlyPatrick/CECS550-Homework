import itertools

a = list(itertools.permutations(list(range(1, 4)), 2))
b = []
for i in range(len(a)):
    for k in range(len(a[i])):
        print(a[i][k])
        print(a[i][k+1], "\n")
        if a[i][k] > a[i][k+1]:
            break
        else:
            b.append(a[i])
            print(b)
            break

print(b)

print(b[0])
temp = {}
#
counter = 0
for j in range(len(b) * 2):
    temp[counter] = (b[j], b[j+1])
    counter += 1
print(temp)