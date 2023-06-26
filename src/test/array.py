import numpy as np

list = [[0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0],
        [255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0],
        [255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0, 0],
        [0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255, 0],
        [0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 255, 255, 255]]

np_arr = np.array(list)

print(np_arr.shape)

arr1 = np.concatenate(np_arr)
print(arr1)

k = 5
nwin = 14

s = 0
max = [0]

sum = np.sum(arr1[s:s+k])
for w in range(nwin):
    s += 3
    sum = np.sum(arr1[s:s+k])
    if sum > max:
        max = sum
    print(sum)
print()
print(max)