import os
import numpy as np

with open("pos.npy",'rb') as f:
    a = np.load(f)

with open("neg.npy",'rb') as f:
    b = np.load(f)

f = open("past.txt",'w')

for i in range(0,a.shape[0]):
    f.write(a[i,0])
for i in range(0,b.shape[0]):
    f.write(b[i,0])

f.close()
