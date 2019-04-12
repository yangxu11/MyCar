import numpy as np

c = np.zeros([2,3])
a = [[1,1,1],
     [1,1,1]]

b = np.vstack((c,a))
print(b)