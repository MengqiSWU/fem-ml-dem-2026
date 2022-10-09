import numpy as np

a = np.random.random(size=[2, 2])

b1 = np.sum(a*a)
a[0, 0] = -a[0, 0]
b2 = np.sum(a**2.)
a[1, 0] = -a[1, 0]
b3 = np.trace(a@a.T)
