import numpy as np
import matplotlib.pyplot as plt

x = 1.5
iterNum = 1
err = 1e5
xl, yl = [1.5], [np.sin(1.5)+1]

while err>1e-9:
    x = x - (np.sin(x)+1)/np.cos(x)
    err = abs(np.sin(x)+1)
    iterNum += 1
    xl.append(x)
    yl.append(np.sin(x)+1)
    print("IterNum: %d  x: %e  err: %e" % (iterNum, x, err))

# plot
plt.scatter(xl, yl)

xx = np.linspace(1.8, -27, 100)
plt.plot(xx, np.sin(xx)+1)
plt.show()

