import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from mpl_toolkits.mplot3d import Axes3D


#  Example independent variable (observations)
X = np.array([[0.,0.], [1.,0.], [2.,0.], [3.,0.], [4.,0.],
                [5.,0.], [6.,0.], [7.,0.], [8.,0.], [9.,0.], [10.,0.],
                [11.,0.], [12.,0.], [13.,0.], [14.,0.],
                [0.,1.], [1.,1.], [2.,1.], [3.,1.], [4.,1.],
                [5.,1.], [6.,1.], [7.,1.], [8.,1.], [9.,1.], [10.,1.],
                [11.,1.], [12.,1.], [13.,1.], [14.,1.],
                [0.,2.], [1.,2.], [2.,2.], [3.,2.], [4.,2.],
                [5.,2.], [6.,2.], [7.,2.], [8.,2.], [9.,2.], [10.,2.],
                [11.,2.], [12.,2.], [13.,2.], [14.,2.]])#.T

# Example dependent variable (observations) - noiseless case
y = np.array([4.0, 3.98, 4.01, 3.95, 3.9, 3.84,3.8,
              3.73, 2.7, 1.64, 0.62, 0.59, 0.3,
              0.1, 0.1,
            4.4, 3.9, 4.05, 3.9, 3.5, 3.4,3.3,
              3.23, 2.6, 1.6, 0.6, 0.5, 0.32,
              0.05, 0.02,
            4.0, 3.86, 3.88, 3.76, 3.6, 3.4,3.2,
              3.13, 2.5, 1.6, 0.55, 0.51, 0.23,
              0.11, 0.01])

x1 = np.linspace(0, 14, 20)
x2 = np.linspace(0, 5, 100)

i = 0
inputs_x = []
# y_test = []
while i < len(x1):
    j = 0
    while j < len(x2):
        inputs_x.append([x1[i],x2[j]])
        # y_test.append(y[j])
        j = j + 1
    i = i + 1
inputs_x_array = np.array(inputs_x)


# Instantiate a Gaussian Process model
# kernel = C(1.0, (1e-3, 1e3)) * RBF((1e-2, 1e2), (1e-2, 1e2))
kernel =RBF((1e-2, 1e2), (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)

gp.fit(X, y.reshape(-1,1))  # removing reshape results in a different error

y_pred, sigma = gp.predict(inputs_x_array, return_std=True)

xx = np.linspace(0, 14, 15)
yy = np.linspace(0, 2, 6)
XX, YY = np.meshgrid(xx, yy)
ZZ = np.zeros_like(XX)
for i in range(15):
    for j in range(6):
        ZZ[j, i] = gp.predict(np.array([[XX[0, i], YY[j, 0]]]))

fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')
ax1.scatter3D(X[:, 0], X[:, 1], y, cmap='Blues')
ax1.plot_surface(XX, YY, ZZ, cmap='rainbow')
plt.show()