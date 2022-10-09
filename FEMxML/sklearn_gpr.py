from sklearn.gaussian_process import GaussianProcessRegressor as gpr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK


class GPR:
    def __init__(self, input_num=1):
        kernel = CK(constant_value=1.0, constant_value_bounds=(1e-3, 1e3)) * \
                 RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
        self.model = gpr(kernel=kernel, n_restarts_optimizer=9)

    def train(self, x, y):
        self.model.fit(x, y)


def plot2d(x_train, y_train, x, y_pre, std_pre):
    # plt.plot(x, y, label='data', linestyle="dotted")
    plt.scatter(x_train, y_train, label='Observations')
    plt.plot(x, y_pre, label='prediction')
    plt.fill_between(
        x=x.ravel(),
        y1=y_pre - 1.96 * std_pre,
        y2=y_pre + 1.96 * std_pre,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    plt.legend()
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    _ = plt.title("Gaussian process regression on noise-free dataset")
    plt.show()
    return


if __name__ == '__main__':
    numx, numy = 31, 16
    x = np.linspace(start=0, stop=10, num=numx).reshape(-1, 1)
    y = np.linspace(start=0, stop=5, num=numy).reshape(-1, 1)
    X, Y = np.meshgrid(x, y)
    Z = X*np.sin(X)+Y*np.sin(Y*Y)

    index = []
    for i in range(numx):
        for j in range(numy):
            index.append([i,j])
    index = np.array(index)
    index_train = index[np.random.choice(range(len(index)), size=int(numx*numy*0.2))]

    x_train = []
    y_train = []
    for index_temp in index_train:
        x_train.append([X[0, index_temp[0]], Y[index_temp[1], 0]])
        y_train.append(Z[index_temp[1], index_temp[0]])
    x_train, y_train = np.array(x_train), np.array(y_train)

    # rng = np.random.RandomState(1)
    # training_indices = rng.choice(np.arange(y.size), size=3, replace=False)
    # x_train, y_train = x[training_indices], y[training_indices]
    gpr_ = GPR()
    gpr_.train(x=x_train, y=y_train)

    # x_pre = []
    # for i in range(numx):
    #     for j in range(numy):
    #         x_pre.append([X[0, i], Y[j, 0]])
    # x_pre = np.array(x_pre)
    # y_pre, std_pre = gpr_.model.predict(x_pre, return_std=True)
    Z_pre = np.zeros_like(X)
    for i in range(numx):
        for j in range(numy):
            Z_pre[j, i] = gpr_.model.predict(np.array([[X[0, i], Y[j, 0]]]))
    # plot2d(x_train=x_train, y_train=y_train, x=x, y_pre=y_pre, std_pre=std_pre)

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection = '3d')
    ax1.scatter3D(x_train[:, 0], x_train[:, 1], y_train, cmap='Blues')
    ax1.plot_surface(X, Y, Z_pre, cmap='rainbow')
    plt.show()
    plt.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='rainbow')
    plt.show()


