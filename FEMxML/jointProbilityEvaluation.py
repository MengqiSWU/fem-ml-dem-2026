import random

import numpy as np
import pandas as pd
import smogn
import pandas
from matplotlib import pyplot as plt
import seaborn as sns

try:
    from train_model_strain_double_tanget import get_data, pickle_load, plot_prection
except:
    from FEMxML.train_model_strain_double_tanget import get_data, pickle_load, plot_prection

x_data, y_data, _ = get_data(root_path_list=[
    '/home/shguan/simu/ABS_DEM_2_4_biaxial',
    # '../../simu/ABS_DEM1_2_4_biaxial',
    # '../../simu/ABS_DEM2_2_4_biaxial',
    # '../../simu/ABS_DEM_2_4_gaussianConfinedPressure1',
    # '../../simu/ABS_DEM_2_4_gaussianConfinedPressure2',
    # '../../simu/ABS_DEM_2_4_gaussianConfinedPressure3',
    # '../../simu/ABS_DEM_8_16_biaxial'
], maxTime=102, scalarPath=None)
dataFrame = pd.DataFrame(data=np.concatenate((x_data, y_data), axis=1),
                         columns=['x0', 'x1', 'x2', 'x_abs', 'y0', 'y1', 'y2'])

x_std, y_std = pickle_load('input_std',
                           'output_std',
                           root_path='/home/shguan/fem-ml-dem/FEMxML/ptModelH4_30_9_Stiffness_double')
x_data_normed = x_data / x_std
y_data_normed = y_data / y_std
sns.set()
sns.jointplot(x_data[:, 0], x_data[:, 2])
plt.show()

# %%
n = 1000
x = np.random.randn(n)
y = np.random.randn(n) ** 2
plt.hist2d(x, y, 30, vmax=10)
plt.show()

sns.set()
sns.jointplot(x, y, kind='scatter', color=[.9, .2, .5]).plot_joint(sns.kdeplot)
plt.show()

# %%
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 100)
plt.plot(x, norm.pdf(x))
plt.show()


# %%
# data generation
def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x


x = make_data(1000)
sns.kdeplot(x, );
plt.tight_layout();
plt.show()
hist = plt.hist(x, bins=30, normed=True, edgecolor='k');
plt.tight_layout();
plt.show()
summ = np.sum(hist[0] * (hist[1][2] - hist[1][1]))

# %%
x = make_data(20)
bins = np.linspace(-5, 10, 10)
fig, ax = plt.subplots(1, 2, figsize=(12, 4),
                       sharex=True, sharey=True,
                       subplot_kw={'xlim': (-4, 9),
                                   'ylim': (-0.02, 0.3)})
fig.subplots_adjust(wspace=0.05)
for i, offset in enumerate([0.0, 0.6]):
    ax[i].hist(x, bins=bins + offset, normed=True, edgecolor='k')
    ax[i].plot(x, np.full_like(x, -0.01), '|k',
               markeredgewidth=1)
plt.tight_layout();
plt.show()

fig, ax = plt.subplots()
bins = np.arange(-3, 8)
ax.plot(x, np.full_like(x, -0.1), '|k',
        markeredgewidth=1)
for count, edge in zip(*np.histogram(x, bins)):
    for i in range(count):
        ax.add_patch(plt.Rectangle((edge, i), 1, 1,
                                   alpha=0.5))
ax.set_xlim(-4, 8)
ax.set_ylim(-0.2, 8)
plt.tight_layout();
plt.show()

# %%
x = make_data(20)
x_d = np.linspace(-4, 8, 2000)
density = sum((abs(xi - x_d) < 0.5) for xi in x)
plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 8]);
plt.tight_layout();
plt.show()

# %% kde
from scipy.stats import norm

x_d = np.linspace(-4, 8, 2000)
density_kde = sum(norm(xi).pdf(x_d) for xi in x)

plt.plot(x_d, density_kde, alpha=0.5)
plt.fill_between(x_d, density, alpha=0.5)
plt.plot(x, np.full_like(x, -0.1), '|k', markeredgewidth=1)

plt.axis([-4, 8, -0.2, 8]);
plt.tight_layout();
plt.show()

# %% kde-sns
distribution = sns.kdeplot(x)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.tight_layout();
plt.show()

# %% kde-sklearn
from sklearn.neighbors import KernelDensity

# instantiate and fit the KDE model
kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
kde.fit(x.reshape(-1, 1))

# score_samples returns the log of the probability density
logprob = kde.score_samples(x_d.reshape(-1, 1))

plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
plt.ylim(-0.02, 0.22)
plt.tight_layout();
plt.show()

# %% bandwidth analysis (grid search) in kde-sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut

bandwidths = 10 ** np.linspace(-1, 1, 100)
grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                    {'bandwidth': bandwidths},
                    cv=LeaveOneOut())
grid.fit(x[:, None]);
best_params = grid.best_params_

# %% kde for the multiscale dataset
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

kde = KernelDensity(bandwidth=1.0, kernel='gaussian')

data = y_data_normed[:, 2]
mmin, mmax = np.min(data, axis=0), np.max(data, axis=0)

# bandwidth search
# grid = GridSearchCV(KernelDensity(),
#                     {'bandwidth': np.exp(np.linspace(-2, 1.0, 20))},
#                     cv=10)  # 10-fold cross-validation
#
# grid.fit(data[:, None])
# print(grid.best_params_)

# kde.bandwidth = 0.25450849798848973
kde.bandwidth = 0.02

n_interval = 500
x_grid = np.linspace(mmin, mmax, n_interval + 1)
dx = (mmax - mmin) / n_interval
dist = sum(np.abs(xi - x_grid) < dx / 2 for xi in data)
kde.fit(data.reshape(-1, 1))
probility_dist = np.exp(kde.score_samples(x_grid.reshape(-1, 1)))

# plt.plot(x_grid, probility_dist / max(probility_dist) * max(dist), label='KDE')
# plt.plot(x_grid, dist, label='Count')
# plt.tick_params(labelsize=15)
# plt.tight_layout();
# plt.legend(fontsize=15)
# plt.show();


# %% re-sampling (under sampling)
def under_sample(index_list, n_choices):
    return random.choices(index_list,
                          k=n_choices,
                          # weights=len(index_list)*[1],
                          )

n_average_interval = int(np.average(dist))
index_resampled = []
datas = np.concatenate((x_data, y_data), axis=1)
sort_index = list(np.argsort(datas[:, -1]))
summ = 0
for i, interval in enumerate(x_grid):
    index_pool = sort_index[summ:(summ + int(dist[i]))]
    summ += dist[i]
    if dist[i] > n_average_interval:
        new_index = under_sample(index_pool, n_average_interval)
        index_resampled += new_index
    elif n_average_interval > dist[i] > n_average_interval/10:
        times = n_average_interval // dist[i]
        add_index = under_sample(index_pool, n_average_interval % dist[i])
        index_resampled = index_resampled+index_pool*times+add_index
    else:
        index_resampled += 10*index_pool

y_new = y_data_normed[index_resampled]
data = y_new[:, 2]
mmin, mmax = np.min(data, axis=0), np.max(data, axis=0)
n_interval = 500
x_grid = np.linspace(mmin, mmax, n_interval + 1)
dx = (mmax - mmin) / n_interval
dist = sum(np.abs(xi - x_grid) < dx / 2 for xi in y_new[:, 2])
plt.plot(x_grid, dist)
plt.show()


