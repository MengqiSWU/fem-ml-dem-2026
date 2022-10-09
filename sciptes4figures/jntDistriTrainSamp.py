import numpy as np
from FEMxML.train_model_strain import get_data
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import os

"""
plot the distribution of the x and y components of the training data
"""

font_1 = {'family': 'Arial', 'weight': 'normal', 'size': 23}
font_2 = {'family': 'Arial', 'weight': 'normal', 'size': 20}
font_3 = {'family': 'Arial', 'weight': 'normal', 'size': 18}
font_4 = {'family': 'Arial', 'weight': 'normal', 'size': 16}
font_5 = {'family': 'Arial', 'weight': 'normal', 'size': 14}

root_path_list = [
    '../../simu/ABS_DEM_2_4_result',
    '../../simu/ABS_DEM_2_4_gaussianConfinedPressure0',
    '../../simu/ABS_DEM_2_4_gaussianConfinedPressure1',
    '../../simu/ABS_DEM_2_4_gaussianConfinedPressure2',
    '../../simu/ABS_DEM_2_4_gaussianConfinedPressure3',
    '../../simu/ABS_DEM_2_4_gaussianConfinedPressure4',
    # '../../simu/ABS_DEM_8_16_biaxial',
]
x_data, y_data, _ = get_data(root_path_list=root_path_list, maxTime=101)

print(len(x_data))

temp = len(x_data)
length = temp
dataFrame = pd.DataFrame(columns=['x', 'y'], data=np.concatenate((x_data[:length, 0:1], x_data[:length, 2:3]), axis=1))
dataFrame1 = pd.DataFrame(columns=['x', 'y'], data=np.concatenate((x_data[:length, 3:4], x_data[:length, 5:6]), axis=1))

# plot
# fig = plt.figure()
# ax = fig.add_subplot(111)
kdeplot = sns.jointplot(x="x", y="y", data=dataFrame,
                        kind="hex",
                        cmap='Blues',
                        # legend=True,
                        # cbar=True,
                        # fill=True,
                        # space=0,
                        # thresh=0
                        )
# g.plot_joint(sns.scatterplot, s=100, alpha=.5)
kdeplot.plot_marginals(sns.histplot, kde=True)
# plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
# get the current positions of the joint ax and the ax for the marginal x
# pos_joint_ax = kdeplot.ax_joint.get_position()
# pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
# reposition the joint ax so it has the same width as the marginal x ax
# kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
# reposition the colorbar using new x positions and y positions of the joint ax
# kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
name = os.path.join(os.getcwd(), 'jointDistribution.png')
kdeplot.set_axis_labels('$\epsilon_{11}$', '$\epsilon_{22}$', fontdict=font_5)
# g.set_yticklabels(g.get_yticks(), fontdict=font_1)
plt.xticks(size=12)
plt.yticks(size=12)
plt.tight_layout()
# plt.show()
plt.savefig(name, dpi=400)
print('Figure saved as \t%s' % name)

# plot the distribution of x & y components of strain
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
sns.displot(data=dataFrame['x'])
plt.show()
# ax2 = fig.add_subplot(122)
sns.displot(data=dataFrame['y'])
plt.show()

# # create some dummy data: gaussian multivariate with 10 centers with each 1000 points
# tumg = np.random.normal(np.tile(np.random.uniform(10, 20, 10), 1000), 2)
# pumg = np.random.normal(np.tile(np.random.uniform(10, 20, 10), 1000), 2)
#
# kdeplot = sns.jointplot(x=tumg, y=pumg, kind="kde", cbar=True, fill=True)
# plt.subplots_adjust(left=0.1, right=0.8, top=0.9, bottom=0.1)
# # get the current positions of the joint ax and the ax for the marginal x
# pos_joint_ax = kdeplot.ax_joint.get_position()
# pos_marg_x_ax = kdeplot.ax_marg_x.get_position()
# # reposition the joint ax so it has the same width as the marginal x ax
# kdeplot.ax_joint.set_position([pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height])
# # reposition the colorbar using new x positions and y positions of the joint ax
# kdeplot.fig.axes[-1].set_position([.83, pos_joint_ax.y0, .07, pos_joint_ax.height])
# plt.show()
