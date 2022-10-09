import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sciptes4figures.smoothKernel import savitzky_golay

# ---------------------------------------------------------------------------------------------


flag = 'biax'  # retaining biax
# labelList = ['FEM-DEM-coarse', 'FEM-DEM-fine', 'FEM-ML-coarse', 'FEM-ML-fine', ]
labelList = ['FEM-DEM-coarse', 'A', 'B', 'C']
# labelList = ['FEM-DEM-fine', 'D', 'E', 'F']
pathList = [
    # -----------------------------------------
    ## used to figure in paper
    # '../../simu/ABS_DEM_4_8_biaxial',
    # '../../simu/ABS_DEM_8_16_biaxial',
    # # '/home/shguan/simu/ML_net_4_8_biaxial',# good but not enough
    # '../../simu/ML_net_4_8_biaxial_fixed',# good but not enough
    # # '/home/shguan/simu/ML_net_8_16_biaxial_ptModelH6_30_9_double_withouRetaining_5250',  # good but not enough
    # '../../simu/ML_net_8_16_biaxial_ptModelH6_30_9_double_withouRetaining_5250_fixed',  # good but not enough
    # -----------------------------------------
    # used in the paper ---coarse
    # '../../simu/DEM_implicitSmooth_dem_2_4_stiffnessDouble',
    # '../../simu/ML_implicitSmooth_ml_2_4_stiffnessDouble_24data',
    # '../../simu/ML_implicitSmooth_ml_2_4_stiffnessDouble_816data',
    # '../../simu/ML_implicitSmooth_ml_2_4_Mixed',
    # -----------------------------------------
    # used in the paper ---fine
    '../../simu/DEM_implicitSmooth_dem_8_16_stiffnessDouble',
    # '../../simu/ML_implicitSmooth_ml_8_16_stiffnessDouble_816data',
    # '../../simu/ML_implicitSmooth_ml_8_16_stiffnessDouble_24data',
    # '../../simu/ML_implicitSmooth_ml_8_16_Mixed_epoch1w_ok',

    # '../../simu/ML_implicitSmooth_ml_2_4_stiffnessDouble_total',
    # '../../simu/ML_implicitSmooth_ml_2_4_stiffnessDouble',
    # '../../simu/DEM_implicitSmooth_dem_4_8_stiffnessDouble',
    # '../../simu/ML_implicitSmooth_ml_4_8_stiffnessDouble',
    # '../../simu/ML_implicitSmooth_ml_8_16_stiffnessDouble_24data',
            ]

plt.style.use('seaborn-paper')
fig = plt.figure(figsize=[6, 4])
ax1 = fig.add_subplot(111)
for i, path in enumerate(pathList):
    datas, label = readTopForce(path=path, flag=flag)
    # smooth the data in the 816-24 sample
    if i==2 and 'ML_implicitSmooth_ml_8_16_stiffnessDouble_24data' in pathList[i]:
        datas[20:, 1] = savitzky_golay(y=datas[20:, 1], window_size=17, order=2)
        datas[20:, 3] = savitzky_golay(y=datas[20:, 3], window_size=17, order=2)
    ax1.plot(-datas[:, 0], -datas[:, 1]/1e3, label=labelList[i], linewidth=1)

formmer, later = 16, 21
# plt.plot([-datas[0, 0], -datas[formmer, 0]], [-datas[0, 1]/1e3, -datas[formmer, 1]/1e3], 'r--',  linewidth=2)
# plt.plot([-datas[0, 0], -datas[later, 0]], [-datas[0, 1]/1e3, -datas[later, 1]/1e3], 'r--',  linewidth=2)
# plt.scatter([-datas[formmer, 0], -datas[later, 0]], [-datas[formmer, 1]/1e3, -datas[later, 1]/1e3], c='r', s=70)


ax1.tick_params(axis='x', **tickParamsDic)
ax1.tick_params(axis='y', **tickParamsDic)
# aax.grid()

ax1.set_ylabel(r'Top force (kN)', fontdict=font_3)
# ax1.xaxis.set_ticklabels([])
# ax2.set_xlabel(r'Axial strain', fontdict=font_3)
# ax2.set_ylabel(r'$\epsilon_{v}$', fontdict=font_2)
# ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
# ax2.legend(**legendDic)
plt.tight_layout()
ax1.set_xlim([-0.01, 0.102])
ax1.set_ylim([4.5, 17])
# ax2.set_xlim([-0.01, 0.102])
# ax2.set_ylim([-0.01, 0.045])
# plt.subplots_adjust(left=0.17)
# plt.show()
filename = './topForceCompare_single_line_816_2.png'
plt.savefig(filename, dpi=200)
print('Figure is saved as %s' % os.path.join(os.getcwd(), filename))
