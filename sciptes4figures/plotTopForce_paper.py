import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sciptes4figures.smoothKernel import savitzky_golay


def readTopForce(path='/home/shguan/simu/ABS_DEM_2_4_biaxial', flag='biax'):
    if flag == 'biax':
        fileName = os.path.join(path, 'biaxial_surf.dat')
        file = open(fileName)
        head = file.readline()
        file.close()
        if 'Axial' not in head:
            dataframe = pd.read_csv(fileName,
                                    header=None,
                                    delimiter=' ',
                                    names=['AxialStrain', 'forceTop', 'lengthTop', 'volumeStrain'],
                                    skiprows=[101]
                                    )
        else:
            dataframe = pd.read_csv(fileName,
                                    delimiter=' ',
                                    skiprows=[102]
                                    )
        temp = os.path.split(path)[1].split('_')[1:4]
        label = 'FEM-'+(temp[0] if 'DEM' in temp[0] else 'ML') + '_' + temp[1] + '_' + temp[2]
        dataNp = dataframe.to_numpy()
        return dataNp, label
    elif flag == 'retaining':
        dataframe = pd.read_csv(os.path.join(path, 'pressure.dat'),
                                header=None,
                                delimiter=' ',
                                names=['Strain', 'Top force', 'Top length'],
                                # index_col=[1, 2, 3, 4],
                                skiprows=[101])
        temp = os.path.split(path)[1].split('_')[1:4]
        label = 'FEM-' + (temp[0] if 'DEM' in temp[0] else 'ML') + '_' + temp[1] + '_' + temp[2]
        dataNp = dataframe.to_numpy()
        return dataNp, label


# ---------------------------------------------------------------------------------------------
#       FIGURE CONFIGURATION
font_1 = {'weight': 'normal', 'size': 23}
font_2 = {'weight': 'normal', 'size': 20}
font_3 = {'weight': 'normal', 'size': 18}
font_4 = {'weight': 'normal', 'size': 16}
font_5 = {'weight': 'normal', 'size': 14}

tickParamsDic = {'which': 'major',
                 'direction': 'out',
                 'length': 6,
                 'width': 1.5,
                 'labelsize': 16}
legendDic = {'prop': font_4,
             'fancybox': True,
             'framealpha': 0.1,
             'facecolor': 'c',
             }
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
    # '../../simu/DEM_implicitSmooth_dem_8_16_stiffnessDouble',
    # '../../simu/ML_implicitSmooth_ml_8_16_stiffnessDouble_816data',
    # '../../simu/ML_implicitSmooth_ml_8_16_stiffnessDouble_24data',
    # '../../simu/ML_implicitSmooth_ml_8_16_Mixed_epoch1w_ok',

    # '../../simu/ML_implicitSmooth_ml_2_4_stiffnessDouble_total',
    # '../../simu/ML_implicitSmooth_ml_2_4_stiffnessDouble',
    # '../../simu/DEM_implicitSmooth_dem_4_8_stiffnessDouble',
    # '../../simu/ML_implicitSmooth_ml_4_8_stiffnessDouble',
    # '../../simu/ML_implicitSmooth_ml_8_16_stiffnessDouble_24data',

    # active learning
    '../../simu/ML_implicitSmooth_ml_8_16_ActiveLearning',
            ]

plt.style.use('seaborn-paper')
fig = plt.figure(figsize=[8, 6])
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i, path in enumerate(pathList):
    datas, label = readTopForce(path=path, flag=flag)
    # smooth the data in the 816-24 sample
    if i == 2 and 'ML_implicitSmooth_ml_8_16_stiffnessDouble_24data' in pathList[i]:
        datas[20:, 1] = savitzky_golay(y=datas[20:, 1], window_size=17, order=2)
        datas[20:, 3] = savitzky_golay(y=datas[20:, 3], window_size=17, order=2)
    if i == 0:
        index = range(1, len(datas), 5)
        ax1.scatter(-datas[:, 0][index], -datas[:, 1][index] / 1e3, c='r', marker='o', edgecolors='r', label=labelList[i], s=60)
        ax2.scatter(-datas[:, 0][index], datas[:, 3][index] - 1.0, c='r', marker='o', edgecolors='r', label=labelList[i], s=60)
    else:
        ax1.plot(-datas[:, 0], -datas[:, 1]/1e3, label=labelList[i], linewidth=3, alpha=0.8)
        ax2.plot(-datas[:, 0], datas[:, 3] - 1.0, label=labelList[i], linewidth=3, alpha=0.8)

for aax in [ax1, ax2]:
    aax.tick_params(axis='x', **tickParamsDic)
    aax.tick_params(axis='y', **tickParamsDic)
    aax.grid()

ax1.set_ylabel(r'Top force (kN)', fontdict=font_3)
ax1.xaxis.set_ticklabels([])
ax2.set_xlabel(r'Axial strain', fontdict=font_3)
ax2.set_ylabel(r'$\epsilon_{v}$', fontdict=font_2)
# ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax2.legend(**legendDic)
plt.tight_layout()
ax1.set_xlim([-0.01, 0.102])
ax2.set_xlim([-0.01, 0.102])
ax2.set_ylim([-0.01, 0.045])
# plt.subplots_adjust(left=0.17)
plt.show()
# plt.savefig('./topForceCompare_coarse.png', dpi=200)
print('Figure is saved as %s' % os.path.join(os.getcwd(), 'topForceCompare.png'))
