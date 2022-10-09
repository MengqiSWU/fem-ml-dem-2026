import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sciptes4figures.smoothKernel import savitzky_golay


def readTopForce(path='/home/shguan/simu/ABS_DEM_2_4_biaxial'):
    fileName = os.path.join(path, 'biaxial_surf.dat')
    # label = os.path.split(path)[-1].split('dem_')[-1]
    file = open(fileName)
    datas = file.readlines()
    file.close()
    head = datas[0]
    data = np.array([[float(j) for j in i.replace('\n', '').split(' ')] for i in datas[1:-1]])
    return data


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

pathList = [
    # active learning
    '../../simu/MisesMath_implicitSmooth_vonmises_12_24_FrobeniusNorm',
    '../../simu/MisesSemi_implicitSmooth_vonmisesml_8_16_FrobeniusNorm',
    '../../simu/MisesNet_implicitSmooth_vonmisesml_8_16_FrobeniusNorm',
            ]
label_list = ['mathematic', r'$\mathcal{NN}$ hardening', r'$\mathcal{NN}$']

plt.style.use('seaborn-paper')
fig = plt.figure(figsize=[8, 6])
# fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i, path in enumerate(pathList):
    datas = readTopForce(path=path)
    # ax1.plot(-datas[:, 0], datas[:, 1]/1e3, label=label, linewidth=3, alpha=0.8)
    # ax2.plot(-datas[:, 0], -datas[:, 2]/1e3, label=label, linewidth=3, alpha=0.8)
    ax1.plot(-datas[:, 0], -datas[:, 1]/1e3, label=label_list[i], linewidth=3, alpha=0.8) # top force
    ax2.plot(-datas[:, 0], 1-datas[:, 3], label=label_list[i], linewidth=3, alpha=0.8) # volume strain

for aax in [ax1, ax2]:
    aax.tick_params(axis='x', **tickParamsDic)
    aax.tick_params(axis='y', **tickParamsDic)
    aax.grid()

ax1.set_ylabel(r'Top force X axis(kN)', fontdict=font_3)
ax1.xaxis.set_ticklabels([])
# ax1.legend(**legendDic)
ax2.set_xlabel(r'Axial strain', fontdict=font_3)
ax2.set_ylabel(r'Volume strain', fontdict=font_3)
# ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax2.legend(**legendDic)
plt.tight_layout()
# plt.show()
fname = os.path.join('../../simu/vonmises-simulation/footing_Top_force.png')
plt.savefig(fname, dpi=200)
print('Figure is saved as %s' % fname)
