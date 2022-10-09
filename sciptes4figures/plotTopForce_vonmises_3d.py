import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sciptes4figures.smoothKernel import savitzky_golay


def readTopForce(path='/home/shguan/simu/ABS_DEM_2_4_biaxial'):
    fileName = os.path.join(path, 'biaxial_surf.dat')
    label = os.path.split(path)[-1].split('Disp')[-1][1:]
    file = open(fileName)
    datas = file.readlines()
    file.close()
    data = np.array([[float(j) for j in i.replace('\n', '').split(' ')] for i in datas[1:-1]])
    return data, label


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
pathList = [
    # '../../simu/mises_conventionalDisp_2_2_4_3D',
    # '../../simu/mises_conventionalDisp_6_6_12_3D',
    # '../../simu/mises_conventionalDisp_8_8_16_3D',
    # '../../simu/mises_conventionalDisp_77_77_154_3D_revised',
    # '../../simu/mises_conventionalDisp_2_2_4_3D_revised',
    '../../simu/lade_simulations/lade_conventionalDisp_6_6_12_3D_revised_15',
    '../../simu/lade_simulations/lade_conventionalDisp_6_6_12_3D_revised_30',
    '../../simu/lade_conventionalDisp_6_6_12_3D_revised_45',
            ]

plt.style.use('seaborn-paper')
fig = plt.figure(figsize=[8, 6])
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i, path in enumerate(pathList):
    datas, label = readTopForce(path=path)
    ax1.plot(-datas[:, 0], -datas[:, 1]/1e3, label=label, linewidth=3, alpha=0.8)
    ax2.plot(-datas[:, 0], datas[:, 3], label=label, linewidth=3, alpha=0.8)

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
plt.show()
# plt.savefig('/media/shguan/Elements SE/ubuntu_home/simu/csuh/conventional displacement compression_e0.png', dpi=200)
print('Figure is saved as %s' % os.path.join(os.getcwd(), 'topForceCompare_MISES_1_1.png'))
