import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sciptes4figures.utils_plot import configurations, readTopForce_biaxial


# -------------------------------------------------------------------------------------
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()


labelList = [
    'FEM-DEM',
    'FEM-ML',
    'FEM-ML 2'
]

pathList = [
    '/home/shguan/simu/DEM_retaining_20_10',
    # '/home/shguan/simu/ML_retaining_net_20_10',
    # '/home/shguan/simu/ML_retaining_net_include_20_10',
    # '/home/shguan/simu/ML_retaining_1net_20_10_noINCLUDE',
    '/home/shguan/simu/mlRetaining/ML_retaining_net_OnlyInclude_Nodouble_20_10_1w',
    # '/home/shguan/simu/ML_retaining_net_include_20_10_2w',
    # '/home/shguan/simu/ML_retaining_net_include_20_10_5w',
            ]

plt.style.use('seaborn-paper')
fig = plt.figure(figsize=[8, 4])
ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(212)
for i, path in enumerate(pathList):
    datas, label = readTopForce_biaxial(path=path)
    ax1.plot(-datas[:, 0], -datas[:, 1]/1e2, label=labelList[i], linewidth=3)
    # ax2.plot(-datas[:, 0], datas[:, 3] - 1.0, linewidth=3)


ax1.set_ylabel(r'Wall force (kN)', fontdict=font_3)
ax1.set_xlabel(r'Transverse strain', fontdict=font_3)
ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
# ax1.xaxis.set_ticklabels([])
# ax2.set_xlabel(r'Axial strain', fontdict=font_3)
# ax2.set_ylabel(r'$\epsilon_{v}$', fontdict=font_2)
# ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
# ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
# ax1.legend(prop=font_5, loc='lower left', bbox_to_anchor=(0.1, 0.01), fancybox='sawtooth', shadow=True,
#            markerscale=0.6, ncol=2)
ax1.legend(prop=font_4, loc='best',
           markerscale=0.6, ncol=1)
# ax2.legend(prop=font_5, fancybox='sawtooth', shadow=True,
#            markerscale=0.6, ncol=2)
plt.tight_layout()
plt.show()
# plt.savefig('./retainingForceCompare_NotConverge_ppt.png', dpi=400)
