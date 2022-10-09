import os
import matplotlib.pyplot as plt
from sciptes4figures.utils_plot import read_footing, configurations


# ---------------------------------------------------------------------------------------------
#       FIGURE CONFIGURATION
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
# ---------------------------------------------------------------------------------------------

pathList = [
    # '../../simu/footing/footing_implicit_csuh_intorder1_numg254_footing303_p100kPa_ocr_346.4',
    '../../simu/footing/footing_implicit_csuh_intorder1_numg254_footing303_p20kPa_ocr_346.4',
    '../../simu/footing/footing_implicit_csuh_intorder1_numg254_footing303_p100kPa_ocr_346.4',
            ]
label_list = [
    "DEM 3114",
    "DEM 3618",
    "ML ",
]


plt.style.use('seaborn-paper')
fig = plt.figure(figsize=[8, 6])
# fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i, path in enumerate(pathList):
    datas = read_footing(path=path)
    ax1.plot(-datas[:, 0], datas[:, 1]/1e3, label=label_list[i], linewidth=3, alpha=0.8)
    ax2.plot(-datas[:, 0], -datas[:, 2]/1e3, label=label_list[i], linewidth=3, alpha=0.8)

for aax in [ax1, ax2]:
    aax.tick_params(axis='x', **tickParamsDic)
    aax.tick_params(axis='y', **tickParamsDic)
    aax.grid()

ax1.set_ylabel(r'Top force X axis(kN)', fontdict=font_3)
ax1.xaxis.set_ticklabels([])
# ax1.legend(**legendDic)
ax2.set_xlabel(r'Axial strain', fontdict=font_3)
ax2.set_ylabel(r'Top force Y axis(kN)', fontdict=font_3)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax2.legend(**legendDic)
plt.tight_layout()
plt.show()
# fname = os.path.join('../../simu/footing', 'footing_Top_force.png')
# plt.savefig(fname, dpi=200)
# print('Figure is saved as %s' % fname)
