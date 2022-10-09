import matplotlib.pyplot as plt
from sciptes4figures.utils_plot import readTopForce_biaxial, configurations


# ---------------------------------------------------------------------------------------------
#       FIGURE CONFIGURATION
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
# ---------------------------------------------------------------------------------------------


flag = 'biax'  # retaining biax
pathList = [
    '../../simu/biaxial/biaxial_implicit_rough_dem_x9_y18_2D_order2_numG648',
    # '../../simu/lade_simulations/lade_conventionalDisp_6_6_12_3D_revised_30',
            ]

plt.style.use('seaborn-paper')
fig = plt.figure(figsize=[8, 6])
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
for i, path in enumerate(pathList):
    datas, label = readTopForce_biaxial(path=path)
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
# plt.savefig('../../simu/lade_simulations/topForceCompare_lade.png', dpi=200)
# print('Figure is saved as %s' % '../../simu/lade_simulations/topForceCompare_lade.png')
