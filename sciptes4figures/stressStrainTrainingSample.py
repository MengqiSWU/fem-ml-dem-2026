import random
import matplotlib.pyplot as plt
from FEMxML.getSeriesDataConverged import getSeriasData, getSeriasDataCoupling


#%%
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

#%%
strain, strain_increment, stress, tangent, strain_abs, n, convergeList = getSeriasData(
    path='../../simu/ABS_DEM_8_16_biaxial', time=100)
strainCoupling, strain_incrementCoupling, stressCoupling, \
tangentCoupling, strain_absCoupling, nCoupling, convergeListCoupling = getSeriasDataCoupling(
    path='../../simu/ML_net_8_16_biaxial_ptModelH6_30_9_double_withouRetaining_5250', time=100)

#%%
indexOfPoint = random.choices(range(512), k=5)
fig = plt.figure(figsize=(8, 9))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
axList = [ax1, ax2, ax3]
labelList = ['$\sigma_{00}$', '$\sigma_{01}$', '$\sigma_{11}$']
# plot stain
for i in indexOfPoint:
    for j in range(3):
        axList[j].plot(range(len(stress[i, :, j])), stress[i, :, j], label=('%s_%d' % (labelList[j], i)))
# configuration
for aax in axList:
    aax.tick_params(axis='x', **tickParamsDic)
    aax.tick_params(axis='y', **tickParamsDic)
    aax.legend(**legendDic)

ax1.xaxis.set_ticklabels([])
ax2.xaxis.set_ticklabels([])
ax3.set_xlabel(r'Axial strain', fontdict=font_3)
plt.tight_layout()
plt.show()