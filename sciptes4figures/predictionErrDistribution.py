import pandas as pd

from FEMxML.getSeriesDataConverged import getSeriasData, getSeriasDataCoupling
from FEMxML.train_model_strain_lastDouble import get_data, pickle_load, plot_prection
import matplotlib.pyplot as plt
import numpy as np
from FEMxML.netTorchLastDouble import Net
from FEMxML.netTorch import modelRestore
import os
import seaborn as sns

print(os.getcwd())

x_data, y_data, _ = get_data(root_path_list=[
    '../../simu/ABS_DEM_2_4_result',
    # '/home/shguan/simu/ABS_DEM1_2_4_biaxial',
    # '/home/shguan/simu/ABS_DEM2_2_4_biaxial',
    # '/home/shguan/simu/ABS_DEM_8_16_biaxial'
    # '/home/shguan/simu/Right_DEM_2_4_biaxial',
    # '/home/shguan/simu/Right_DEM_8_16_biaxial',
], maxTime=101)
print('Read data finished!')
stress, tangent = y_data[:, :3], y_data[:, 3:]

# net restore
ml_model_path = '../FEMxML'
net = modelRestore(savedPath=os.path.join(ml_model_path, 'ptModelH4_30_9_Lastdouble_without'), trainFlag=False)

stress_predict, tangent_predict = net.get_stressAndStiffness(x_data)
stress_err = stress_predict / stress - 1.0
tangent_err = tangent_predict / tangent - 1.0

stress_err11 = sorted(stress_err[:, 0], key=lambda x: abs(x))
stress_err12 = sorted(stress_err[:, 1], key=lambda x: abs(x))
stress_err22 = sorted(stress_err[:, 2], key=lambda x: abs(x))

# ---------------------------------------------------------------------------------------------
#       FIGURE CONFIGURATION
font_1 = {'weight': 'normal', 'size': 23}
font_2 = {'weight': 'normal', 'size': 20}
font_3 = {'weight': 'normal', 'size': 18}
font_4 = {'weight': 'normal', 'size': 16}
font_5 = {'weight': 'normal', 'size': 14}

kdeConfigDic = {'shade': True,
                # 'multiple': "stack",
                'palette': "crest",
                'linewidth': 3,
                'bw_adjust': 2.0,
                'alpha': 0.6}
tickParamsDic = {'which': 'major', 'direction': 'out', 'length': 6, 'width': 1.5, 'labelsize': 16}
legendDic = {'prop': font_4, 'fancybox': True, 'framealpha': 0.1,
             }

# ---------------------------------------------------------------------------------------------
#       Error of prediction stress
#
# # plt.style.use('seaborn-paper')
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # np.random.seed(0)
# # x=np.random.randn(100)
#
# sns.kdeplot(x=stress_err11[:int(len(stress_err11)*0.9)],
#             label='$\sigma_{11}$',
#             **kdeConfigDic)
# sns.kdeplot(x=stress_err22[:int(len(stress_err11)*0.9)],
#             label='$\sigma_{22}$',
#             **kdeConfigDic)
# sns.kdeplot(x=stress_err12[:int(len(stress_err11)*0.9)],
#             label='$\sigma_{12}$',
#             **kdeConfigDic)
# ax.set_xlabel(r'Relative error', fontdict=font_3)
# ax.set_ylabel(r'Density', fontdict=font_3)
# ax.tick_params(axis='x', **tickParamsDic)
# ax.tick_params(axis='y', **tickParamsDic)
# ax.set_yticks([0., 0.4, 0.8, 1.2, 1.6, 2.0])
# ax.legend(**legendDic)
# plt.xlim([-1, 1])
# plt.ylim([0, 2])
# plt.tight_layout()
# plt.show()
# # plt.savefig("./stressErrDis.png", dpi=200)
# # plt.close()


colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
          '#d62728', '#9467bd', '#8c564b',
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
errs = [np.abs(np.array(stress_err11[:int(len(stress_err11)*0.9)])),
                np.abs(np.array(stress_err22[:int(len(stress_err11)*0.9)])),
                np.abs(np.array(stress_err12[:int(len(stress_err11)*0.9)]))]
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(errs)):
    colorDic = dict(color=colors[i], linewidth=3)
    ax.boxplot(x=errs[i], positions=[i+1], widths=[0.3], notch=False, sym='',
               showcaps=True,
               boxprops=colorDic, medianprops=colorDic, whiskerprops=colorDic, capprops=colorDic)
ax.set_xticklabels(['$\sigma_{11}$', '$\sigma_{22}$', '$\sigma_{12}$'],
                    rotation=0, fontsize=20)
ax.tick_params(axis='x', **tickParamsDic)
ax.tick_params(axis='y', **tickParamsDic)
plt.ylim([0.015, 2])
plt.yscale('log')
plt.tight_layout()
plt.savefig("./stressErrDisBox.png", dpi=200)

# ---------------------------------------------------------------------------------------------
#       Error of prediction tangent operator

tangent_err1111 = sorted(tangent_err[:, 0], key=lambda x: abs(x))
tangent_err1212 = sorted(tangent_err[:, 3], key=lambda x: abs(x))
tangent_err2222 = sorted(tangent_err[:, 5], key=lambda x: abs(x))
tangent_err1112 = sorted(tangent_err[:, 1], key=lambda x: abs(x))
tangent_err1122 = sorted(tangent_err[:, 2], key=lambda x: abs(x))
tangent_err1222 = sorted(tangent_err[:, 4], key=lambda x: abs(x))

tangentErrList = [tangent_err1111,
                  tangent_err1212,
                  tangent_err2222,
                  tangent_err1122,
                  tangent_err1112[:int(len(tangent_err1112)*0.9)],
                  tangent_err1222[:int(len(tangent_err1112)*0.9)]]
labelList = [1111, 1212, 2222, 1122, 1112, 1222]
#
# # plt.style.use('seaborn-paper')
# fig = plt.figure()
# ax = fig.add_subplot(111)
#
# # for i in range(len(tangentErrList)):
# for i in range(6):
#     x = tangentErrList[i]
#     sns.kdeplot(x=x, **kdeConfigDic, label='$D_{%d}$' % labelList[i])
#
# ax.set_xlabel(r'Relative error', fontdict=font_3)
# ax.set_ylabel(r'Density', fontdict=font_3)
# ax.tick_params(axis='x', **tickParamsDic)
# ax.tick_params(axis='y', **tickParamsDic)
# ax.set_yticks([0., 0.4, 0.8, 1.2, 1.6, 2.0])
# plt.legend(**legendDic)
# plt.xlim([-1, 1])
# plt.ylim([0, 2])
# plt.tight_layout()
# plt.show()
# plt.savefig("./tangentErrDis.png", dpi=200)
# plt.close()

errs = [np.abs(np.array(tangent_err1111)),
      np.abs(np.array(tangent_err1212)),
      np.abs(np.array(tangent_err2222)),
      np.abs(np.array(tangent_err1122)),
      np.abs(np.array(tangent_err1112[:int(len(tangent_err1112)*0.9)])),
      np.abs(np.array(tangent_err1222[:int(len(tangent_err1112)*0.9)]))]
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(errs)):
    colorDic = dict(color=colors[i], linewidth=3)
    ax.boxplot(x=errs[i], positions=[i+1], widths=[0.3], notch=False, sym='',
               showcaps=True,
               boxprops=colorDic, medianprops=colorDic, whiskerprops=colorDic, capprops=colorDic)
ax.set_xticklabels(['$D_{1111}$', '$D_{1212}$', '$D_{2222}$',
                    '$D_{1122}$', '$D_{1112}$', '$D_{1222}$'],
                    rotation=0, fontsize=20)
ax.tick_params(axis='x', **tickParamsDic)
ax.tick_params(axis='y', **tickParamsDic)
plt.ylim([0.003, 2])
plt.yscale('log')
plt.tight_layout()
plt.savefig("./tangentErrDisBox.png", dpi=200)
