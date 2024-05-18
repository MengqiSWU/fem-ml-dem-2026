# import pandas as pd

from FEMxML.getSeriesDataConverged import getSeriasData, getSeriasDataCoupling
from FEMxML.train_model_strain_lastDouble import get_data, pickle_load, plot_prection
import matplotlib.pyplot as plt
import numpy as np
from FEMxML.netTorchLastDouble import Net
from FEMxML.netTorch import modelRestore
import os
import seaborn as sns

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
ml_model_path = os.path.join('../FEMxML', 'ptModelH4_30_9_Lastdouble_without')

net = modelRestore(savedPath=ml_model_path, trainFlag=False)
print(ml_model_path, net.input_mean[0])
stress_predict, tangent_predict = net.get_stressAndStiffness(x_data)

np.random.seed(1)
x = np.random.permutation(range(len(stress)))
index = list(x[:100])
stress = stress[index]
tangent = tangent[index]
stress_predict = stress_predict[index]
tangent_predict = tangent_predict[index]

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

# ----------------------------------------------------------------------------------
markList = ['v', '^', '>', 'o', '<', 'h']
stressLabelList = [11, 12, 22]
tangentlabelList = [1111, 1112, 1122, 1212, 1222, 2222]

#%%
# ----------------------------------------------------------------------------------
#       Stress prediction
fig = plt.figure()
ax = fig.add_subplot(111)

minn, maxx = np.min(stress), np.max(stress)
plt.plot([minn, maxx], [minn, maxx], 'r--', alpha=0.5)
for i in range(3):
    ax.scatter(stress[:, i], stress_predict[:, i],
               s=50,
               edgecolors='k',
               marker=markList[i],
               alpha=0.5,
               label='$\sigma_{%d}$' % stressLabelList[i])

ax.tick_params(axis='x', **tickParamsDic)
ax.tick_params(axis='y', **tickParamsDic)
ax.set_xticks([0, -1e5, -2e5, -3e5])
ax.set_yticks([0, -1e5, -2e5, -3e5])
plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.set_xlabel(r'DEM simulation', fontdict=font_3)
ax.set_ylabel(r'Network prediction', fontdict=font_3)
ax.legend(**legendDic)
plt.tight_layout()
# plt.show()
plt.savefig("./stressPrediction.png", dpi=200)

# ----------------------------------------------------------------------------------
#       Tangent operator prediction
fig = plt.figure()
ax = fig.add_subplot(111)

minn, maxx = np.min(tangent), np.max(tangent)
plt.plot([minn, maxx], [minn, maxx], 'r--', alpha=0.5)
for i in range(6):
    ax.scatter(tangent[:, i], tangent_predict[:, i],
               s=50,
               edgecolors='k',
               marker=markList[i],
               alpha=0.5,
               label='$D_{%d}$' % tangentlabelList[i])

ax.tick_params(axis='x', **tickParamsDic)
ax.tick_params(axis='y', **tickParamsDic)
ax.set_xlabel(r'DEM simulation', fontdict=font_3)
ax.set_ylabel(r'Network prediction', fontdict=font_3)
ax.set_yticks([0, 1e7, 2e7, 3e7])

ax_new = fig.add_axes([0.6, 0.2, 0.35, 0.35])  # left, bottom, width, height
plt.plot([minn, maxx], [minn, maxx], 'r--', alpha=0.5)
for i in range(6):
    ax_new.scatter(tangent[:, i], tangent_predict[:, i],
               s=50,
               edgecolors='k',
               marker=markList[i],
               alpha=0.5,
               label='$D_{%d}$' % tangentlabelList[i])

plt.xlim([-0.15e7, 0.22e7])
plt.ylim([-0.15e7, 0.22e7])

ax_new.set_xticks([])
ax_new.set_yticks([])
ax.legend(**legendDic, ncol=2)
plt.tight_layout()
# plt.show()
plt.savefig("./tangentPrediction.png", dpi=200)