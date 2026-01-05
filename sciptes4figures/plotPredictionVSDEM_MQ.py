# import pandas as pd

# from FEMxML.getSeriesDataConverged import getSeriasData, getSeriasDataCoupling
from FEMxML.utils_ml import get_data, pickle_load, plot_prection
import matplotlib.pyplot as plt
import numpy as np
# from FEMxML.netTorchLastDouble import Net
from FEMxML.torch_restore import modelRestore
import os
# import seaborn as sns

returned_dict = get_data(root_path_list=[
    '../../simu/biaxial_0.08/biax_rough_implicit_dem_intorder1_numg32_x2_y4_t4',
    # '/home/shguan/simu/ABS_DEM1_2_4_biaxial',
    # '/home/shguan/simu/ABS_DEM2_2_4_biaxial',
    # '/home/shguan/simu/ABS_DEM_8_16_biaxial'
    # '/home/shguan/simu/Right_DEM_2_4_biaxial',
    # '/home/shguan/simu/Right_DEM_8_16_biaxial',
], maxTime=80)
print('Read data finished!')
strain, H_3F, stress, tangent = returned_dict['eps'], returned_dict['H_3F'], returned_dict['sig'], returned_dict['tangent']
# stress, tangent = y_data[:, :3], y_data[:, 3:]
x_data = np.concatenate((strain, H_3F),axis=1)


# net restore
ml_model_path = os.path.join('../FEMxML/biax_ml_1e5/Trial4', 'X_epsAND3f_Y_D_ddd10_Fourier_noRotate_FEM_DEM_D')
# ml_model_path = os.path.join('../FEMxML/biax_ml_1e5/Trial4', 'X_epsAND3f_Y_sig_ddd10_Fourier_noRotate_FEM_DEM_sig')

net = modelRestore(savedPath=ml_model_path, trainFlag=False)
# print(ml_model_path, net.input_mean[0])

tangent_predict = net.get_prediction(x_data)
stress_predict = net.get_prediction(x_data)

# np.savetxt("tangent_predict.csv", tangent_predict, delimiter=',')
# np.savetxt("tangent.csv", tangent, delimiter=',')

np.savetxt("stress_predict.csv", stress_predict, delimiter=',')
np.savetxt("stress.csv", stress, delimiter=',')


np.random.seed(15)
x = np.random.permutation(range(len(stress)))
index = list(x[:200])



# tangent = tangent[index]
# stress = stress[index]

# tangent_predict = tangent_predict[index]
stress_predict = stress_predict[index]

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
# fig = plt.figure()
# ax = fig.add_subplot(111)

# minn, maxx = np.min(stress), np.max(stress)
# plt.plot([minn, maxx], [minn, maxx], 'r--', alpha=0.5)
# for i in range(3):
#     ax.scatter(stress[:, i], stress_predict[:, i],
#                s=50,
#                edgecolors='k',
#                marker=markList[i],
#                alpha=0.5,
#                label='$\sigma_{%d}$' % stressLabelList[i])
#
# ax.tick_params(axis='x', **tickParamsDic)
# ax.tick_params(axis='y', **tickParamsDic)
# ax.set_xticks([0, -1e5, -2e5, -3e5])
# ax.set_yticks([0, -1e5, -2e5, -3e5])
# plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
# plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
# ax.set_xlabel(r'DEM simulation', fontdict=font_3)
# ax.set_ylabel(r'Network prediction', fontdict=font_3)
# ax.legend(**legendDic)
# plt.tight_layout()
# # plt.show()
# plt.savefig("./stressPrediction.png", dpi=200)

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
plt.show()
# plt.savefig("./tangentPrediction.png", dpi=200)