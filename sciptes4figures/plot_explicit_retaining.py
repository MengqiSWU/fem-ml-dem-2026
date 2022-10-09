import matplotlib.pyplot as plt
from sciptes4figures.utils_plot import readTopForce_biaxial, configurations, get_color_list
import numpy as np

# -------------------------------------------------------------------------------------
#       FIGURE CONFIGURATION
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
color_list = get_color_list()

pathList = [
    # CSUH
    # '../../simu/explicit/retaining/retaining_explicit_csuh_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_timestep2.0e-04',
    '../../simu/explicit/retaining/retaining_explicit_csuh_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_theta8_timestep4.0e-04',
    # mldem -> csuh
    '../../simu/explicit/retaining/retaining_explicit_mldem_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep4.0e-04',
    '../../simu/explicit/retaining/retaining_explicit_mldem_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep4.0e-04',
    '../../simu/explicit/retaining/retaining_explicit_mldem_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_6_timestep4.0e-04',
    # mixed -> csuh
    # '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep4.0e-04',
    # '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_1_timestep4.0e-04',
    # '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_2_timestep4.0e-04',
    # '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep4.0e-04',
    # '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_4_timestep4.0e-04',

    # vonmises
    #     '../../simu/explicit/retaining/retaining_explicit_vonmises_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5',
    # ml-> vonmises
    # '../../simu/explicit/retaining/retaining_explicit_mldem_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0',
    # '../../simu/explicit/retaining/retaining_explicit_mldem_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dd20_Fourier_noRotate_von_all_0',
    # '../../simu/explicit/retaining/retaining_explicit_mldem_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDqH_Y_sigANDH_dd20_Fourier_noRotate_von_all_5',
    # '../../simu/explicit/retaining/retaining_explicit_mldem_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_5',
    # '../../simu/explicit/retaining/retaining_explicit_mldem_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDqH_Y_sigANDH_md20_Fourier_noRotate_von_all_5',

    # mixed -> vonmises
    # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0',
    # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_1',
    # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_2',
    #     '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_4',

    # 2ml
    # '../../simu/explicit/retaining/retaining_explicit_2ml_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5',
    # '../../simu/explicit/retaining/retaining_explicit_2ml_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_epsANDH',

]

plt.style.use('seaborn-paper')
fig = plt.figure(figsize=[10, 10])
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
label_list = [
    # 'damp_0',

    'CSUH',
    r'$\mathcal{NN}$ 0',
    r'$\mathcal{NN}$ 3',
    r'$\mathcal{NN}$ 6',
    r'$\mathcal{NN}$',
    r'$\mathcal{NN}$',
    r'$\mathcal{NN}$',
# r'$\mathcal{NN}$',
# r'$\mathcal{NN}_{dd40}^{CSUH}$',
# r'$\mathcal{NN}_{dd60}^{CSUH}$',
# r'$\mathcal{NN}_{dd100}^{CSUH}$',
# r'$\mathcal{NN}_{ddd40}^{CSUH}$',
    # r'$\mathcal{NN}_{dd20}^{absxy}$',
    r'Double $\mathcal{NN}_{dd20}^{H}$',
    # r'Double $\mathcal{NN}_{dd20}^{absxy}$',
    # r'$\mathcal{NN}_{dmdd40}^{5}$',
    # r'$\mathcal{NN}_{dm20}^{5}$',
    # r'$\mathcal{NN}^{H}$',
    # r'$\mathcal{NN}^{H} 1$',
    # 'Hybrid 1 ',
    # 'Hybrid 2',
    # 'Hybrid 3'
    # 'Hybrid 5'
]
for i, path in enumerate(pathList):
    datas, label = readTopForce_biaxial(path=path, split_keyword='safe0.4')
    if "NN" not in label_list[i]:
        n = len(datas[:, 0])
        plot_index = np.arange(0, n, n//20)
        ax1.scatter(-datas[:, 0][plot_index], -datas[:, 1][plot_index] / 1e3, label=label_list[i], c=color_list[i], s=70)
        ax2.scatter(-datas[:, 0][plot_index], datas[:, 3][plot_index], c=color_list[i], s=70)
        # ax3.scatter(-datas[:, 0][plot_index], np.log(datas[:, 4][plot_index]), c=color_list[i], s=50)
        # ax4.scatter(-datas[:, 0][plot_index], datas[:, 6][plot_index] + datas[:, 7][plot_index] + datas[:, 9][plot_index], c=color_list[i])
        # ax4.scatter(-datas[:, 0][plot_index], datas[:, 8][plot_index], c=color_list[i])
    else:
        ax1.plot(-datas[:, 0], -datas[:, 1]/1e3, label=label_list[i], c=color_list[i], linewidth=3)
        ax2.plot(-datas[:, 0], datas[:, 3], c=color_list[i], linewidth=3)
    ax3.semilogy(-datas[:, 0], datas[:, 4], c=color_list[i],  linewidth=3)
        # ax4.plot(-datas[:, 0], datas[:, 6]+datas[:, 7]+datas[:, 9], c=color_list[i], linewidth=3)
        # ax4.plot(-datas[:, 0], datas[:, 8], c=color_list[i], linewidth=3)
    i+=1
for i, aax in enumerate([ax1, ax2, ax3]):
    aax.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16)
    aax.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16)
    if i != 2:
        aax.xaxis.set_ticklabels([])

ax1.legend(**legendDic)
ax1.set_ylabel(r'Wall force (kN)', fontdict=font_3)
ax2.set_ylabel(r'$\epsilon_{v}$', fontdict=font_2)
ax3.set_ylabel(r'Maximum $a$', fontdict=font_2)
ax3.set_xlabel(r'Axial strain', fontdict=font_3)

plt.tight_layout()
plt.show()
# plt.savefig('../../simu/explicit2D_biaxial_compression/topforce_expilict.png', dpi=200)
