import matplotlib.pyplot as plt
import numpy as np

from sciptes4figures.utils_plot import readTopForce_biaxial, configurations, get_color_list

# -------------------------------------------------------------------------------------
#       FIGURE CONFIGURATION
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()

color_list = get_color_list()

pathList = [
    # csuh
    # '../../simu/explicit/footing/footing_explicit_csuh_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_timestep4.0e-04_b0.30_theta13',
    '../../simu/explicit/footing/footing_explicit_csuh_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_theta8_timestep4.0e-04_b0.30',

    # mldem -> csuh
    '../../simu/explicit/footing/footing_explicit_mldem_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep4.0e-04_b0.30',
    '../../simu/explicit/footing/footing_explicit_mldem_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep4.0e-04_b0.30',
    '../../simu/explicit/footing/footing_explicit_mldem_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_6_timestep4.0e-04_b0.30',
    # mixed -> csuh
    # '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep4.0e-04_b0.30',
    # '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_1_timestep4.0e-04_b0.30',
    # '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_2_timestep4.0e-04_b0.30',
    # '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep4.0e-04_b0.30',
    # '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_4_timestep4.0e-04_b0.30',

    # ml -> vonmises
    # '../../simu/explicit/footing/footing_explicit_mldem_intorder1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_biax_retain_0',
    # '../../simu/explicit/footing/footing_explicit_mldem_intorder1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_biax_retain_addedFooting_1',
    # '../../simu/explicit/footing/footing_explicit_mldem_intorder1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_biax_retain_addedFooting_2',
    # vonmises
    # '../../simu/explicit/footing/footing_explicit_vonmises_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5',
    # '../../simu/explicit/footing/footing_explicit_vonmises_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5',
    # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0',
    # # mixed -> vonmises
    # '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_biax_retain_0',
    # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.20_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all',
    # 2ml
    # '../../simu/explicit/footing/footing_explicit_2ml_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5',
    # '../../simu/explicit/footing/footing_explicit_2ml_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_epsANDH',

    # # DEM
    # '../../simu/explicit/footing/footing_explicit_dem_order1_numg480_footing552_vel0.40_damp1.0e+06_safe0.5',
    # CSUH-> DEM
    # '../../simu/explicit/footing/footing_explicit_csuh_order1_numg480_footing552_vel0.40_damp1.0e+06_safe0.5_p100kPa_ocr_48.1',
    # '../../simu/explicit/footing/footing_explicit_csuh_order1_numg480_footing552_vel0.40_damp1.0e+06_safe0.5_p100kPa_ocr_48.1',
    # '../../simu/explicit/footing/footing_explicit_csuh_order1_numg480_footing552_vel0.40_damp1.0e+06_safe0.5_p100kPa_ocr_489.1_confining1.0e+05',

]
# label_list = ['Von-mises', 'Hybrid 1', 'Hybrid 2', 'Hybrid 3']
label_list = [
    # 'damp_0',
    # 'damp_1e6',
    'CSUH',
r'$\mathcal{NN}$ 0',
r'$\mathcal{NN}$ 3',
r'$\mathcal{NN}$ 6',
r'$\mathcal{NN}$',
r'$\mathcal{NN}$',
r'$\mathcal{NN}$',
# r'$\mathcal{NN}_{dd40}^{CSUH}$',
# r'$\mathcal{NN}_{dd60}^{CSUH}$',
# r'$\mathcal{NN}_{dd100}^{CSUH}$',
# r'$\mathcal{NN}_{ddd40}^{CSUH}$',
#     'dem',
#     'ml->csuh 552',
#     'ml->csuh 552',
#     r'$\mathcal{NN}$ iteration 0',
#     r'$\mathcal{NN}$ iteration 1',
#     r'$\mathcal{NN}$ iteration 2',

    'IME',
    # 'Von-mises_1',
    # r'$\mathcal{NN}_{qH}^{5}$',
    # r'$\mathcal{NN}_{absxy}^{dd20}$',
    # r'Double $\mathcal{NN}^{H}$',
    # r'Double $\mathcal{NN}_{absxy}^{dd20}$',
    # r'$\mathcal{NN}^{absy}$',
    # r'$\mathcal{NN}^{H}$',
    # 'Hybrid 1 H',
    # 'Hybrid 2',
    # 'Hybrid 3'
]
plt.style.use('seaborn-paper')

fig = plt.figure(figsize=[10, 10])
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
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
    aax.tick_params(**tickParamsDic)
    aax.tick_params(**tickParamsDic)
    if i != 2:
        aax.xaxis.set_ticklabels([])

ax1.legend(**legendDic)
ax1.set_ylabel(r'Footing force (kN)', fontdict=font_3)
ax2.set_ylabel(r'$\epsilon_{v}$', fontdict=font_2)
ax3.set_ylabel(r'Maximum $a$', fontdict=font_2)
ax3.set_xlabel(r'Axial strain', fontdict=font_3)

plt.tight_layout()
plt.show()
# plt.savefig('../../simu/explicit2D_biaxial_compression/topforce_expilict.png', dpi=200)
