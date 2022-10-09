import matplotlib.pyplot as plt
import numpy as np

from sciptes4figures.utils_plot import readTopForce_biaxial, configurations, get_color_list

# -------------------------------------------------------------------------------------
#       FIGURE CONFIGURATION
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
color_list = get_color_list()

pathList = [
    # EB
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_eb_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_fric_10.0',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_eb_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_fric_20.0',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_eb_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_fric_30.0',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_eb_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_fric_45.0',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_eb_order1_numg484_biaxial_0.05_548_vel0.20_damp1.0e+06_safe0.5',

    # csuh
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_csuh_intorder1_numg128_biaxial_0.1_162_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_timestep2.0e-04',
    '../../simu/explicit/biaxial/biaxial_rough_explicit_csuh_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_theta8_timestep2.0e-04',
    #mldem-> csuh
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mldem_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep2.0e-04',
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mldem_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep2.0e-04',
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mldem_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_6_timestep2.0e-04',
    # mixed ->csuh
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep2.0e-04',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_1_timestep2.0e-04',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_2_timestep2.0e-04',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_4_timestep2.0e-04',

    # vonmises model
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_save_order1_numg128_biaxial_0.1_162_vel0.20_safe0.5',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_safe0.5_test',
    #     '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5',

    # damp test
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp100000.0_safe0.5_test',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp200000.0_safe0.5_test',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp500000.0_safe0.5_test',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp600000.0_safe0.5_test',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp10000000.0_safe0.5_test',
    # '../../simu/biaxial/biax_rough_explicit_vonmises_x9_y18_order1_numg648_rate0.8',
    # '../../simu/biaxial/biax_rough_explicit_vonmises_x2_y4_order1_numg32_rate0.8',
    # '../../simu/biaxial/biax_rough_explicit_vonmises_x3_y6_order1_numg72_rate2.00',
    # '../../simu/biaxial/biax_rough_explicit_vonmises_x3_y6_order1_numg72_rate2.00_safe0.4',
    # '../../simu/biaxial/biax_rough_explicit_vonmises_x3_y6_order1_numg72_rate2.00_safe0.5',
    # '../../simu/biaxial/biax_rough_explicit_vonmises_x9_y18_order1_numg648_rate0.20_safe0.4',
    # '../../simu/biaxial/biax_rough_explicit_vonmises_order1_numg384_biaxial_0.1_162_rate0.20_safe0.4',
    # '../../simu/biaxial/biax_rough_explicit_vonmises_order1_numg114_biaxial_rate0.20_safe0.4',
    # mldem-> vonmises
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mldem_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mldem_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dd20_Fourier_noRotate_von_all_0',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mldem_order1_numg128_biaxial_0.1_162_vel0.20_safe0.5_NNX_epsANDabsy_Y_sig_dmdd40_Fourier_noRotate_von_all',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mldem_order1_numg128_biaxial_0.1_162_vel0.20_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mldem_order1_numg128_biaxial_0.1_162_vel0.20_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_mix_all_1',
    # 2 ml
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_2ml_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_2ml_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_epsANDH',

    # mixed (ml->vonmises)
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_safe0.5_NNX_eps_Y_sig_dmdd40_Fourier_noRotate_von_all'
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all'
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all'

    # norsand model
    # '../../simu/biaxial/biax_smooth_explicit_norsand_x2_y4_order1_numg32_rate1.0_p100kPa_e0.833',
    # '../../simu/biaxial/biax_smooth_explicit_norsand_x2_y4_order1_numg32_rate1.0_p100kPa_e0.700',
    # '../../simu/biaxial/biax_smooth_explicit_norsand_x2_y4_order1_numg32_rate1.0_p100kPa_e0.600',

    # DEM & CSUH
    # '../../simu/explicit/biaxial/biax_rough_explicit_csuh_order1_numg484_biaxial_0.05_548_rate0.80_safe0.5',
    # '../../simu/biaxial/biaxial_explicit_rate1_smooth_dem_x2_y4_2D_order1_numG32',
    # '../../simu/biaxial/biaxial_explicit_rate0.500_smooth_csuh_x2_y4_2D_order2_numG32',
    # '../../simu/biaxial/biaxial_explicit_rate0.500_smooth_csuh_ocr20.0_x2_y4_2D_order2_numG32',
    # '../../simu/biaxial/biaxial_explicit_rate0.500_smooth_csuh_ocr40.0_x2_y4_2D_order2_numG32',
    # '../../simu/biaxial/biaxial_explicit_rate0.500_smooth_csuh_ocr80.0_x2_y4_2D_order2_numG32',
    # '../../simu/biaxial/biaxial_explicit_rate0.500_smooth_csuh_ocr80.0_x9_y18_2D_order1_numG648',
    # '../../simu/biaxial/biaxial_explicit_rate0.500_rough_csuh_ocr80.0_x9_y18_2D_order1_numG648',
]

plt.style.use('seaborn-paper')

fig = plt.figure(figsize=[10, 10])
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
# ax4 = fig.add_subplot(414)
i= 0
label_list = [
    'CSUH',
    r'$\mathcal{NN}$ 0',
    r'$\mathcal{NN}$ 3',
    r'$\mathcal{NN}$ 6',
    'IME',
    r'$\mathcal{NN}$',
# r'$\mathcal{NN}_{dd20}^{CSUH}$',
# r'$\mathcal{NN}_{dd40}^{CSUH}$',
# r'$\mathcal{NN}_{dd60}^{CSUH}$',
# r'$\mathcal{NN}_{dd100}^{CSUH}$',
# r'$\mathcal{NN}_{ddd40}^{CSUH}$',
# # r'$\mathcal{NN}_{ddd40}^{mixed}$',
#     'EB_128 $f=10^{\circ}$',
#     'EB_128 $f=20^{\circ}$',
#     'EB_128 $f=30^{\circ}$',
#     'EB_128 $f=45^{\circ}$',
#     'EB_484',
    'Von-mises',
#     r'$\mathcal{NN}$ Von-mises',
    # r'$\mathcal{NN}$ CSUH',
    # r'$\mathcal{NN}^{absy}$',
    # r'$\mathcal{NN}_{dd20}^{5}$',
    # r'$\mathcal{NN}_{dmdd40}^{5}$',
    # r'$\mathcal{NN}^{H1}$',
    # r'$\mathcal{NN}^{absxy}$',
    r'$\mathcal{NN}^{H}$',
    r'Double $\mathcal{NN}^{H}$',
    # 'Hybrid 1 H',
    # 'Hybrid 2',
    # 'Hybrid 3'
    # 'CSUH $ocr=20.0$',
    # 'CSUH $ocr=40.0$',
    # 'CSUH $ocr=80.0$',
    # 'CSUH $ocr=120.0$',
]
# label_list = [ # damp
#     'damp_0',
#     'damp_1e5',
#     'damp_2e5',
#     'damp_5e5',
#     'damp_6e5',
#     'damp_1e6',
#     'damp_1e7',



# ]
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
ax1.set_ylabel(r'Top force (kN)', fontdict=font_3)
ax2.set_ylabel(r'$\epsilon_{v}$', fontdict=font_2)
ax3.set_ylabel(r'Maximum $a$', fontdict=font_2)
# ax4.set_ylabel(r'Energy', fontdict=font_3)
ax3.set_xlabel(r'Axial strain', fontdict=font_3)

plt.tight_layout()
plt.show()
# plt.savefig('../../simu/explicit2D_biaxial_compression/topforce_expilict.png', dpi=200)
