import matplotlib.pyplot as plt
import os

import numpy
import numpy as np
from sciptes4figures.utils_plot import configurations, get_num_error_prediction

# -------------------------------------------------------------------------------------
#       FIGURE CONFIGURATION
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()

path_list = [
    # ml->csuh
    [
        # biaxial
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep2.0e-04',
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_1_timestep2.0e-04',
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_2_timestep2.0e-04',
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep2.0e-04',
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_4_timestep2.0e-04',
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_5_timestep2.0e-04',
     ],
    [
        # retaining
    '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep4.0e-04',
    '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_1_timestep4.0e-04',
    '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_2_timestep4.0e-04',
    '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep4.0e-04',
    '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_4_timestep4.0e-04',
    '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_5_timestep4.0e-04',
    ],
    [
        # footing
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep4.0e-04_b0.30',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_1_timestep4.0e-04_b0.30',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_2_timestep4.0e-04_b0.30',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep4.0e-04_b0.30',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_4_timestep4.0e-04_b0.30',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_5_timestep4.0e-04_b0.30',
    ],

    # [# ml -> vonmises
    # # biaxial
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_1',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_2',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_3',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_4',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_5',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNactive_0',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_1',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_2',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_3',
    # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_4',
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_4_long',
    # ],
    #
    # [# retaining
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all',
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_1',
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_2',
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_3',
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_4',
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_5',
    # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0',
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_1',
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_2',
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_3',
    # # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_4',
    # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_4_long',
    # ],
    # [# footing
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all',
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_1',
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_2',
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_3',
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_4',
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_5',
    # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0',
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_1',
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_2',
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_3',
    # # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_4',
    # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_4_long',
    #     ],
]

title_list = [
    'biaxial',
    'retaining',
    'footing']

error_point_num_list = []
for j in range(0, 3):
    title = title_list[j]
    num_temp_list = []
    steps, nums = [], []
    for path_temp in path_list[j]:
        step_sorted, num_sorted = get_num_error_prediction(path_temp=path_temp)
        steps.append(step_sorted)
        nums.append(num_sorted)
        num_temp_list.append(np.sum(num_sorted))
    error_point_num_list.append(num_temp_list)
    fig = plt.figure(dpi=200)
    plt.style.use('seaborn-paper')
    ax1 = fig.add_subplot(111)
    for i in range(len(path_list[j])):
        plt.scatter(steps[i], nums[i], label="Iteration %d" % i, s=10)

    for i, aax in enumerate([ax1]):
        aax.tick_params(**tickParamsDic)
        aax.tick_params(**tickParamsDic)
        if i != 0:
            aax.xaxis.set_ticklabels([])

    ax1.legend(**legendDic)
    ax1.set_ylabel(r'Number of points with high error', fontdict=font_4)
    ax1.set_xlabel(r'Loading step', fontdict=font_3)
    plt.title(label=title, fontdict=font_3)
    plt.tight_layout()
    plt.show()

# plot the total number summary
# fig = plt.figure()
# plt.style.use('seaborn-paper')
# ax1 = fig.add_subplot(111)
# for i, label in enumerate(title_list):
#     plt.plot(error_point_num_list[i], label=label)
#
# for i, aax in enumerate([ax1]):
#     aax.tick_params(**tickParamsDic)
#     aax.tick_params(**tickParamsDic)
#     if i != 0:
#         aax.xaxis.set_ticklabels([])
# ax1.legend(**legendDic)
# ax1.set_ylabel(r'Total number of points with high error', fontdict=font_4)
# ax1.set_xlabel(r'Iteration', fontdict=font_3)
# # plt.title(label=label, fontdict=font_3)
# plt.tight_layout()
# plt.show()

