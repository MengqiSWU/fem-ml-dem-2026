from FEMxML.utils_ml import *
# from FEMxML.netTorchLastDouble import modelRestore
# from FEMxML.netTorchLastDouble import Net
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sciptes4figures.utils_plot import configurations

from matplotlib import pyplot
from numpy import arange, random, concatenate, square, subtract
import matplotlib as mpl
# import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


#### predict mesh 5*10
# returned_dict_dem = get_data_series(
#     root_path_list=['../../simu/biaxial_0.08/biax_rough_implicit_dem_intorder1_numg200_x5_y10_t10_t12/iteration_gauss'],
#     numg=200)
# returned_dict_ml = get_data_series(
#     root_path_list=[
#         '../../simu/biaxial_0.08/biax_rough_implicit_mldem_intorder1_numg200_x5_y10_NNX_epsAND3f_Y_sig_dmd12_Fourier_noRotate_FEM_DEM_sig_NEICHA5_savedata/iteration_gauss'],
#     numg=200)






#### predict mesh 10*20
returned_dict_dem = get_data_series(
    root_path_list=['../../simu/biaxial_Reld/biax_rough_implicit_dem_intorder1_numg800_x10_y20_Reld_St12_2H_final_use/iteration_gauss'],
    numg=800)
returned_dict_ml = get_data_series(
    root_path_list=[
        '../../simu/biaxial_Reld/biax_rough_implicit_mldem_intorder1_numg800_x10_y20_NNX_epsAND3f_Y_sig_ddd14_Fourier_noRotate_FEM_DEM_sig_Reld_St12_2H_final_use/iteration_gauss'],
    numg=800)





strain_dem, stress_dem, strain_abs_dem, stress_last_dem = returned_dict_dem['eps'], returned_dict_dem['sig'], \
                                                          returned_dict_dem['eps_abs'], returned_dict_dem['sig_last']

strain_ml, stress_ml, strain_abs_ml, stress_last_ml = returned_dict_ml['eps'], returned_dict_ml['sig'], \
                                                      returned_dict_ml['eps_abs'], returned_dict_ml['sig_last']


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'NSimSun'

def plot_case11(strain1, strain2, stress1, stress2, indexOfPoint):

    ax = pyplot.gca()
    fig = pyplot.gcf()
    fig.set_size_inches(6, 5)
    ax.grid(True, linestyle="--")
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    x_major_locator = MultipleLocator(0.0015)
    y_major_locator = MultipleLocator(0.01)
    # #
    ax.xaxis.set_major_locator(x_major_locator)  # 手动设置刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)  # 手动设置刻度间隔

    pyplot.xlim([-0.001, 0.0085])
    ax.set_ylim(-0.14, -0.09)
    ax.ticklabel_format(style='plain', scilimits=(-1, -1), axis='y')
    ax.set_xlabel('$\epsilon_{11}$',fontsize =12)
    ax.set_ylabel('$\sigma_{11}$ (MPa)',fontsize =12)

    ax.plot(strain1[72, ::10, 0], stress1[72, ::10, 0] / 1000000, linestyle='-', color='mediumspringgreen', linewidth=3,
                    label='$\sigma_{11}-DEM$')
    # ax.plot(strain1[3, ::5, 0], stress1[3, ::5, 0] / 1000000, linestyle='-', color='mediumspringgreen',linewidth=3,)

    ax.plot(strain2[72, ::10, 0], stress2[72, ::10, 0] / 1000000, linestyle='-', color='r', marker='o', markerfacecolor='none',
                    markersize=4, linewidth=1., label='$\sigma_{11}-ML$')
    # ax.plot(strain2[3,::5, 0],  stress2[3, ::5, 0] / 1000000, linestyle='-', color='r', marker='o',markerfacecolor='none',markersize=4, linewidth=1.,)

        # ax.plot(_x, _y_hat/ 1000, linestyle='--', color='b', marker='o', markerfacecolor='none', markersize=3,
        #             linewidth=1.5, label='$\sigma_{11}-GRU$')
    ax.legend(loc=(0.3, 0.7), frameon=True, ncol=2,fontsize =12)
    pyplot.savefig('./NO70_11_Reld_coarse.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.02)
    pyplot.show()
    pyplot.close()


def plot_case12(strain1, strain2, stress1, stress2, indexOfPoint):

    ax = pyplot.gca()
    fig = pyplot.gcf()
    fig.set_size_inches(6, 5)
    ax.grid(True, linestyle="--")
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    x_major_locator = MultipleLocator(0.0004)
    y_major_locator = MultipleLocator(0.006)
    #
    ax.xaxis.set_major_locator(x_major_locator)  # 手动设置刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)  # 手动设置刻度间隔
    #
    # ax.set_xlim([-0.006, 0.005])
    ax.set_ylim([-0.002, 0.03])
    ax.ticklabel_format(style='plain', scilimits=(-1, -1), axis='y')
    ax.set_xlabel('$\epsilon_{12}$',fontsize =12)
    ax.set_ylabel('$\sigma_{12}$ (MPa)',fontsize =12)

    ax.plot(strain1[72, ::10, 1] , stress1[72, ::10, 1]  / 1000000, linestyle='-', color='deeppink', linewidth=3, label='$\sigma_{12}-DEM$')
    # ax.plot(strain1[3, ::5, 1], stress1[3, ::5, 1] / 1000000, linestyle='-', color='deeppink', linewidth=3,)
    ax.plot(strain2[72, ::10, 1] , stress2[72, ::10, 1]  / 1000000, linestyle='-', color='black', marker='o', markerfacecolor='None',
                    markersize=4, linewidth=1, label='$\sigma_{12}-ML$')
    # ax.plot(strain2[3, ::5, 1], stress2[3, ::5, 1] / 1000000, linestyle='-', color='black', marker='o',
    #         markerfacecolor='None', markersize=4, linewidth=1,)

    ax.legend(loc=(0.2, 0.2), frameon=True, ncol=2,fontsize =12)
    pyplot.savefig('./NO70_12_Reld_coarse.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.02)
    pyplot.show()
    pyplot.close()



def plot_case22(strain1, strain2, stress1, stress2, indexOfPoint):

    ax = pyplot.gca()
    fig = pyplot.gcf()
    fig.set_size_inches(6, 5)
    ax.grid(True, linestyle="--")
    ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
    x_major_locator = MultipleLocator(0.004)
    y_major_locator = MultipleLocator(0.05)
    #
    ax.xaxis.set_major_locator(x_major_locator)  # 手动设置刻度间隔
    ax.yaxis.set_major_locator(y_major_locator)  # 手动设置刻度间隔
    #
    ax.ticklabel_format(style='plain', scilimits=(-1, -1), axis='y')
    ax.set_xlim([-0.016, 0.0011])
    ax.set_ylim([-0.35, -0.1])
    ax.set_xlabel('$\epsilon_{22}$',fontsize =12)
    ax.set_ylabel('$\sigma_{22}$ (MPa)',fontsize =12)

    ax.plot(strain1[72, ::10, 2] , stress1[72, ::10, 2] / 1000000, linestyle='-', color='y', linewidth=3, label='$\sigma_{22}-DEM$')
    # ax.plot(strain1[3, ::5, 2], stress1[3, ::5, 2] / 1000000, linestyle='-', color='y', linewidth=3,)

    ax.plot(strain2[72, ::10, 2] , stress2[72, ::10, 2] / 1000000, linestyle='-', color='dodgerblue', marker='o', markerfacecolor='None',
                    markersize=4, linewidth=1, label='$\sigma_{22}-ML$')
    # ax.plot(strain2[3, ::5, 2], stress2[3, ::5, 2] / 1000000, linestyle='-', color='dodgerblue', marker='o',markerfacecolor='None',markersize=4, linewidth=1)


    ax.legend(loc=(0.15, 0.75), frameon=True, ncol=2,fontsize =12)
    pyplot.savefig('./NO70_22_Reld_coarse.png', format='png', dpi=600, bbox_inches='tight', pad_inches=0.02)
    pyplot.show()
    pyplot.close()

x = np.arange(0, 800, 1)
indexOfPoint = list(x[0:2])

plot_case11(strain_dem, strain_ml, stress_dem, stress_ml, indexOfPoint)
plot_case12(strain_dem, strain_ml, stress_dem, stress_ml, indexOfPoint)
plot_case22(strain_dem, strain_ml, stress_dem, stress_ml, indexOfPoint)





# , strainCoupling, stressCoupling,

# def plotCurve(strain1, stress1, strain2, stress2,indexOfPoint, mesh, file_dir='biaxial_gauss'):
#     font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic=configurations()

    # ------------------------------------------------------------------------------------------
    # 11 direction
    # plt.style.use('seaborn-paper')
    # fig = plt.figure(figsize=[8, 6])
    # # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    #
    # for num in indexOfPoint:
    #     # ax1.plot(range(len(strain[0])), strain[num, :, 0], linewidth=2, label='Point #%d' % num)
    #     # ax2.plot(range(len(stress[0])), stress[num, :, 0], linewidth=2, label='Point #%d' % num)
    #     ax1.plot(strain1[num, :, 0], stress1[num, :, 0], linewidth=2, label='Point #%d' % num)
    #     ax2.plot(strain2[num, :, 0], stress2[num, :, 0], linewidth=2, label='Point #%d' % num)
    #
    # ax1.set_ylabel(r'$\epsilon_{11}$', fontdict=font_3)
    # ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax2.set_xlabel(r'Loading step', fontdict=font_3)
    # ax2.set_ylabel(r'$\sigma_{11}$', fontdict=font_3)
    # ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax1.xaxis.set_ticklabels([])
    # # ax1.legend(prop=font_5, loc='lower left', bbox_to_anchor=(0.1, 0.01), fancybox='sawtooth', shadow=True,
    # #            markerscale=0.6, ncol=2)
    # ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    # # ax2.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    # #            markerscale=0.6, ncol=2)
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig('./%s/strainStress11_%s.png' % (file_dir, mesh), dpi=200)
    # plt.close()
    #
    # # # ------------------------------------------------------------------------------------------
    # # # 22 direction
    # plt.style.use('seaborn-paper')
    # fig = plt.figure(figsize=[8, 6])
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    #
    # for num in indexOfPoint:
    #     # ax1.plot(range(len(strain[0])), strain[num, :, 1], linewidth=2, label='Point #%d' % num)
    #     # ax2.plot(range(len(stress[0])), stress[num, :, 1], linewidth=2, label='Point #%d' % num)
    #     ax1.plot(strain1[num, :, 2], stress1[num, :, 2], linewidth=2, label='Point #%d' % num)
    #     ax2.plot(strain2[num, :, 2], stress2[num, :, 2], linewidth=2, label='Point #%d' % num)
    #
    # # ax1.set_xlabel(r'Loading step', fontdict=font_3)
    # ax1.set_ylabel(r'$\epsilon_{22}$', fontdict=font_3)
    # ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax2.set_xlabel(r'Loading step', fontdict=font_3)
    # ax2.set_ylabel(r'$\sigma_{22}$', fontdict=font_3)
    # ax2.ticklabel_format(axis='y', style='sci')
    # ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax1.xaxis.set_ticklabels([])
    # # ax1.legend(prop=font_5, loc='lower left', bbox_to_anchor=(0.1, 0.01), fancybox='sawtooth', shadow=True,
    # #            markerscale=0.6, ncol=2)
    # ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    # # ax2.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    # #            markerscale=0.6, ncol=2)
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig('./%s/strainStress22_%s.png' % (file_dir, mesh), dpi=200)
    # plt.close()
    # #
    # # # ------------------------------------------------------------------------------------------
    # # # 12 direction
    # plt.style.use('seaborn-paper')
    # fig = plt.figure(figsize=[8, 6])
    # ax1 = fig.add_subplot(211)
    # ax2 = fig.add_subplot(212)
    #
    # for num in indexOfPoint:
    #     # ax1.plot(range(len(strain[0])), strain[num, :, 1], linewidth=2, label='Point #%d' % num)
    #     # ax2.plot(range(len(stress[0])), stress[num, :, 1], linewidth=2, label='Point #%d' % num)
    #     ax1.plot(strain1[num, :, 1], stress1[num, :, 1], linewidth=2, label='Point #%d' % num)
    #     ax2.plot(strain2[num, :, 1], stress2[num, :, 1], linewidth=2, label='Point #%d' % num)
    #
    # # ax1.set_xlabel(r'Loading step', fontdict=font_3)
    # ax1.set_ylabel(r'$\epsilon_{12}$', fontdict=font_3)
    # ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax2.set_xlabel(r'Loading step', fontdict=font_3)
    # ax2.set_ylabel(r'$\sigma_{12}$', fontdict=font_3)
    # ax2.ticklabel_format(axis='y', style='sci')
    # ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    # ax1.xaxis.set_ticklabels([])
    # # ax1.legend(prop=font_5, loc='lower left', bbox_to_anchor=(0.1, 0.01), fancybox='sawtooth', shadow=True,
    # #            markerscale=0.6, ncol=2)
    # ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    # # ax2.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    # #            markerscale=0.6, ncol=2)
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig('./%s/strainStress12_%s.png' % (file_dir, num), dpi=200)
    # plt.close()
    #
    # # ------------------------------------------------------------------------------------------
    # # stress prediction
    # for num in indexOfPoint:
    #     markersize = 10
    #     plt.style.use('seaborn-paper')
    #     fig = plt.figure(figsize=[8, 6])
    #     ax1 = fig.add_subplot(311)
    #     ax2 = fig.add_subplot(312)
    #     ax3 = fig.add_subplot(313)
    #     strainAndStrainAbs = np.concatenate((strain[num], strain_abs[num]), axis=1)
    #     # stressPrdc, stiffness = net.get_stressAndStiffness(inputs=strainAndStrainAbs)
    #     strainAndStrainAbs_coupling = np.concatenate((strainCoupling[num], strain_absCoupling[num]), axis=1)
    #     # stressPrdc_coupling_ml, stiffness_coupling_ml = net.get_stressAndStiffness(inputs=strainAndStrainAbs_coupling)
    #     # index = range(0, len(stressPrdc_coupling_ml), 5)
    #     ax1.plot(range(len(stress[0])), stress[num, :, 0], linewidth=2, label='Point #%d FEM-DEM' % num)
    #     ax1.plot(range(len(stressCoupling[0])), stressCoupling[num, :, 0], linewidth=2, label='Point #%d FEM-ML' % num)
    #     # ax1.plot(range(len(stressPrdc)), stressPrdc[:, 0], linewidth=2, label='Point #%d DEM to ML' % num)
    #     # ax1.plot(index, stressPrdc_coupling_ml[:, 0][index], '^', markersize=markersize, label='Point #%d ML to ML' % num)
    #
    #     ax2.plot(range(len(stress[0])), stress[num, :, 1], linewidth=2, label='Point #%d FEM-DEM' % num)
    #     ax2.plot(range(len(stressCoupling[1])), stressCoupling[num, :, 1], linewidth=2, label='Point #%d FEM-ML' % num)
    #     # ax2.plot(range(len(stressPrdc)), stressPrdc[:, 1], linewidth=2, label='Point #%d ML' % num)
    #     # ax2.plot(index, stressPrdc_coupling_ml[:, 1][index], '^',  markersize=markersize, label='Point #%d ML to ML' % num)
    #
    #     ax3.plot(range(len(stress[0])), stress[num, :, 2], linewidth=2, label='Point #%d FEM-DEM' % num)
    #     ax3.plot(range(len(stressCoupling[2])), stressCoupling[num, :, 2], linewidth=2, label='Point #%d FEM-ML' % num)
    #     # ax3.plot(range(len(stressPrdc)), stressPrdc[:, 2], linewidth=2, label='Point #%d ML' % num)
    #     # ax3.plot(index, stressPrdc_coupling_ml[:, 2][index], '^',  markersize=markersize, label='Point #%d ML to ML' % num)
    #
    #     # ax1.set_xlabel(r'Loading step', fontdict=font_3)
    #     ax1.set_ylabel(r'$\sigma_{11}$', fontdict=font_3)
    #     ax1.ticklabel_format(axis='y', style='sci')
    #     ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     ax1.xaxis.set_ticklabels([])
    #     ax2.xaxis.set_ticklabels([])
    #     # ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #     #            markerscale=0.6, ncol=2)
    #     ax1.legend(fontsize=12,
    #                # prop=font_6,
    #                ncol=2, loc="best")
    #     # ax2.set_xlabel(r'Loading step', fontdict=font_3)
    #     ax2.set_ylabel(r'$\sigma_{12}$', fontdict=font_3)
    #     ax2.ticklabel_format(axis='y', style='sci')
    #     ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     # ax2.legend(prop=font_6, ncol=2, loc="upper left")
    #     ax3.set_xlabel(r'Loading step', fontdict=font_3)
    #     ax3.set_ylabel(r'$\sigma_{22}$', fontdict=font_3)
    #     ax3.ticklabel_format(axis='y', style='sci')
    #     ax3.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     ax3.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     # ax3.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #     #            markerscale=0.6, ncol=2)
    #     ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #     ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #     ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #
    #     plt.tight_layout()
    #     plt.savefig('./%s/stressPrediction_%d.png' % (file_dir, num), dpi=200)
    #     plt.close()
    #
    #
    # # ------------------------------------------------------------------------------------------
    # # strain comparation
    # for num in indexOfPoint:
    #     plt.style.use('seaborn-paper')
    #     fig = plt.figure(figsize=[8, 6])
    #     ax1 = fig.add_subplot(311)
    #     ax2 = fig.add_subplot(312)
    #     ax3 = fig.add_subplot(313)
    #     strainAndStrainAbs = np.concatenate((strain[num], strain_abs[num]), axis=1)
    #     # stressPrdc, stiffness = net.get_stressAndStiffness(inputs=strainAndStrainAbs)
    #     ax1.plot(strain[num, :, 0], linewidth=2, label='Point #%d FEM-DEM' % num)
    #     ax1.plot(strainCoupling[num, :, 0], linewidth=2, label='Point #%d FEM-ML' % num)
    #     ax2.plot(strain[num, :, 1], linewidth=2, label='Point #%d FEM-DEM' % num)
    #     ax2.plot(strainCoupling[num, :, 1], linewidth=2, label='Point #%d FEM-ML' % num)
    #     # ax2.plot(range(len(stressPrdc)), stressPrdc[:, 1], linewidth=2, label='Point #%d ML' % num)
    #     ax3.plot(strain[num, :, 2], linewidth=2, label='Point #%d FEM-DEM' % num)
    #     ax3.plot(strainCoupling[num, :, 2], linewidth=2, label='Point #%d FEM-ML' % num)
    #     # ax3.plot(range(len(stressPrdc)), stressPrdc[:, 2], linewidth=2, label='Point #%d ML' % num)
    #
    #     # ax1.set_xlabel(r'Loading step', fontdict=font_3)
    #     ax1.set_ylabel(r'$\epsilon_{11}$', fontdict=font_3)
    #     ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     ax1.xaxis.set_ticklabels([])
    #     ax2.xaxis.set_ticklabels([])
    #     # ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #     #            markerscale=0.6, ncol=2)
    #     ax1.legend(fontsize=12,
    #                # prop=font_6,
    #                ncol=2, loc="best")
    #     # ax2.set_xlabel(r'Loading step', fontdict=font_3)
    #     ax2.set_ylabel(r'$\epsilon_{12}$', fontdict=font_3)
    #     ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     # ax2.legend(prop=font_6, ncol=2, loc="upper left")
    #     ax3.set_xlabel(r'Loading step', fontdict=font_3)
    #     ax3.set_ylabel(r'$\epsilon_{22}$', fontdict=font_3)
    #     ax3.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     ax3.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    #     # ax3.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #     #            markerscale=0.6, ncol=2)
    #     ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #     ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #     ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    #
    #     plt.tight_layout()
    #     plt.savefig('./%s/strainComparation_%d.png' % (file_dir, num), dpi=200)
    #     plt.close()
    # return


# if __name__ == "__main__":
    # -----------------------------------------------------------------------------------------------
    # indexOfPoint = [100, 200, 300, 400, 500]
    # strain, strain_increment, stress, tangent, strain_abs, n, convergeList = getSeriasData(
    #     path='../../simu/ABS_DEM_8_16_biaxial', time=100)
    # strainCoupling, strain_incrementCoupling, stressCoupling, \
    # tangentCoupling, strain_absCoupling, nCoupling, convergeListCoupling = getSeriasDataCoupling(
    #     path='../../simu/ML_net_8_16_biaxial_ptModelH6_30_9_double_withouRetaining_5250', time=100)
    # plotCurve(strain, stress, strainCoupling, stressCoupling, indexOfPoint, mesh='fine')

    # -----------------------------------------------------------------------------------------------
    # strain, strain_increment, stress, tangent, strain_abs, n, convergeList = getSeriasData(
    #     path='/home/shguan/simu/DEM_retaining_20_10', time=100)
    # strainCoupling, strain_incrementCoupling, stressCoupling, \
    # tangentCoupling, strain_absCoupling, nCoupling, convergeListCoupling = getSeriasDataCoupling(
    #     path='/home/shguan/simu/mlRetaining/ML_retaining_net_OnlyInclude_Nodouble_20_10_1w', time=100)




    # strain_dem=pd.DataFrame(strain_dem)
    # strain_ml = pd.DataFrame(strain_ml)
    # strain_dem.to_csv('strain_dem.csv')


    # np.savetxt("strain_dem.csv", strain_dem, delimiter=',')
    # np.savetxt("strain_ml.csv", strain_ml, delimiter=',')
    #
    # np.savetxt("stress_dem.csv", stress_dem, delimiter=',')
    # np.savetxt("stress_ml.csv", stress_ml, delimiter=',')
    #
    # np.random.seed(2)
    # # x = np.random.permutation(range(len(strain_dem)))
    # x = np.arange(0, 800, 10)
    # indexOfPoint = list(x[78:79])



    # indexOfPoint = range(0, len(strain_dem), 400)
    #plotCurve(strain_dem, stress_dem, strain_ml, stress_ml, indexOfPoint, mesh='coarse') #strainCoupling, stressCoupling,
