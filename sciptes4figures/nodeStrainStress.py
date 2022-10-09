from FEMxML.getSeriesDataConverged import getSeriasData, getSeriasDataCoupling
from FEMxML.netTorchLastDouble import modelRestore
from FEMxML.netTorchLastDouble import Net
import matplotlib.pyplot as plt
import numpy as np
import os
from sciptes4figures.utils_plot import configurations


# net restore
# ml_model_path = '../FEMxML/ptModels'
# net = modelRestore(savedPath=os.path.join(
#     ml_model_path, 'ptModelH6_30_9_double_withouRetaining/epoch_5250'), trainFlag=False)


def plotCurve(strain, stress, strainCoupling, stressCoupling, indexOfPoint, mesh, file_dir='retaining_gauss_implicit'):
    font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic=configurations()

    # ------------------------------------------------------------------------------------------
    # 11 direction
    plt.style.use('seaborn-paper')
    fig = plt.figure(figsize=[8, 6])
    # fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for num in indexOfPoint:
        ax1.plot(range(len(strain[0])), strain[num, :, 0], linewidth=2, label='Point #%d' % num)
        ax2.plot(range(len(stress[0])), stress[num, :, 0], linewidth=2, label='Point #%d' % num)

    ax1.set_ylabel(r'$\epsilon_{11}$', fontdict=font_3)
    ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax2.set_xlabel(r'Loading step', fontdict=font_3)
    ax2.set_ylabel(r'$\sigma_{11}$', fontdict=font_3)
    ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.xaxis.set_ticklabels([])
    # ax1.legend(prop=font_5, loc='lower left', bbox_to_anchor=(0.1, 0.01), fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
               markerscale=0.6, ncol=2)
    # ax2.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    plt.tight_layout()
    plt.savefig('./%s/strainStress11_%s.png' % (file_dir, mesh), dpi=200)
    plt.close()

    # ------------------------------------------------------------------------------------------
    # 22 direction
    plt.style.use('seaborn-paper')
    fig = plt.figure(figsize=[8, 6])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for num in indexOfPoint:
        ax1.plot(range(len(strain[0])), strain[num, :, 2], linewidth=2, label='Point #%d' % num)
        ax2.plot(range(len(stress[0])), stress[num, :, 2], linewidth=2, label='Point #%d' % num)
        # ax1.plot(strain[num, :, 0], strain[num, :, 2], linewidth=2, label='Point #%d' % num)
        # ax2.plot(stress[num, :, 0], stress[num, :, 2], linewidth=2, label='Point #%d' % num)

    # ax1.set_xlabel(r'Loading step', fontdict=font_3)
    ax1.set_ylabel(r'$\epsilon_{22}$', fontdict=font_3)
    ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax2.set_xlabel(r'Loading step', fontdict=font_3)
    ax2.set_ylabel(r'$\sigma_{22}$', fontdict=font_3)
    ax2.ticklabel_format(axis='y', style='sci')
    ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.xaxis.set_ticklabels([])
    # ax1.legend(prop=font_5, loc='lower left', bbox_to_anchor=(0.1, 0.01), fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
               markerscale=0.6, ncol=2)
    # ax2.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    plt.tight_layout()
    plt.savefig('./%s/strainStress22_%s.png' % (file_dir, mesh), dpi=200)
    plt.close()

    # ------------------------------------------------------------------------------------------
    # 12 direction
    plt.style.use('seaborn-paper')
    fig = plt.figure(figsize=[8, 6])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    for num in indexOfPoint:
        ax1.plot(range(len(strain[0])), strain[num, :, 1], linewidth=2, label='Point #%d' % num)
        ax2.plot(range(len(stress[0])), stress[num, :, 1], linewidth=2, label='Point #%d' % num)
        # ax1.plot(strain[num, :, 0], strain[num, :, 2], linewidth=2, label='Point #%d' % num)
        # ax2.plot(stress[num, :, 0], stress[num, :, 2], linewidth=2, label='Point #%d' % num)

    # ax1.set_xlabel(r'Loading step', fontdict=font_3)
    ax1.set_ylabel(r'$\epsilon_{12}$', fontdict=font_3)
    ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax2.set_xlabel(r'Loading step', fontdict=font_3)
    ax2.set_ylabel(r'$\sigma_{12}$', fontdict=font_3)
    ax2.ticklabel_format(axis='y', style='sci')
    ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.xaxis.set_ticklabels([])
    # ax1.legend(prop=font_5, loc='lower left', bbox_to_anchor=(0.1, 0.01), fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
               markerscale=0.6, ncol=2)
    # ax2.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    plt.tight_layout()
    plt.savefig('./%s/strainStress12_%s.png' % (file_dir, num), dpi=200)
    plt.close()

    # ------------------------------------------------------------------------------------------
    # stress prediction
    for num in indexOfPoint:
        markersize = 10
        plt.style.use('seaborn-paper')
        fig = plt.figure(figsize=[8, 6])
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        strainAndStrainAbs = np.concatenate((strain[num], strain_abs[num]), axis=1)
        # stressPrdc, stiffness = net.get_stressAndStiffness(inputs=strainAndStrainAbs)
        strainAndStrainAbs_coupling = np.concatenate((strainCoupling[num], strain_absCoupling[num]), axis=1)
        # stressPrdc_coupling_ml, stiffness_coupling_ml = net.get_stressAndStiffness(inputs=strainAndStrainAbs_coupling)
        # index = range(0, len(stressPrdc_coupling_ml), 5)
        ax1.plot(range(len(stress[0])), stress[num, :, 0], linewidth=2, label='Point #%d FEM-DEM' % num)
        ax1.plot(range(len(stressCoupling[0])), stressCoupling[num, :, 0], linewidth=2, label='Point #%d FEM-ML' % num)
        # ax1.plot(range(len(stressPrdc)), stressPrdc[:, 0], linewidth=2, label='Point #%d DEM to ML' % num)
        # ax1.plot(index, stressPrdc_coupling_ml[:, 0][index], '^', markersize=markersize, label='Point #%d ML to ML' % num)

        ax2.plot(range(len(stress[0])), stress[num, :, 1], linewidth=2, label='Point #%d FEM-DEM' % num)
        ax2.plot(range(len(stressCoupling[1])), stressCoupling[num, :, 1], linewidth=2, label='Point #%d FEM-ML' % num)
        # ax2.plot(range(len(stressPrdc)), stressPrdc[:, 1], linewidth=2, label='Point #%d ML' % num)
        # ax2.plot(index, stressPrdc_coupling_ml[:, 1][index], '^',  markersize=markersize, label='Point #%d ML to ML' % num)

        ax3.plot(range(len(stress[0])), stress[num, :, 2], linewidth=2, label='Point #%d FEM-DEM' % num)
        ax3.plot(range(len(stressCoupling[2])), stressCoupling[num, :, 2], linewidth=2, label='Point #%d FEM-ML' % num)
        # ax3.plot(range(len(stressPrdc)), stressPrdc[:, 2], linewidth=2, label='Point #%d ML' % num)
        # ax3.plot(index, stressPrdc_coupling_ml[:, 2][index], '^',  markersize=markersize, label='Point #%d ML to ML' % num)

        # ax1.set_xlabel(r'Loading step', fontdict=font_3)
        ax1.set_ylabel(r'$\sigma_{11}$', fontdict=font_3)
        ax1.ticklabel_format(axis='y', style='sci')
        ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        # ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
        #            markerscale=0.6, ncol=2)
        ax1.legend(fontsize=12,
                   # prop=font_6,
                   ncol=2, loc="best")
        # ax2.set_xlabel(r'Loading step', fontdict=font_3)
        ax2.set_ylabel(r'$\sigma_{12}$', fontdict=font_3)
        ax2.ticklabel_format(axis='y', style='sci')
        ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        # ax2.legend(prop=font_6, ncol=2, loc="upper left")
        ax3.set_xlabel(r'Loading step', fontdict=font_3)
        ax3.set_ylabel(r'$\sigma_{22}$', fontdict=font_3)
        ax3.ticklabel_format(axis='y', style='sci')
        ax3.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        ax3.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        # ax3.legend(prop=font_5, fancybox='sawtooth', shadow=True,
        #            markerscale=0.6, ncol=2)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig('./%s/stressPrediction_%d.png' % (file_dir, num), dpi=200)
        plt.close()


    # ------------------------------------------------------------------------------------------
    # strain comparation
    for num in indexOfPoint:
        plt.style.use('seaborn-paper')
        fig = plt.figure(figsize=[8, 6])
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        strainAndStrainAbs = np.concatenate((strain[num], strain_abs[num]), axis=1)
        # stressPrdc, stiffness = net.get_stressAndStiffness(inputs=strainAndStrainAbs)
        ax1.plot(strain[num, :, 0], linewidth=2, label='Point #%d FEM-DEM' % num)
        ax1.plot(strainCoupling[num, :, 0], linewidth=2, label='Point #%d FEM-ML' % num)
        ax2.plot(strain[num, :, 1], linewidth=2, label='Point #%d FEM-DEM' % num)
        ax2.plot(strainCoupling[num, :, 1], linewidth=2, label='Point #%d FEM-ML' % num)
        # ax2.plot(range(len(stressPrdc)), stressPrdc[:, 1], linewidth=2, label='Point #%d ML' % num)
        ax3.plot(strain[num, :, 2], linewidth=2, label='Point #%d FEM-DEM' % num)
        ax3.plot(strainCoupling[num, :, 2], linewidth=2, label='Point #%d FEM-ML' % num)
        # ax3.plot(range(len(stressPrdc)), stressPrdc[:, 2], linewidth=2, label='Point #%d ML' % num)

        # ax1.set_xlabel(r'Loading step', fontdict=font_3)
        ax1.set_ylabel(r'$\epsilon_{11}$', fontdict=font_3)
        ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        ax1.xaxis.set_ticklabels([])
        ax2.xaxis.set_ticklabels([])
        # ax1.legend(prop=font_5, fancybox='sawtooth', shadow=True,
        #            markerscale=0.6, ncol=2)
        ax1.legend(fontsize=12,
                   # prop=font_6,
                   ncol=2, loc="best")
        # ax2.set_xlabel(r'Loading step', fontdict=font_3)
        ax2.set_ylabel(r'$\epsilon_{12}$', fontdict=font_3)
        ax2.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        ax2.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        # ax2.legend(prop=font_6, ncol=2, loc="upper left")
        ax3.set_xlabel(r'Loading step', fontdict=font_3)
        ax3.set_ylabel(r'$\epsilon_{22}$', fontdict=font_3)
        ax3.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        ax3.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
        # ax3.legend(prop=font_5, fancybox='sawtooth', shadow=True,
        #            markerscale=0.6, ncol=2)
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig('./%s/strainComparation_%d.png' % (file_dir, num), dpi=200)
        plt.close()
    return


if __name__ == "__main__":
    # -----------------------------------------------------------------------------------------------
    # indexOfPoint = [100, 200, 300, 400, 500]
    # strain, strain_increment, stress, tangent, strain_abs, n, convergeList = getSeriasData(
    #     path='../../simu/ABS_DEM_8_16_biaxial', time=100)
    # strainCoupling, strain_incrementCoupling, stressCoupling, \
    # tangentCoupling, strain_absCoupling, nCoupling, convergeListCoupling = getSeriasDataCoupling(
    #     path='../../simu/ML_net_8_16_biaxial_ptModelH6_30_9_double_withouRetaining_5250', time=100)
    # plotCurve(strain, stress, strainCoupling, stressCoupling, indexOfPoint, mesh='fine')

    # -----------------------------------------------------------------------------------------------
    strain, strain_increment, stress, tangent, strain_abs, n, convergeList = getSeriasData(
        path='/home/shguan/simu/DEM_retaining_20_10', time=100)
    strainCoupling, strain_incrementCoupling, stressCoupling, \
    tangentCoupling, strain_absCoupling, nCoupling, convergeListCoupling = getSeriasDataCoupling(
        path='/home/shguan/simu/mlRetaining/ML_retaining_net_OnlyInclude_Nodouble_20_10_1w', time=100)

    indexOfPoint = range(0, len(strain), 20)
    plotCurve(strain, stress, strainCoupling, stressCoupling, indexOfPoint, mesh='coarse')

