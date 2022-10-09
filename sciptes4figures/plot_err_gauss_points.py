import matplotlib.pyplot as plt
from utils_plot import configurations, get_color_list
import os
from esys.escript import whereZero, FunctionOnBoundary, interpolate, kronecker, Vector, Solution, matrix_mult, sup, \
    integrate, symmetric, trace, sqrt, inner, ReducedSolution, Data, whereNonPositive, wherePositive, inf, Tensor, \
    Function,whereNegative
from esys.escript.pdetools import Projector
from esys.finley import ReadGmsh, Rectangle
from esys.weipa import saveVTK
import numpy as np
from plot_configuration_from_msh import plot_configuration_from_gmsh


"""
    This script is used to plot the Gauss points with higher prediction error 
        collected during the hybrid iteration in the footing simulation.
"""

font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
color_list = get_color_list()


def plot_err_gauss_points(order, index_err_gauss_list, integration_order=1, save_path=None, color_index=None):
    # mesh_name = 'biaxial_0.05_548'
    mesh_name = 'footing_615'
    # mesh_name = 'biaxial_0.1_162'

    test_name = mesh_name.split('_')[0]
    mesh_path = '../meshes/%s_msh/%s.msh' % (test_name, mesh_name)
    domain = ReadGmsh(
        mesh_path, numDim=2,
        order=order, integrationOrder=integration_order)
    plot_configuration_from_gmsh(mesh_name=mesh_name)
    x = np.array(domain.getX().toListOfTuples())
    gx = np.array(Function(domain).getX().toListOfTuples())
    # plt.scatter(x=x[:, 0], y=x[:, 1], c=color_list[0], s=10, label='Node')
    for i, index_err_gauss in enumerate(index_err_gauss_list):
        if color_index is not None:
            color = color_list[color_index]
            label = 'Iteration %d' % color_index
        else:
            color = color_list[i]
            label = 'Iteration %d' % i
        plt.scatter(
            x=gx[index_err_gauss, 0], y=gx[index_err_gauss, 1], c=color, s=30, label=label)
    xmin, xmax, ymin, ymax = np.min(x[:, 0]), np.max(x[:, 0]), np.min(x[:, 1]), np.max(x[:, 1])
    plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], c='k')
    plt.legend(**legendDic)
    plt.axis('equal')
    # plt.title('order%d_integration%d' % (order, integration_order))
    plt.tight_layout()
    if save_path:
        fname = os.path.join(save_path, '%s_model_%d.svg' % (test_name, max(color_index, 0)))
        plt.savefig(fname, dpi=200)
    else:
        plt.show()
        plt.close()
    return


def get_numg_index(fname):
    numg_index_all = []
    files = os.listdir(fname)
    index_file = np.array([int(i.split('_')[1]) for i in files])
    index_argsort = np.argsort(index_file)
    for i in index_argsort:
        ffname = os.path.join(fname, files[i])
        f = open(ffname, mode='r')
        datas = f.readlines()
        f.close()
        n = len(datas)
        line_num = 0
        while line_num < n:
            if "numg_index" in datas[line_num]:
                err_point_num = int(datas[line_num].split(' ')[1])
                num_index_temp = np.zeros(err_point_num, dtype=int)
                for j in range(err_point_num):
                    num_index_temp[j] = int(datas[line_num+j+1])
                numg_index_all.append(num_index_temp)
                break
            else:
                line_num += 1
    return numg_index_all


if __name__ == "__main__":
    file_list = [  # footing
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep4.0e-04_b0.30/added_points',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_1_timestep4.0e-04_b0.30/added_points',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_2_timestep4.0e-04_b0.30/added_points',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep4.0e-04_b0.30/added_points',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_4_timestep4.0e-04_b0.30/added_points',
    '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_5_timestep4.0e-04_b0.30/added_points',
    ]
    # file_list = [  # biaxial
    #     # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_biaxial_0/added_points',
    #     '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_biaxial_1/added_points',
    #     '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_biaxial_2/added_points',
    #     '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_biaxial_3/added_points',
    #     '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_biaxial_4/added_points',
    #     '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg128_biaxial_162_large_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_biaxial_4/added_points',
    # ]
    for i in range(len(file_list)):
        plot_err_gauss_points(
            order=1, index_err_gauss_list=[get_numg_index(fname=file)[-1] for file in file_list[i:i+1]],
            color_index=i, save_path='./gauss_points_position')

