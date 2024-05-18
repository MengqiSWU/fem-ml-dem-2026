import matplotlib.pyplot as plt
from utils_plot import configurations, get_color_list
import os
from esys.escript import whereZero, FunctionOnBoundary, interpolate, kronecker, Vector, Solution, matrix_mult, sup, \
    integrate, symmetric, trace, sqrt, inner, ReducedSolution, Data, whereNonPositive, wherePositive, inf, Tensor, \
    Function,whereNegative
from esys.finley import ReadGmsh, Rectangle
import numpy as np
# import gmshparser
from utilSelf.general import check_mkdir


"""
    This script is used to plot the Gauss points positions.
"""
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
color_list = get_color_list()


mesh_number = 5
nx, ny = mesh_number, mesh_number * 2  # sample discretization, 8 by 16 quadrilateral elements
order = 1
mydomain = Rectangle(l0=0.5, l1=1.0, n0=nx, n1=ny,
                     order=1, integrationOrder=2)

def plot_err_gauss_points(index_err_gauss_list,
                          highlight_list=None, save_path=None, plot_mesh_flag=False):

    mesh_name = 'biaxial_0.08'

    test_name = mesh_name.split('_')[0]
    mesh_path = '../meshes/%s_msh/%s.msh' % (test_name, mesh_name)

    domain = Rectangle(l0=0.5, l1=1.0, n0=nx, n1=ny,
                         order=1, integrationOrder=2)
    # plt.figure(figsize=[3.2, 6], dpi=200)
    plt.figure(figsize=[6, 3.2], dpi=200)
    plt.show()
    x = np.array(domain.getX().toListOfTuples())
    gx = np.array(Function(domain).getX().toListOfTuples())
    # plt.scatter(x=x[:, 0], y=x[:, 1], c=color_list[0], s=10, label='Node')
    for i, index_err_gauss in enumerate(index_err_gauss_list):
        for gauss_num in index_err_gauss:
            if highlight_list is not None and highlight_list[i] is not None and gauss_num in highlight_list[i]:
                plt.scatter(
                    x=gx[gauss_num, 0], y=gx[gauss_num, 1],
                    c='r', s=50)
                # plt.annotate(text='%d' % gauss_num, xy=[gx[gauss_num, 0], gx[gauss_num, 1]], color='r',**font_4)
            else:
                plt.scatter(
                    x=gx[gauss_num, 0], y=gx[gauss_num, 1],
                    c='k', s=20)
                # plt.annotate(text='%d' % gauss_num, xy=[gx[gauss_num, 0], gx[gauss_num, 1]], **font_5)

    xmin, xmax, ymin, ymax = np.min(x[:, 0]), np.max(x[:, 0]), np.min(x[:, 1]), np.max(x[:, 1])
    # plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], c='k')
    plt.fill_between(x=[xmin, xmax], y1=[ymax, ymax], y2=[ymin, ymin], color=(0.086, 0.957, 1.0), zorder=-1)
    if plot_mesh_flag:
        mesh = gmshparser.parse(mesh_path)
        X, Y, T = gmshparser.helpers.get_triangles(mesh)
        plt.triplot(X, Y, T, color='k', linewidth=0.5)
    # plt.legend(**legendDic)
    plt.axis('equal')
    plt.axis('off')
    # plt.title('order%d_integration%d' % (order, integration_order))
    plt.tight_layout()
    plt.show()
    if save_path:
        check_mkdir(save_path)
        fname = os.path.join(save_path, '%s_model_gauss_points.svg' % test_name)
        plt.savefig(fname, dpi=200)
    else:
        plt.show()
        plt.close()
    return


if __name__ == "__main__":
    plot_err_gauss_points(

        index_err_gauss_list=[
            list(range(0, 200, 1)),
        ],
        # highlight_list=[list(range(0,200,7))],
        highlight_list=[[66,68],],
        save_path='./gauss_points_position'
    )

