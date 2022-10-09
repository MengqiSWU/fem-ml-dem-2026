import os

import gmshparser
import matplotlib.pyplot as plt
import numpy as np

from utilSelf.general import check_mkdir
from utils_plot import configurations, get_color_list

"""
    This script is used to plot the Gauss points positions.
"""
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
color_list = get_color_list()


def plot_configuration_from_gmsh(save_path=None, mesh_name='biaxial_0.05_548'):
    simu_name = mesh_name.split('_')[0]
    mesh_path = '../meshes/%s_msh/%s.msh' % (simu_name, mesh_name)
    fig_size = [3.2, 6] if simu_name == 'biaxial' else [6, 3]
    plt.figure(figsize=fig_size, dpi=200)
    mesh = gmshparser.parse(mesh_path)
    X, Y, T = gmshparser.helpers.get_triangles(mesh)

    xmin, xmax, ymin, ymax = np.min(X), np.max(X), np.min(Y), np.max(Y)
    # plt.plot([xmin, xmax, xmax, xmin, xmin], [ymin, ymin, ymax, ymax, ymin], c='k')
    plt.fill_between(x=[xmin, xmax], y1=[ymax, ymax], y2=[ymin, ymin], color=(0.086, 0.957, 1.0), zorder=-1)
    plt.triplot(X, Y, T, color='k', linewidth=0.5)
    # plt.legend(**legendDic)
    plt.axis('equal')
    plt.axis('off')
    # plt.title('order%d_integration%d' % (order, integration_order))
    if save_path:
        plt.tight_layout()
        check_mkdir(save_path)
        fname = os.path.join(save_path, '%s.svg' % mesh_name)
        plt.savefig(fname, dpi=200)
    return


if __name__ == "__main__":
    # mesh_name = 'biaxial_0.1_162'
    # mesh_name = 'biaxial_0.05_548'
    mesh_name = 'footing_615'
    # mesh_name = 'retaining_321'
    plot_configuration_from_gmsh(
        mesh_name=mesh_name,
        save_path='./gauss_points_position',
    )
