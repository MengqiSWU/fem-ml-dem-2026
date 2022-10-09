import matplotlib.pyplot as plt
import numpy as np
from utils_plot import configurations
from esys.escript import Function, FunctionOnBoundary
from esys.finley import ReadGmsh

font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()


class basic_plot_active_learning:
    def __init__(self, mshname, order=1, int_order=2, ):
        self.domain = ReadGmsh(
            '../meshes/footing_msh/%s.msh' % mshname, numDim=2,
            order=order, integrationOrder=int_order)
        self.x = np.array(self.domain.getX().toListOfTuples())
        self.bx = np.array(FunctionOnBoundary(self.domain).getX().toListOfTuples())
        self.x_gauss = np.array(Function(self.domain).getX().toListOfTuples())
        self.boundary = self.get_bounary()

    def plot_plot(self, ):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # plot bound
        ax.plot([self.boundary[0, 0], self.boundary[0, 1], self.boundary[0, 1], self.boundary[0, 0], self.boundary[0, 0]],
                [self.boundary[1, 0], self.boundary[1, 0], self.boundary[1, 1], self.boundary[1, 1], self.boundary[1, 0]], 'k')
        # plot the gauss points
        numg_step = self.get_active_numg_index()
        for i in range(len(numg_step)):
            if len(numg_step[i]) == 0:
                break
            ax.scatter(self.x_gauss[numg_step[i]][:, 0], self.x_gauss[numg_step[i]][:, 1], label="%d" % i, s=10)
        plt.legend(ncol=4, **legendDic)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        plt.close()

    def get_active_numg_index(
            self,
            fname = '/home/tongming/fem-ml-dem/FEMxML/footing_ml/active_footing_3618_4/X_epsANDabsxy_Y_sig_numNN3_dd5/sample_index.txt'):
        f = open(fname, mode='r')
        datas = f.readlines()
        f.close()
        line_num = 0
        numg_step_selected =[]
        while line_num < len(datas):
            if 'added index' in datas[line_num]:
                temp_line_pieces = datas[line_num].split('[')[1].split(']')[0].split(' ')
                temp_index_list = []
                for i in temp_line_pieces:
                    try:
                        temp_int = int(i)
                        temp_index_list.append(temp_int)
                    except:
                        pass
                numg_step_selected.append(temp_index_list)
            line_num += 1
        return np.array(numg_step_selected)

    def get_bounary(self):
        bound = np.array([np.min(self.x, axis=0), np.max(self.x, axis=0)]).transpose()
        return bound


if __name__ == "__main__":
    plotactive_obj = basic_plot_active_learning(mshname='footing1206')
    plotactive_obj.plot_plot()
    print()