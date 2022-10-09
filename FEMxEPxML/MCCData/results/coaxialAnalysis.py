import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class coaxialAnalysis:
    def __init__(self):
        '''
        0        1         2           3           4            5            6          7          8            9
        number	case	sigma_xx	sigma_yy	sigma_xy	epsilon_xx	epsilon_yy	epsilon_xy	vonMises	epsPlastic
            10          11               12              13             14          15
        hardening	epsilonP__xx	epsilonP__yy	epsilonP__xy	yieldValue	iteration
        '''
        dataArray = pd.read_csv('./DEM_vonMises2D.csv').values
        self.sigma, self.epsilon = dataArray[:1001, 2:5], dataArray[:1001, 5:8]
        self.epsilon[:, 2] = .5*self.epsilon[:, 2]
        self.sigmaTheta = np.array([self.getPrincipleComponents(self.sigma[i]) for i in range(0, len(self.sigma), 1)])
        self.epsilonTheta = np.array([self.getPrincipleComponents(self.epsilon[i]) for i in range(0, len(self.epsilon), 1)])
        self.epsilonPlastic = dataArray[:1000, 9]

    def getPrincipleComponents(self, voigtVector):
        tensor = np.array([[voigtVector[0], voigtVector[2]],
                           [voigtVector[2], voigtVector[1]]])
        value = np.linalg.eig(tensor)[1]
        theta = np.arcsin(value[0, 0])/np.pi*180
        theta = theta if theta > 0 else theta+90
        theta = theta if theta < 90 else theta-90
        # theta = np.arctan(value[0]/value[1] if value[1] != 0 else 0.)
        return theta

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.sigmaTheta[2:], label=r'$\theta_{\sigma}$')
        ax.plot(self.epsilonTheta[2:], label=r'$\theta_{\epsilon}$')
        plt.legend(fontsize=15, loc='upper left')
        plt.ylabel('Angle')
        axTwin = ax.twinx()
        axTwin.plot(self.epsilonPlastic[2:], 'r.', label=r'$\epsilon_{s}^{plastic}$')
        plt.legend(fontsize=15, loc='lower right')
        plt.xlabel('Load step')
        plt.ylabel(r'$\epsilon_{s}^{plastic}$')
        plt.tight_layout()
        plt.savefig('./coaxialAnalysis.png')
        plt.close()


if __name__ == '__main__':
    cc = coaxialAnalysis()
    cc.plot()
