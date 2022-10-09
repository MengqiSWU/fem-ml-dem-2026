import os
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class GuanssianRandomPath:
    """
        Used for random loading path generation via Gaussian Process method
    """
    def __init__(self, curlDegree, amplitudeValue,
                 showFlag=False, generatingNum=0, maxEpsilonLimitation=1e5):
        self.seed = 10001
        # np.random.seed(self.seed)

        self.showFlag = showFlag
        self.generatingNum = generatingNum
        self.maxEpsilonLimitation = maxEpsilonLimitation
        self.numberOfPoints = 100
        self.numberOfFuncutions = 1
        self.meanValue = -1e5

        self.curlDegree = curlDegree  # 1~5
        self.amplitudeValue = (0.25*np.abs(self.meanValue))**2.  # generally 0.25
        # self.amplitude = 1
        self.amplitude = np.linspace(0, self.amplitudeValue, int(self.numberOfPoints))
        self.x = self.curlDegree*np.linspace(0, 1., self.numberOfPoints)[:, np.newaxis]
        self.cov = self.CovarienceMatrix(self.x, self.x)*self.amplitude

        self.y = np.random.multivariate_normal(mean=np.ones(self.numberOfPoints)*self.meanValue,
                                               cov=self.cov,
                                               size=self.numberOfFuncutions)
        self.yList = []

        if self.showFlag:
            self.plotPaths()
            self.plotCovarianceMatrix(kernel=self.cov, curl=self.curlDegree)

        if self.generatingNum > 0:
            self.generation()

    def generation(self):
        print()
        print('='*80)
        print('\t Loading path generation ...')
        i = 0
        numSample = 0
        while numSample < self.generatingNum:
            print('\t\tPath random %d seed %d' % (numSample, i))
            self.seed = i
            np.random.seed(self.seed)
            curlDegree = np.random.choice(range(1, 6))
            self.x = curlDegree * np.linspace(0, 1., self.numberOfPoints)[:, np.newaxis]
            self.cov = self.CovarienceMatrix(self.x, self.x) * self.amplitude
            self.y = np.random.multivariate_normal(mean=np.ones(self.numberOfPoints)*self.meanValue,
                                               cov=self.cov,
                                               size=self.numberOfFuncutions)
            maxEpsilon = np.max(np.abs(self.y-self.meanValue))
            if maxEpsilon > self.maxEpsilonLimitation/2.:
                i += 1
                continue
            else:
                self.yList.append(self.y.reshape(-1))
                self.plotPaths(numSample=numSample, curlDegree=curlDegree)
                numSample += 1
                i += 1
        self.writeDownPaths(numSample)

    def CovarienceMatrix(self, x, y):
        """
            Use the kernel fucntion: $\kappa(x_i, x_j)=\mathrm{exp}(-\sum_{k=1}^{m}\theta_k(x_i^k-x_j^k)^2))$
                where the dimensional number is 1 in this project.

            Reference:
                [1] https://blog.dominodatalab.com/fitting-gaussian-process-models-python
        :param x:
        :param y:
        :return:
        """
        mesh = np.meshgrid(x, y)
        kernel = np.exp(-(mesh[0]-mesh[1])**2)
        return kernel

    def plotCovarianceMatrix(self, kernel, curl, path='../sciptes4figures/gaussian'):
        numberOfticksInFigure = 7
        interval = int(len(kernel)/numberOfticksInFigure)
        ax = sns.heatmap(kernel, xticklabels=interval, yticklabels=interval, cmap="YlGnBu")
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        cax = plt.gcf().axes[-1]
        cax.tick_params(labelsize=15)
        plt.tight_layout()
        fName = os.path.join(path, 'CovariabceHeatMap_curl%d.png' % curl)
        plt.savefig(fName, dpi=200)
        plt.close()

    def plotPaths(self, path='gaussianRandom', numSample=-1, curlDegree=-1):
        # Plot the sampled functions
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=200)
        totalPointsOnFigure = 50
        interval = int(len(self.y[0])/50)
        for i in range(self.numberOfFuncutions):
            plt.plot(list(range(1, len(self.y[0]) + 1))[::interval], list(self.y[i]/1e3)[::interval],
                     linestyle='-', marker='o', markersize=4, label='Confining pressure')
        plt.xlabel('Loading step', fontsize=15)
        plt.ylabel('kPa', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim([0, len(self.y[0])])
        plt.tight_layout()
        plt.legend(fontsize=15)
        figName = 'ConfiningPressureGP_curl%d_seed%d.png' % (self.curlDegree, self.seed)
        if numSample >= 0:
            figName = 'ConfiningPressureGP_curl%d_seed%d_%d.png' % (self.curlDegree, self.seed, numSample)
        if curlDegree > 0:
            figName = 'ConfiningPressureGP_curl%d_seed%d_%d.png' % (curlDegree, self.seed, numSample)
        plt.savefig(os.path.join(path, figName), dpi=200)
        print('Figure saved as %s' % os.path.join(path, figName))
        plt.close()

    def writeDownPaths(self, numSample):
        filePath = './gaussianRandom/path_%d.csv' % numSample
        dataFrame = pd.DataFrame(data=np.array(self.yList).T)
        dataFrame.to_csv(filePath, header=False, index=False)


if __name__ == "__main__":
    debugFlag = False
    if debugFlag:
        for i in range(1, 6):
            gaussian = GuanssianRandomPath(curlDegree=i,
                                           amplitudeValue=(0.25*1e5)**2,
                                           generatingNum=1,
                                           showFlag=True, )  # generally 1~5, 0.25
    else:
        gaussian = GuanssianRandomPath(curlDegree=2,
                                       amplitudeValue=(0.25*1e5)**2,
                                       generatingNum=30,
                                       showFlag=False)  # generally 1~5, 0.25
