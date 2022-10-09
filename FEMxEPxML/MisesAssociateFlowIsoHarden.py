import os
import numpy as np
import matplotlib.pyplot as plt
from FEMxEPxML.MCCUtil import plotSubFigures, loadingPathReader
from sciptes4figures.plotConfiguration2D import plotConfiguration2D
from FEMxEPxML.misesTraining import Restore, Net, DenseResNet

"""
        The constitutive model is under the elastoplastic 
        framework of Mises yield function, associated-flow 
        rule, and iso-hardening.

        Gaussian process engaged for random loading path 
        generation.

        This script is used to generate datasets for phys-
        ics-constrained constitutive network training.

        Author: Shaoheng Guan
        Email:  shaohengguan@gmail.com

        Reference:
            [1] Bonatti C, Mohr D (2021) One for all: Universal material model based on minimal 
                state-space neural networks. Sci Adv 7:1–9. https://doi.org/10.1126/sciadv.abf3658


"""


class MisesAssociateFlowIsoHarden:
    def __init__(self, loadMode='axial', mode='math', nonlinearHardening=True, verboseFlag=False):
        # ---------------------------------------------------
        # network initialization if needed
        self.mode = mode
        if 'net' in self.mode or 'semi' in self.mode:
            try:
                self.vonMisesNet = Restore(
                    savedPath=os.path.join('misesModel', 'mdmddmd_Residual_Fourier_data_minmax'),
                )
                self.vonMisesNet.model.minmaxFlag = True
                self.hardeningNet = Restore(
                    savedPath=os.path.join('HardeningModel', 'mmmmd_Residual_Fourier_data_normalized'),
                )
            except:
                self.vonMisesNet = Restore(
                    savedPath=os.path.join('./FEMxEPxML/misesModel', 'mdmddmd_Residual_Fourier_data_minmax'),
                )
                self.vonMisesNet.model.minmaxFlag = True
                self.hardeningNet = Restore(
                    savedPath=os.path.join('./FEMxEPxML/HardeningModel', 'mmmmd_Residual_Fourier_data_normalized'),
                )
        # ---------------------------------------------------
        # material parameters
        self.nonlinearHardening = nonlinearHardening
        self.youngsModulus = 0.5e9  # 200e9
        self.poisson = 0.3
        # self.A = 500e6
        self.A = self.youngsModulus * 0.2
        self.n = 0.2
        self.epsilon0 = 0.05
        self.yieldStress = 0.5e6  # 200e6
        self.hardening = self.getHardening(epsPlastic=0.)
        self.D = self.tangentAssemble(
            lam=self.youngsModulus * self.poisson / (1 + self.poisson) / (1 - 2 * self.poisson),
            G=self.youngsModulus / 2 / (1 + self.poisson))
        self.M = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.5]])

        # ---------------------------------------------------
        # state variables [stress and strain vector in voigt notion]
        self.sig = np.zeros(3)
        self.vonMises = self.getVonMises(self.sig)
        self.eps = np.zeros(3)
        self.epsPlastic = 0.
        self.epsPlasticVector = np.zeros(3)
        self.sigTrial = np.zeros(3)
        self.lastYield = -1
        self.yieldValue = self.yieldFunction(self.sig, hardening=self.getHardening(self.epsPlastic))
        self.loadHistoryList = [np.array(list(self.sig) + list(self.eps) +
                                         [self.vonMises, self.epsPlastic, self.hardening] +
                                         list(self.epsPlasticVector) + [self.yieldValue, 0])]

        # ---------------------------------------------------
        # load configuration
        self.verboseFlag = verboseFlag
        self.loadMode = loadMode  # 'axial' or 'random'
        if self.loadMode == 'random':
            self.epsAxialObject = 0.004  # random loading
        else:
            self.epsAxialObject = 0.01
        self.iterationNum = int(1e2)
        self.depsAxial = self.epsAxialObject / self.iterationNum

        # ---------------------------------------------------
        # Tolerance
        self.yieldToleranceNegtive = -1000.
        self.yieldTolerancePositive = 1000.
        if 'math' in self.mode:
            self.yieldToleranceNegtive = -10.
            self.yieldTolerancePositive = 10.

    def forward(self, st=None, path=None, sampleIndex=None):
        mmax = np.max(np.abs(path))
        if mmax > 0.5:
            path = path / mmax * 0.5
        if self.loadMode == 'random':
            self.iterationNum = len(path)
        for i in range(self.iterationNum):
            print()
            print('Load step: %d' % i)
            if self.loadMode == 'random':  # load under the gaussian random loading path
                if i == 0:
                    deps = path[0]
                else:
                    # deps = 10*(path[i] - path[i - 1]) * self.epsAxialObject / np.max(np.abs(path))
                    deps = path[i] - path[i - 1]
            elif st is not None:
                deps = st
            else:  # load under the conventional triaxial compression test
                deps = self.getAxialDeps()
                if 0.3 * self.iterationNum < i < 0.65 * self.iterationNum:
                    deps = - deps
            iteration, sigTrial, materialMatrix, epsPlastic, deps_plasticVector, yieldValue = self.solver(deps)
            # yieldValue = self.yieldFunction(self.sig)
            self.updateState(
                sig=sigTrial, deps=deps, yieldValue=yieldValue,
                epsPlastic=epsPlastic, deps_plasticVector=deps_plasticVector)
            self.loadHistoryList.append(np.array(
                list(self.sig) + list(self.eps) +
                [self.getVonMises(self.sig), self.epsPlastic, self.getHardening(self.epsPlastic)] +
                list(self.epsPlasticVector) + [self.yieldValue, iteration]))
        self.plotMask(sampleIndex)

    def solver(self, deps):
        sigTrial = self.sig + self.D @ deps.T
        # vonMises = self.getVonMises(sigTrial)
        # hardening = self.getHardening(self.epsPlastic)
        yieldValue = self.yieldFunction(sigTrial, hardening=self.getHardening(self.epsPlastic))
        iteration = 0
        epsPlastic = self.epsPlastic
        deps_plasticVector = np.zeros(3)
        elasticFlag = True
        r_mid = 1.
        # -----------------------------------
        # elastic
        if yieldValue <= 0:
            sig = sigTrial
            deps_plastic = np.zeros(3)
            materialMatrix = self.D

        # -----------------------------------
        # plastic and last step is elastic
        elif self.yieldValue < self.yieldToleranceNegtive:
            if self.verboseFlag:
                print()
                print('Bisection!!  yieldValue %.3f LastYieldValue %.3f' % (yieldValue, self.yieldValue))
                print()
            r_mid, yield_mid = self.transiformationSplit(deps)  # searching for the transit point
            sig = self.sig + np.dot(self.D, r_mid * deps)
            # return iteration, sig, self.D, epsPlastic, deps_plasticVector, yieldValue, elasticFlag, r_mid
            # vonMises = self.getVonMises(self.sig)
            # eps += r_mid * deps
            # self.loadHistoryList.append(np.array(
            #     list(self.sig) + list(self.eps) +
            #     [vonMises, self.epsPlastic, hardening] +
            #     list(self.epsPlasticVector) + [self.yieldValue, iteration]))
            depsLeft = (1 - r_mid) * deps
            # lastYield = yield_mid
            # update the trial stress
            # sigTrial = self.sig
            # vonMises = self.getVonMises(sigTrial)
            # yieldValue = self.yieldFunction(sig=sigTrial)
            iteration, sigTrial, materialMatrix, deps_plasticVector, epsPlastic, yieldValue = \
                self.plasticReturnMapping_new(deps=depsLeft, sigAfterBisection=sig)
            # materialMatrix = r_mid * self.D + (1 - r_mid) * materialMatrix
            # materialMatrix = self.getAveragedMatrix(materialMatrix, r_mid)
            materialMatrix = self.D

        # -----------------------------------
        # last step is plastic
        else:
            iteration, sigTrial, materialMatrix, deps_plasticVector, epsPlastic, yieldValue = \
                self.plasticReturnMapping_new(deps=deps)

        return iteration, sigTrial, materialMatrix, epsPlastic, deps_plasticVector, yieldValue

    def getAveragedMatrix(self, materialMatrix, r_mid):
        temp = np.zeros(shape=(3, 3))
        for ix, iy in [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]]:
            temp[ix, iy] = 1. / (r_mid / self.D[ix, iy] + (1.0 - r_mid) / materialMatrix[ix, iy])
        return temp

    def updateState(self, sig, deps, yieldValue, epsPlastic, deps_plasticVector):
        self.sig = sig
        self.epsPlastic = epsPlastic
        self.hardening = self.getHardening(epsPlastic)
        self.epsPlasticVector += deps_plasticVector
        self.eps += deps
        self.yieldValue = yieldValue

    def plotMask(self, sampleIndex):
        figTitle = 'Mises_%d_%s' % (
            self.iterationNum, self.loadMode + str(sampleIndex) if 'random' in self.loadMode else self.loadMode)
        if 'net' in self.mode or 'semi' in self.mode:
            figTitle = '%s_Mises_%d_%s' % (self.mode,
                                           self.iterationNum,
                                           self.loadMode + str(sampleIndex) if 'random' in self.loadMode else self.loadMode)
        if 'random' in self.loadMode:
            savePath = 'MCCData'
            figTitle = os.path.join('results', figTitle)
            writeDownPaths(
                path='./MCCData/results',
                data=np.array(self.loadHistoryList),
                sampleIndex=sampleIndex,
                mode=self.mode)
        else:
            savePath = 'figSav'
            figTitle = os.path.join('MisesBaseline', figTitle)
        plotHistory(loadHistory=self.loadHistoryList,
            figTitle=figTitle, savePath=savePath)

    def yieldFunction(self, sig, hardening=None):
        if hardening:
            yieldValue = self.getVonMises(sig) - hardening - self.yieldStress
        else:
            yieldValue = self.getVonMises(sig) - self.hardening - self.yieldStress
        return yieldValue

    def getHardening(self, epsPlastic):
        if 'net' in self.mode:
            hardingValue = self.hardeningNet.prediction(np.array([[epsPlastic]]))[0, 0]
        else:
            ' 500e6*(0.05+eplastic)**0.2'
            if self.nonlinearHardening:
                hardingValue = self.A * (self.epsilon0 + epsPlastic) ** self.n
            else:
                hardingValue = self.A * epsPlastic
        return hardingValue

    def getVonMises(self, sig):
        if 'net' in self.mode:
            vonMises = self.vonMisesNet.prediction(sig.reshape(1, 3))[0, 0]
        elif 'semi' in self.mode:
            vonMises = np.sqrt(sig[0] ** 2 - sig[0] * sig[1] + sig[1] ** 2 + 3. * sig[2] ** 2)
        else:  # 'math' in self.mode
            vonMises = np.sqrt(sig[0] ** 2 - sig[0] * sig[1] + sig[1] ** 2 + 3. * sig[2] ** 2)
        return vonMises

    def tangentAssemble(self, lam, G):
        D = np.zeros([3, 3])
        for i in range(2):
            for j in range(2):
                D[i, j] += lam
        D[0, 0] += 2. * G
        D[1, 1] += 2. * G
        D[2, 2] += G
        return D

    def getAxialDeps(self):
        dEps = np.array(
            [self.depsAxial, -self.D[1, 0] / self.D[1, 1] * self.depsAxial, 0.])
        return dEps

    def getDiffVectorOfYieldFunction(self, sig, epsPlastic):
        if 'net' in self.mode:
            mises, dmises = self.vonMisesNet.prediction2(sig.reshape(1, 3))
            hardening, dhardening = self.hardeningNet.prediction2(np.array([[epsPlastic]]))
            dfds = dmises[0]
            dfdEps_p = -dhardening[0, 0]
        else:
            mises = self.getVonMises(sig)
            if mises == 0:
                dfds = np.array([1., 1., np.sqrt(3.)])
            else:
                dfds = np.array([(2. * sig[0] - sig[1]) / 2. / mises,
                                 (2. * sig[1] - sig[0]) / 2. / mises,
                                 3. * sig[2] / mises])
            '''
                The negative sign is caused by subtracting the hardening term from the yield function
            '''
            if self.nonlinearHardening:
                if 'semi' in self.mode:
                    hardening, dhardening = self.hardeningNet.prediction2(np.array([[epsPlastic]]))
                    dfdEps_p = -dhardening[0, 0]
                else:
                    dfdEps_p = -self.A * self.n * (self.epsilon0 + epsPlastic) ** (self.n - 1.)
            else:
                if 'semi' in self.mode:
                    raise ValueError('There is no NN trained for LinearHardening!')
                dfdEps_p = -self.A
        return dfds, dfdEps_p

    def transiformationSplit(self, deps):
        """
                Used to search the point where the loading
                transform into the plasticity from the ela-
                sticity.

        :return:
        """
        r_min, r_max = 0., 1.0
        r_mid = 0.5 * (r_min + r_max)
        yield_mid = self.yieldFunction(
            self.sig + np.dot(self.D, r_mid * deps),
            hardening=self.getHardening(self.epsPlastic))
        i = 0
        # while yield_mid < -self.yieldTolerance/10. or yield_mid > 0.:
        while yield_mid < self.yieldToleranceNegtive or yield_mid > 0.:
            if yield_mid < 0:
                r_min = r_mid
            else:
                r_max = r_mid
            r_mid = 0.5 * (r_min + r_max)
            yield_mid = self.yieldFunction(
                self.sig + np.dot(self.D, r_mid * deps),
                hardening=self.getHardening(self.epsPlastic))
            if i > 100 and i % 10 == 0:
                print('\titeration: %i yieldValue: %.3f rmid: %.3f' % (i, yield_mid, r_mid))
            i += 1
        return r_mid, yield_mid

    ''' 
    1. The extra components in x direction is sensible or not (this is right)
    2. Check the materialMatrix !!!
    '''

    def plasticReturnMapping(self, deps, sigAfterBisection=None):
        iteration = 0
        if sigAfterBisection is None:
            sigTrial = self.sig + self.D @ deps
        else:
            sigTrial = sigAfterBisection + self.D @ deps
        epsPlastic = self.epsPlastic
        hardening = self.getHardening(epsPlastic)
        yieldValue = self.yieldFunction(sigTrial, hardening=hardening)
        deps_plasticVector = np.zeros(3)
        dfds_mat = np.zeros([3, 1])
        H = 0.
        returnFlag = False
        # sigTrial = self.sig
        """
                Yield surface correction scheme for general elastoplastic models:

            Reference:
            1. Sloan SW, Abbo AJ, Sheng D (2001) Refined explicit integration of 
                elastoplastic models with automatic error control. Eng Comput (Swansea
                , Wales) 18:121–154. https://doi.org/10.1108/02644400110365842

            2. https://github.com/guanshaoheng/NorSand-Jefferies-2015
        """
        while yieldValue > self.yieldTolerancePositive or yieldValue < self.yieldToleranceNegtive:
            returnFlag = True
            # sigMidTrial = .5*(self.sig+sigTrial)
            dFdS, dFdEps_p = self.getDiffVectorOfYieldFunction(
                sig=sigTrial,
                epsPlastic=epsPlastic)
            dfds_mat = dFdS.reshape(-1, 1)
            B = dHdLambda = np.sqrt(2. / 3. * dfds_mat.T @ self.M @ dfds_mat)[0, 0]
            A = h = -dFdEps_p * B
            H = h + (dfds_mat.T @ self.D @ dfds_mat)[0, 0]
            dLambda = yieldValue / (dfds_mat.T @ self.D @ dfds_mat + A)[0, 0]
            # dsig = - yieldValue * self.D @ dFdS / H
            dsig = (-dLambda * self.D @ dfds_mat).reshape(-1)
            # depsPlastic = yieldValue * (-A / dFdEps_p) / H
            depsPlastic = dLambda * B
            sigTrial = dsig + sigTrial
            epsPlastic = depsPlastic + epsPlastic
            hardening = self.getHardening(epsPlastic=epsPlastic)
            deps_plasticVector += dLambda * dFdS
            yieldValue = self.yieldFunction(sigTrial, hardening=hardening)
            iteration += 1
            if self.verboseFlag:
                print('iteration: %d yieldValue: %.8f' % (iteration, yieldValue))
            if iteration >= 20:
                # if yieldValue < 0:
                #     break
                raise ValueError('Iteration number exceeds!!!')
        '''
            Mark: this is where the bug comes from
        '''
        # materialMatrix = A/H*self.D
        if returnFlag:
            materialMatrix = self.D - 1. / H * self.D @ dfds_mat @ dfds_mat.T @ self.D
        else:
            materialMatrix = self.D
        deps_plasticVector = deps_plasticVector.reshape(-1)
        self.hardening = hardening
        return iteration, sigTrial, materialMatrix, deps_plasticVector, epsPlastic, yieldValue

    def plasticReturnMapping_new(self, deps, sigAfterBisection=None):
        iteration = 0
        if sigAfterBisection is None:
            sigTrial = self.sig + self.D @ deps
        else:
            sigTrial = sigAfterBisection + self.D @ deps
        epsPlastic = self.epsPlastic
        hardening = self.getHardening(epsPlastic)
        yieldValue = self.yieldFunction(sigTrial, hardening=hardening)
        deps_plasticVector = np.zeros(3)
        dfds_mat = np.zeros([3, 1])
        H = 0.
        returnFlag = False
        # sigTrial = self.sig
        """
                Yield surface correction scheme for general elastoplastic models:

            Reference:
            1. Sloan SW, Abbo AJ, Sheng D (2001) Refined explicit integration of 
                elastoplastic models with automatic error control. Eng Comput (Swansea
                , Wales) 18:121–154. https://doi.org/10.1108/02644400110365842

            2. https://github.com/guanshaoheng/NorSand-Jefferies-2015
        """
        while yieldValue > self.yieldTolerancePositive or yieldValue < self.yieldToleranceNegtive:
            returnFlag = True
            # sigMidTrial = .5*(self.sig+sigTrial)
            dFdS, dFdEps_p = self.getDiffVectorOfYieldFunction(
                sig=sigTrial,
                epsPlastic=epsPlastic)
            dfds_mat = dFdS.reshape(-1, 1)

            dLambda = 1
            B = dHdLambda = np.sqrt(2. / 3. * dfds_mat.T @ self.M @ dfds_mat)[0, 0]
            A = h = -dFdEps_p * B
            H = h + (dfds_mat.T @ self.D @ dfds_mat)[0, 0]
            dLambda = yieldValue / (dfds_mat.T @ self.D @ dfds_mat + A)[0, 0]
            # dsig = - yieldValue * self.D @ dFdS / H
            dsig = (-dLambda * self.D @ dfds_mat).reshape(-1)
            # depsPlastic = yieldValue * (-A / dFdEps_p) / H
            depsPlastic = dLambda * B
            sigTrial = dsig + sigTrial
            epsPlastic = depsPlastic + epsPlastic
            hardening = self.getHardening(epsPlastic=epsPlastic)
            deps_plasticVector += dLambda * dFdS
            yieldValue = self.yieldFunction(sigTrial, hardening=hardening)
            iteration += 1
            if self.verboseFlag:
                print('iteration: %d yieldValue: %.8f' % (iteration, yieldValue))
            if iteration >= 20:
                # if yieldValue < 0:
                #     break
                raise ValueError('Iteration number exceeds!!!')
        '''
            Mark: this is where the bug comes from
        '''
        # materialMatrix = A/H*self.D
        if returnFlag:
            materialMatrix = self.D - 1. / H * self.D @ dfds_mat @ dfds_mat.T @ self.D
        else:
            materialMatrix = self.D
        deps_plasticVector = deps_plasticVector.reshape(-1)
        self.hardening = hardening
        return iteration, sigTrial, materialMatrix, deps_plasticVector, epsPlastic, yieldValue


def plotHistory(loadHistory, dim=2, vectorLen=3, figTitle=None, savePath='./figSav'):
    load_history = np.array(loadHistory)
    sig = load_history[..., :vectorLen]
    eps = load_history[..., vectorLen:vectorLen * 2]
    epsPlasticVector = load_history[..., (vectorLen * 2 + 3):(vectorLen * 2 + 6)]
    misesStress = load_history[..., vectorLen * 2]
    strainPlastic = load_history[..., vectorLen * 2 + 1]
    hardening = load_history[..., vectorLen * 2 + 2]
    yieldVlue = load_history[..., (vectorLen * 3 + 3):(vectorLen * 3 + 4)]
    iteration = load_history[..., (vectorLen * 3 + 4):(vectorLen * 3 + 5)]
    epsPlastic = load_history[..., (vectorLen * 2 + 1):(vectorLen * 2 + 2)]

    plt.figure(figsize=(16, 7))
    # strain
    ax = plt.subplot(221)
    epsLabel = ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{xy}$'] if dim == 2 else \
        ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{zz}$', '$\epsilon_{xy}$', '$\epsilon_{yz}$',
         '$\epsilon_{xz}$']
    plotSubFigures(ax, x=[range(len(eps)) for _ in range(len(eps[0]))], y=eps.T,
        label=epsLabel,
        xlabel='Load step', ylabel='$\epsilon$', num=vectorLen)

    # yield Value
    ax = plt.subplot(222)
    yieldVlue = yieldVlue.reshape(-1)
    plotSubFigures(ax=ax, x=range(len(sig)), y=yieldVlue, label='yieldValue', xlabel='Load step', ylabel='yieldValue')
    # plt.yscale('log')
    # plt.ylim([np.min(yieldVlue), np.max(yieldVlue)])
    ax2 = ax.twinx()
    ax2.plot(range(len(sig)), iteration, label='iterationNum', color='r', marker='o', lw=3)
    plt.ylabel('iterationNum', fontsize=12)
    plt.ylim([-0.5, 8.0])
    plt.legend(fontsize=15)
    plt.yticks(fontsize=12)

    # stress
    ax = plt.subplot(223)
    sigLabel = ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{xy}$'] if dim == 2 else \
        ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{zz}$', '$\sigma_{xy}$', '$\sigma_{yz}$',
         '$\sigma_{xz}$']
    plotSubFigures(ax, x=[range(len(sig)) for _ in range(len(sig[0]))], y=sig.T,
        label=sigLabel,
        xlabel='Load step', ylabel='$Pa$', num=vectorLen)

    #
    ax = plt.subplot(224)
    epsLabelPlastic = ['$\epsilon_{xx}^p$', '$\epsilon_{yy}^p$', '$\epsilon_{xy}^p$'] if dim == 2 else \
        ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{zz}$', '$\epsilon_{xy}$', '$\epsilon_{yz}$',
         '$\epsilon_{xz}$']
    plotSubFigures(ax, x=[range(len(epsPlasticVector)) for _ in range(len(epsPlasticVector[0]))], y=epsPlasticVector.T,
        label=epsLabelPlastic,
        xlabel='Load step', ylabel='$\epsilon$', num=vectorLen)
    ax2 = ax.twinx()
    plotSubFigures(ax=ax2, x=range(len(sig)), y=epsPlastic, label=r'$\int |\mathrm{d}\bar{\epsilon}^p|$',
        xlabel='Load step', ylabel='$\epsilon$', color='r')

    plt.tight_layout()
    fname = './%s/%s.png' % (savePath, figTitle if figTitle else 'Mises')
    plt.savefig(fname, dpi=200)
    plt.close()
    print('Figrue save as %s' % fname)
    return


def writeDownPaths(path, sampleIndex, data, mode):
    """
    np.array(list(self.sig) + list(self.eps) +
                                                 [self.vonMises, self.epsPlastic, self.hardening] +
                                                 list(self.epsPlasticVector) + [self.yieldValue, iteration])
    :param path:
    :param sampleIndex:
    :param data:
    :return:
    """
    if 'net' in mode:
        name = 'Net_random_%d.dat' % sampleIndex
    elif 'semi' in mode:
        name = 'Semi_random_%d.dat' % sampleIndex
    else:
        name = 'random_%d.dat' % sampleIndex
    filePath = os.path.join(path, name)
    np.savetxt(fname=filePath, X=data, fmt='%10.5f', delimiter=',',
        header='sigma_xx, sigma_yy, sigma_xy, epsilon_xx, epsilon_yy, epsilon_xy, ' +
               'vonMises, epsPlastic, hardening, ' +
               'epsilonP__xx, epsilonP__yy, epsilonP__xy, yieldValue, iteration')


# --------------------------------------------
# main
# load path reader
if __name__ == '__main__':
    baselineFlag = False
    mode = 'math'  # math net semi
    if baselineFlag:
        # ----------------------------------------
        # training data generation  (in conventional triaxial loading mode)
        mises = MisesAssociateFlowIsoHarden(loadMode='axial', nonlinearHardening=True)
        mises.forward()
    else:
        # ----------------------------------------
        # training data generation
        loadPathList = loadingPathReader()[:1]
        print()
        print('=' * 80)
        print('\t Path loading ...')
        for i in range(len(loadPathList)):
            print('\t\tPath %d' % i)
            mises = MisesAssociateFlowIsoHarden(loadMode='random', mode=mode, nonlinearHardening=True)
            mises.forward(path=loadPathList[i], sampleIndex=i)
