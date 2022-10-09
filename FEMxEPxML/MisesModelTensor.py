import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from FEMxEPxML.MCCUtil import plotSubFigures, loadingPathReader, getMaterialMatrix, getQ, getP, \
    getQEps, voigt2tensor, get_dpdsig_dqdsigma, getdEpsMagtitude, tensor2voigt, get_dqeps_deps, \
    getSigma_ts, get_qc_smp, get_dqc_di, get_di_dsig
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
    '''
      In this model  the internal variable is selected as the deviatoric value of the plastic strain:

      \bar{\epsilon}^p = 2/3*\sqrt{3*J2_{\epsilon}}

    '''

    def __init__(self, loadMode='undrained', mode='math',
                 nonlinearHardening=True, verboseFlag=False, p0=1e5,
                 ladeFlag=True, frictionalAngle=30):
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

        # parameters for the Lade criterion
        self.ladeFlag = ladeFlag
        self.phi = frictionalAngle / 180 * np.pi  # friction angle for lade criterion
        self.eta = 4. * np.tan(self.phi) ** 2. * (9. - np.tan(self.phi)) / (1. - np.sin(self.phi))
        self.B = 1.

        # parameters for the Mises criterion
        self.nonlinearHardening = nonlinearHardening
        self.youngsModulus = 0.5e8  # 200e9
        self.poisson = 0.3
        # self.A = 500e6
        self.A = self.youngsModulus/800.
        self.n = 0.5
        self.epsilon0 = 0.01
        self.yieldStress = 0.  # 200e6
        self.hardening = self.getHardening(epsPlastic=0.)
        self.lam = self.youngsModulus * self.poisson / (1 + self.poisson) / (1 - 2 * self.poisson)
        self.G = self.youngsModulus / 2 / (1 + self.poisson)
        self.D = getMaterialMatrix(
            lam=self.lam,
            G=self.G)
        # self.D_inv = np.linalg.inv(self.D)
        self.M = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 0.5]])

        # ---------------------------------------------------
        # state variables [stress and strain vector in voigt notion]
        self.sig = np.diag([p0, p0, p0])
        self.vonMises = getQ(self.sig)
        self.eps = np.zeros([3, 3])
        self.epsPlasticVector = np.ones([3, 3]) * 1e-8
        self.epsPlastic = getQEps(self.epsPlasticVector)
        # self.epsPlastic = np.trace(self.epsPlasticVector)
        self.sigTrial = np.zeros([3, 3])
        self.lastYield = -1
        self.yieldValue = self.yieldFunction(self.sig, hardening=self.getHardening(self.epsPlastic))
        iteration = 0
        self.loadHistoryList = [list(tensor2voigt(self.sig, epsFlag=False)) + \
                                list(tensor2voigt(self.eps, epsFlag=True)) + \
                                [self.vonMises, self.epsPlastic, self.hardening] + \
                                list(tensor2voigt(self.epsPlasticVector, epsFlag=True)) + \
                                [self.yieldValue, iteration]]

        # ---------------------------------------------------
        # load configuration
        self.verboseFlag = verboseFlag
        self.loadMode = loadMode  # 'axial' or 'random'
        if self.loadMode == 'random':
            self.epsAxialObject = 0.004  # random loading
        elif self.loadMode == 'undrained':
            self.epsAxialObject = 0.1
        else:
            raise ValueError('No load mode %s' % self.loadMode)
        self.iterationNum = int(1e3)
        self.depsAxial = self.epsAxialObject / self.iterationNum

        # ---------------------------------------------------
        # Tolerance
        self.yieldTolerance = 2e4
        if 'math' in self.mode:
            self.yieldTolerance = 2e4
        if self.ladeFlag:
            self.yieldTolerance = 1.

    def forward(self, st=None, path=None, sampleIndex=None):
        if 'random' in self.loadMode:
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
                    deps = np.array(deps[0], deps[1], 0., deps[2], 0., 0.)
            elif st is not None:
                deps = st
            elif 'undrained' in self.loadMode:
                deps = np.array([-self.depsAxial * .5, -self.depsAxial * .5, self.depsAxial, 0., 0., 0.])
            else:  # load under the conventional triaxial compression test
                deps = self.getAxialDeps()
                if 0.3 * self.iterationNum < i < 0.65 * self.iterationNum:
                    deps = - deps

            # transform vector to tensor
            deps_tensor = voigt2tensor(vector=deps, epsFlag=True)
            iteration, sigTrial, D_ep, epsPlastic_vector, epsPlastic, yieldValue = self.solver(deps_tensor)
            # yieldValue = self.yieldFunction(self.sig)
            self.updateState(
                sig=sigTrial, deps=deps_tensor, yieldValue=yieldValue,
                epsPlastic=epsPlastic, eps_plasticVector=epsPlastic_vector)
            self.loadHistoryList.append(list(tensor2voigt(self.sig, epsFlag=False)) + \
                                        list(tensor2voigt(self.eps, epsFlag=True)) + \
                                        [self.vonMises, self.epsPlastic, self.hardening] + \
                                        list(tensor2voigt(self.epsPlasticVector, epsFlag=True)) + \
                                        [self.yieldValue, iteration])
        self.loadHistoryList = np.array(self.loadHistoryList)
        self.plotMask(sampleIndex)

    def solver(self, deps):
        sigTrial = self.sig + np.einsum('ijkl, kl->ij', self.D, deps)
        yieldValue = self.yieldFunction(sigTrial, hardening=self.getHardening(self.epsPlastic))
        iteration = 0
        epsPlastic = self.epsPlastic
        epsPlastic_vector = self.epsPlasticVector

        # check if the Gauss point totally fails
        failureFlag, sigTrial_, D_ep_, epsPlastic_vector_, epsPlastic_, yieldValue_ = \
            self.failureCheck(sigTrial=sigTrial, deps=deps)
        if failureFlag:
            return iteration, sigTrial_, D_ep_, epsPlastic_vector_, epsPlastic_, yieldValue_

        # -----------------------------------
        # elastic
        if yieldValue <= 0.:
            D_ep = self.D

        # -----------------------------------
        # plastic and last step is elastic
        elif self.yieldValue < -self.yieldTolerance:
            if self.verboseFlag:
                print()
                print('Bisection!!  yieldValue %.3f LastYieldValue %.3f' % (yieldValue, self.yieldValue))
                print()
            r_mid, yield_mid = self.transiformationSplit(deps)  # searching for the transit point
            sig = self.sig + np.einsum('ijkl, kl->ij', self.D, deps * r_mid)
            depsLeft = (1 - r_mid) * deps
            iteration, sigTrial, D_ep, epsPlastic_vector, epsPlastic, yieldValue = \
                self.plasticReturnMapping(deps=depsLeft, sigAfterBisection=sig)

        # -----------------------------------
        # last step is plastic
        else:
            iteration, sigTrial, D_ep, epsPlastic_vector, epsPlastic, yieldValue = \
                self.plasticReturnMapping(deps=deps, sigAfterBisection=self.sig)

        return iteration, sigTrial, D_ep, epsPlastic_vector, epsPlastic, yieldValue

    def updateState(self, sig, deps, yieldValue, epsPlastic, eps_plasticVector):
        self.sig = sig
        self.vonMises = self.getVonMises(sig=sig)
        self.epsPlastic = epsPlastic
        self.hardening = self.getHardening(epsPlastic)
        self.epsPlasticVector = eps_plasticVector
        self.eps += deps
        self.yieldValue = yieldValue

    def plotMask(self, sampleIndex):
        '''
                [self.sig, self.eps, self.vonMises, self.epsPlastic, self.hardening,

                                    self.epsPlasticVector, self.yieldValue, iteration]
        '''
        figTitle = 'Mises_%d_%s' % (
            self.iterationNum, self.loadMode + str(sampleIndex) if 'random' in self.loadMode else self.loadMode)
        if 'net' in self.mode or 'semi' in self.mode:
            figTitle = '%s_Mises_%d_%s' % (self.mode,
                                           self.iterationNum,
                                           self.loadMode + str(sampleIndex) if 'random' in self.loadMode else self.loadMode)
        if 'random' in self.loadMode:
            savePath = 'misesData'
            figTitle = os.path.join('results', figTitle)
            writeDownPaths(
                path='./misesData/results',
                data=self.loadHistoryList,
                sampleIndex=sampleIndex,
                mode=self.mode)
        else:
            savePath = 'figSav'
            figTitle = os.path.join('MisesBaseline', figTitle)
        plotHistory(loadHistory=self.loadHistoryList,
            figTitle=figTitle, savePath=savePath)

    def yieldFunction(self, sig, hardening):
        if self.ladeFlag:
            yieldValue = np.trace(sig) ** 3. / np.linalg.det(sig) - 27. - self.eta - hardening
        else:  # mises
            yieldValue = getQ(sig) - hardening - self.yieldStress
        return yieldValue

    def getHardening(self, epsPlastic):
        if 'net' in self.mode:
            hardingValue = self.hardeningNet.prediction(np.array([[epsPlastic]]))[0, 0]
        elif self.ladeFlag:
            hardingValue = epsPlastic * self.B
        else:  # mises
            ' 500e6*(0.05+eplastic)**0.2'
            if self.nonlinearHardening:
                hardingValue = self.A * (self.epsilon0 + epsPlastic) ** self.n
            else:
                hardingValue = self.A * epsPlastic + 1e6
        return hardingValue

    def getVonMises(self, sig):
        if 'net' in self.mode:
            vonMises = self.vonMisesNet.prediction(sig.reshape(1, 3))[0, 0]
        else:  # 'math' or 'semi' in self.mode
            vonMises = getQ(sig)
        return vonMises

    def getAxialDeps(self):
        dEps = np.array(
            [self.depsAxial, -self.D[1, 0] / self.D[1, 1] * self.depsAxial, 0.])
        return dEps

    def getDiffVectorOfYieldFunction(self, sig, epsPlastic):
        if 'net' in self.mode:
            mises, dmises = self.vonMisesNet.prediction2(sig.reshape(1, 3))
            hardening, dhardening = self.hardeningNet.prediction2(np.array([[epsPlastic]]))
            dfdsig = dmises[0]
            # dfdEps_p = -dhardening[0, 0]
        elif self.ladeFlag:
            i1, i3 = np.trace(sig), np.linalg.det(sig)
            dfdi = np.array([3. * i1 ** 2. / i3, 0., -i1 ** 3. / i3 ** 2.])
            di_dsig = get_di_dsig(sig)
            dfdsig = np.einsum('i, ikl->kl', dfdi, di_dsig)

            # dfdH = -1.
            # dHdEps_p = self.B * getdepsbar_deps(self.epsPlasticVector)
            # dfdEps_p = dfdH * dHdEps_p

        else:  # mises
            dpdsig, dqdsig = get_dpdsig_dqdsigma(sig)
            dfdsig = dqdsig
            '''
                The negative sign is caused by subtracting the hardening term from the yield function
            '''
            # if self.nonlinearHardening:
            #     if 'semi' in self.mode:
            #         hardening, dhardening = self.hardeningNet.prediction2(np.array([[epsPlastic]]))
            #         dfdEps_p = -dhardening[0, 0]
            #     else:
            #         # dfdeps_bar = -self.A * self.n * (self.epsilon0 + self.epsPlastic) ** (self.n - 1.)
            #         # depsbar_deps = getdepsbar_deps(self.epsPlasticVector)
            #         # dfdEps_p = dfdeps_bar * depsbar_deps
            # else:  # linear hardening
            #     if 'semi' in self.mode:
            #         raise ValueError('There is no NN trained for LinearHardening!')
            #     depsbar_deps = getdepsbar_deps(self.epsPlasticVector)
            #     dfdEps_p = -self.A * depsbar_deps
        return dfdsig

    def failureCheck(self, sigTrial, deps):
        # if the material is failed
        if getP(sigTrial) < 0.:
            if self.verboseFlag:
                print('Material fails P=%.3e, all of the strain is set as plastic strain' % getP(sigTrial))
            sigTrial = self.sig
            D_ep = np.zeros_like(self.D)
            epsPlastic_vector = self.epsPlasticVector + deps
            epsPlastic = self.epsPlastic + getQEps(deps)
            yieldValue = 0.
            return True, sigTrial, D_ep, epsPlastic_vector, epsPlastic, yieldValue
        else:
            return False, None, None, None, None, None

    def getDiffVectorOfG(self, sig_ts):
        dpdsig, dqdsig = get_dpdsig_dqdsigma(sig_ts)
        dgdsig = dqdsig
        return dgdsig

    def transiformationSplit(self, deps):
        """
                Used to search the point where the loading
                transform into the plasticity from the ela-
                sticity.

        :return:
        """
        incremental = np.einsum('ijkl, kl->ij', self.D, deps)
        r_min, r_max = 0., 1.0
        r_mid = 0.5 * (r_min + r_max)
        sigTrial = self.sig + incremental * r_mid
        # sigTrial_ts = getSigma_ts(sigTrial, p=getP(sigTrial), q=getQ(sigTrial), qc=get_qc_smp(sigTrial))
        yield_mid = self.yieldFunction(
            sigTrial,
            hardening=self.hardening)
        i = 0
        while yield_mid < -self.yieldTolerance:
            if yield_mid < 0:
                r_min = r_mid
            else:
                r_max = r_mid
            r_mid = 0.5 * (r_min + r_max)
            sigTrial = self.sig + incremental * r_mid
            yield_mid = self.yieldFunction(
                sigTrial,
                hardening=self.hardening)
            if i > 100 and i % 10 == 0:
                print('\titeration: %i last yieldValue: %.3f yieldValue: %.3f rmid: %.3f' %
                      (i, self.yieldValue, yield_mid, r_mid))
            i += 1
        return r_mid, yield_mid

    ''' 
    1. The extra components in x direction is sensible or not (this is right)
    2. Check the materialMatrix !!!
    '''

    def plasticReturnMapping(self, deps, sigAfterBisection=None):
        iteration = 0
        if sigAfterBisection is None:
            sigTrial = self.sig + np.einsum('ijkl, kl->ij', self.D, deps)
        else:
            sigTrial = sigAfterBisection + np.einsum('ijkl, kl->ij', self.D, deps)

        yieldValueTrial = self.yieldFunction(sigTrial, self.hardening)
        if self.verboseFlag:
            print('\t Trial yield value = %.5f' % yieldValueTrial)

        # check if the Gauss point totally fails
        failureFlag, sigTrial_, D_ep_, epsPlastic_vector_, epsPlastic_, yieldValue_ = \
            self.failureCheck(sigTrial=sigAfterBisection, deps=deps)
        if failureFlag:
            return iteration, sigTrial_, D_ep_, epsPlastic_vector_, epsPlastic_, yieldValue_

        yieldValue_last = self.yieldFunction(sigAfterBisection, hardening=self.hardening)
        """
                Yield surface correction scheme for general elastoplastic models:

            Reference:
            1. Sloan SW, Abbo AJ, Sheng D (2001) Refined explicit integration of 
                elastoplastic models with automatic error control. Eng Comput (Swansea
                , Wales) 18:121–154. https://doi.org/10.1108/02644400110365842

            2. https://github.com/guanshaoheng/NorSand-Jefferies-2015
        """
        while True:
            '''
                dFdS, dFdEps_p are in the original space OS
                
                dgds are in the transformed space TS
            '''
            dFdS = self.getDiffVectorOfYieldFunction(
                sig=sigAfterBisection,
                epsPlastic=self.epsPlastic)

            # non-association
            dgds = dFdS
            temp1 = np.einsum('ij, ijkl, kl->', dFdS, self.D, deps)
            temp2 = np.einsum('ij, ijkl, kl->', dFdS, self.D, dgds)
            # temp3 = np.einsum('ij, ij->', dFdEps_p, dgds)

            # calculate the relation between yield value and the eplstic deformation
            if self.nonlinearHardening:
                dHdeps_p = self.n*(self.epsilon0+self.epsPlastic)**(self.n-1.)
            else:
                dHdeps_p = 1
            if self.ladeFlag:
                temp3 = -self.B * dHdeps_p * getQEps(dgds)
            else:  # mises
                temp3 = -self.A * dHdeps_p * getQEps(dgds)

            dLambda = (temp1 + yieldValue_last) / (temp2 - temp3)
            deps_plasticVector = dLambda * dgds
            dsig = np.einsum('ijkl, kl', self.D, deps - deps_plasticVector)

            # renewation
            sigTrial = sigAfterBisection + dsig
            epsPlastic_vector = self.epsPlasticVector + deps_plasticVector
            epsPlastic = self.epsPlastic + getQEps(deps_plasticVector)
            # epsPlastic = np.trace(epsPlastic_vector)
            H = self.getHardening(epsPlastic)
            D_ep = self.D - np.einsum('ijmn, mn, st, stkl->ijkl', self.D, dgds, dFdS, self.D) / (temp2 - temp3)
            # D_ep = self.D*(1.-temp2/(temp2-temp3))

            yieldValue = self.yieldFunction(sigTrial, hardening=H)

            iteration += 1
            if self.verboseFlag:
                print('iteration: %d yieldValue: %.8f' % (iteration, yieldValue))
            if iteration >= 20:
                # if yieldValue < 0:
                #     break
                raise ValueError('Iteration number exceeds!!!')

            break
        return iteration, sigTrial, D_ep, epsPlastic_vector, epsPlastic, yieldValue


def plotHistory(loadHistory, dim=3, vectorLen=6, figTitle=None, savePath='./figSav'):
    '''
                    [self.sig, self.eps, self.vonMises, self.epsPlastic, self.hardening,

                                        self.epsPlasticVector, self.yieldValue, iteration]
    '''
    load_history = np.array(loadHistory)
    sig = load_history[..., :vectorLen]
    sig_tensor = np.array([voigt2tensor(vector=sig[i], epsFlag=False) for i in range(len(sig))])
    eps = load_history[..., vectorLen:vectorLen * 2]
    epsPlasticVector = load_history[..., (vectorLen * 2 + 3):(vectorLen * 3 + 3)]
    misesStress = load_history[..., vectorLen * 2]
    epsPlastic = load_history[..., vectorLen * 2 + 1]
    hardening = load_history[..., vectorLen * 2 + 2]
    yieldVlue = load_history[..., vectorLen * 3 + 3]
    iteration = load_history[..., vectorLen * 3 + 4]
    p = np.average(sig[:, :3], axis=1)
    q = np.array([getQ(sig_tensor[i]) for i in range(len(sig_tensor))])
    eps_p_v = np.sum(epsPlasticVector[:, :3], axis=1)

    plt.figure(figsize=(7, 16))
    # strain
    ax = plt.subplot(411)
    epsLabel = ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{xy}$'] if dim == 2 else \
        ['$\epsilon_{xx}$', '$\epsilon_{yy}$', '$\epsilon_{zz}$', '$\epsilon_{xy}$', '$\epsilon_{yz}$',
         '$\epsilon_{xz}$']
    plotSubFigures(ax, x=[range(len(eps)) for _ in range(len(eps[0]))], y=eps.T,
        label=epsLabel,
        xlabel='Load step', ylabel='$\epsilon$', num=vectorLen)

    # yield Value
    ax = plt.subplot(412)
    yieldVlue = yieldVlue.reshape(-1)
    plotSubFigures(ax=ax, x=[range(len(sig))], y=[yieldVlue], num=1, label=[
        'yieldValue'], xlabel='Load step', ylabel='yieldValue')
    # plt.yscale('log')
    # plt.ylim([np.min(yieldVlue), np.max(yieldVlue)])
    ax2 = ax.twinx()
    ax2.plot(range(len(sig)), iteration, label='iterationNum', color='r', marker='o', lw=3)
    plt.ylabel('iterationNum', fontsize=12)
    plt.ylim([-0.5, 8.0])
    plt.legend(fontsize=15)
    plt.yticks(fontsize=12)

    # stress
    ax = plt.subplot(413)
    # sigLabel = ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{xy}$'] if dim == 2 else \
    #     ['$\sigma_{xx}$', '$\sigma_{yy}$', '$\sigma_{zz}$', '$\sigma_{xy}$', '$\sigma_{yz}$',
    #      '$\sigma_{xz}$']
    # plotSubFigures(ax, x=[range(len(sig)) for _ in range(len(sig[0]))], y=sig.T,
    #     label=sigLabel,
    #     xlabel='Load step', ylabel='$Pa$', num=vectorLen)
    sigLabel = ['$p-q$']
    plotSubFigures(ax, x=[p / 1e6], y=[q / 1e6],
        label=sigLabel,
        xlabel='$p$(MPa)', ylabel='$q$(MPa)', num=1)
    # ax2 = ax.twinx()
    # plotSubFigures(ax=ax2, x=[range(len(sig))], y=[q], num=1, label=[r'$q$'],
    #     xlabel='Load step', ylabel='$Pa$', color='r')


    # plastic strain
    ax = plt.subplot(414)
    epsLabelPlastic = ['$\epsilon_{v}^p$']
    plotSubFigures(ax, x=[range(len(eps_p_v))], y=[eps_p_v],
        label=epsLabelPlastic,
        xlabel='Load step', ylabel='$\epsilon_v$', num=1)
    ax2 = ax.twinx()
    plotSubFigures(ax=ax2, x=[range(len(sig))], y=[epsPlastic], num=1, label=[r'$\int |\mathrm{d}\bar{\epsilon}^p|$'],
        xlabel='Load step', ylabel=r'$\int |\mathrm{d}\bar{\epsilon}^p|$', color='r')

    fname = './%s/%s.png' % (savePath, figTitle if figTitle else 'Mises')
    plt.show()
    # plt.savefig(fname, dpi=200)
    plt.close()
    print('Figrue save as %s' % fname)

    ax = plt.subplot(111)
    plotSubFigures(ax, x=[range(len(q))], y=[q/1e6], num=1, label=['$q-\epsilon_{axial}$'],
    xlabel='Load step', ylabel='$q (MPa)$')
    plt.show()

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
        header='sigma_xx, sigma_yy, sigma_zz, sigma_xy, sigma_yz, sigma_xz, ' + \
               'epsilon_xx, epsilon_yy, epsilon_zz, epsilon_xy, epsilon_yz, epsilon_xz, ' +
               'vonMises, epsPlastic, hardening, ' +
               'epsilonP_xx, epsilonP_yy, epsilonP_zz, epsilonP_xy, epsilonP_yz, epsilonP_xz, ' + \
               'yieldValue, iteration')


# --------------------------------------------
# main
# load path reader
if __name__ == '__main__':
    baselineFlag = False
    loadMode = 'undrained'
    mode = 'math'  # math net semi
    if baselineFlag:
        # ----------------------------------------
        # training data generation  (in conventional triaxial loading mode)
        mises = MisesAssociateFlowIsoHarden(loadMode='axial', nonlinearHardening=True)
        mises.forward()
    elif 'undrained' in loadMode:
        mises = MisesAssociateFlowIsoHarden(loadMode=loadMode, verboseFlag=True)
        mises.forward()
    else:
        # ----------------------------------------
        # training data generation
        loadPathList = loadingPathReader(path='./misesData')[:50]
        print()
        print('=' * 80)
        print('\t Path loading ...')
        for i in range(len(loadPathList)):
            print('\t\tPath %d' % i)
            mises = MisesAssociateFlowIsoHarden(loadMode='random', mode=mode, nonlinearHardening=True)
            mises.forward(path=loadPathList[i], sampleIndex=i)
