import copy
import os
import warnings
import sympy
import numpy as np
import matplotlib.pyplot as plt
# import sympy
# from sciptes4figures.plotConfiguration2D import plotConfiguration2D
from FEMxEPxML.misesTraining import Restore, Net, DenseResNet
from FEMxEPxML.utils_constitutive import getQ, getP, getS, get_dpdsig_dqdsigma, getJ2, getQEps, getJ2Eps, getEpsDevitoric, \
    getVolStrain, get_M, get_principle_stress
import warnings
from utilSelf.general import  echo

'''
    Author: Shaoheng Gun
            Wuhan University & Swansea University
    Email: shaohengguan@gmail.com
    
    Critical state theory involved Unified Hardening model for both sand and clay

        in tensor notion

        Reference:
        [1] Yao, Y. P., Liu, L., Luo, T., Tian, Y., & Zhang, J. M. (2019).
            Unified hardening (UH) model for clays and sands. Computers and Geotechnics,
            110(March), 326–343. https://doi.org/10.1016/j.compgeo.2019.02.024

'''


class CSUH:
    def __init__(self, Z=None, p0=1e5, e0=0.833,
                 e_c0=0.934, theta_degree=30, lambdaa=0.135, kappa=0.04,
                 poisson=0.3, N=1.973, chi=0.0, m=1.8, verboseFlag=True):
        # -----------------Parameters (fundamental)------------------
        self.M = get_M(theta_degree=theta_degree)  # ratio at critical state
        self.lambdaa = lambdaa
        self.kappa = kappa
        self.poisson = poisson
        self.N = N  # location in normal consolidation on e-lnp space
        self.chi = chi
        self.m = m
        # self.pa = 1e5  # standard pressure
        self.e0 = e0  # intial void ratio
        self.p0 = p0
        self.c_p = (self.lambdaa - self.kappa) / (1. + self.e0)
        if e_c0 is not None:
            self.e_c0 = e_c0  # void ratio on the critical state line at the mean effective stress p=0 kPa
            ''' Eq. (47)'''
            self.ps = np.exp((self.N - self.e_c0) / self.lambdaa)  # compressive hardening parameter corresponding to Z
            # self.Z = self.e_c0 - self.lambdaa * np.log((1+self.ps) / self.ps)
            self.Z = self.N - self.lambdaa * np.log(self.ps + 1.)
        elif Z is not None:
            self.Z = Z
            self.e_c0 = self.N - np.log(np.exp((self.N - self.Z) / self.lambdaa) - 1.) * self.lambdaa
            ''' Eq. (47)'''
            self.ps = np.exp((self.N - self.e_c0) / self.lambdaa)  # compressive hardening parameter corresponding to Z
        else:
            raise ValueError('According to Eq. (45), Z or e_c0, only one is needed!')

        print(self.Z)

        # -----------------States (calculated)------------------
        # according to the current stress and void ratio state
        self.q, self.p = 0., self.p0
        self.K, self.G, self.lam = self.getElasticModulus(self.p, self.e0)
        self.D = self.getMaterialMatrix(lam=self.lam, G=self.G)
        self.sigma = np.diag([self.p, self.p, self.p])

        # self.sigma_principal = np.sort(get_principle_stress(sigma=self.sigma))
        # self.b = get_b(*self.sigma_principal)
        # self.lode = getLode(b=self.b)
        self.e = self.e0
        self.eta = self.q / self.p
        self.e_eta = self.get_e_eta(eta=self.eta, p=self.p)
        self.xi = self.e_eta - self.e
        self.over_overconsolidation_ratio = np.exp(-self.xi / (self.lambdaa - self.kappa))
        # self.px0 = self.over_overconsolidation_ratio*self.p0/1e3 if self.over_overconsolidation_ratio>1. else self.p0/1e3
        self.px0 = self.p0/1e3
        self.M_c = self.getM_c(xi=self.xi)
        self.M_f = self.getM_f(xi=self.xi)

        # -----------------Reference yield surface (calculated)------------------
        self.epsvp = 0.

        # -----------------Calculation parameters-----------------
        self.yieldTolerance = 0.05

        # -----------------Current yield surface (calculated)------------------
        self.H = 0.
        self.yieldValue = self.yieldFunction(q=self.q, p=self.p, H=0., px0=self.px0)
        if self.yieldValue > 0.:
            temp = self.M ** 2 * (self.p / 1e3) ** 2 - self.chi * (self.q/ 1e3) ** 2
            self.px0 = (1. + (self.q / 1e3) ** 2. / temp) * self.p / 1e3

        # ==================== initialization ended ======================
        self.verboseFlag = verboseFlag

        # ==================== Calculation results
        self.results = [[self.p, self.q, self.e, self.H, self.epsvp,
                         self.xi, self.M_c, self.M_f] + [self.D[0, 0, 0, 0], self.D[1, 1, 1, 1],
                                                                  self.D[2, 2, 2, 2]]]
        self.ocr = (np.exp(self.xi/(self.lambdaa-self.kappa))*(self.p+1e3*self.ps)-1e3*self.ps)/self.p
        echo('e0: %.3e xi: %.3e ocr: %.3e lam: %.3e G: %.3e lambdaa: %.3e kappa: %.3e m: %.3e Z: %.3e N: %.3e' %
             (self.e0, self.xi, self.ocr, self.lam, self.G, self.lambdaa, self.kappa, self.m, self.Z, self.N))

    def forward(self):
        '''
        undrained compression: 1

        The compression is positive and the extension is negative
        '''
        axialStrainObject = 0.30
        axialStrainArray = np.linspace(0., axialStrainObject, 1000)
        for i in range(1, len(axialStrainArray)):
            if self.verboseFlag:
                print('\t step %d' % i)
            depsAxial = axialStrainArray[i] - axialStrainArray[i - 1]
            deps = np.diag([-0.5*depsAxial, -0.5*depsAxial, depsAxial])
            scaler, remain_scaler, split_num = 1.0, 1.0, 0
            sig, D_ep, scence = self.solver(deps=deps)
            _, e, p, q, xi, yieldValue, H, epsvp = scence
            self.update(sig=sig, e=e, p=p, q=q, xi=xi, yieldValue=yieldValue, H=H, epsvp=epsvp)
            print('%d p: %.3e q: %.3e xi: %.3e eps_v:%.3e' %
                  (i, getP(sig), getQ(sig), xi, epsvp))
            # while True:
            #     try:
            #         sig, e, p, q, xi, yieldValue, H, epsvp, D_ep = self.solver(deps=deps)
            #         self.updateState(sig=sig, e=e, p=p, q=q, xi=xi, yieldValue=yieldValue, H=H, epsvp=epsvp)
            #         remain_scaler -= scaler
            #     except:
            #         split_num += 1
            #         if split_num>10:
            #             self.plotCurrentResults()
            #             raise
            #         scaler *= 0.5
            #         deps = np.diag([scaler*depsAxial, -.5 * scaler*depsAxial, -.5 * scaler*depsAxial])
            #         print('Unconverge! split_num=%d ' % split_num)
            #     if remain_scaler <= 0.:
            #         break
            self.results.append([p, q, e, H, epsvp,
                                 xi, self.getM_c(xi), self.getM_f(xi)] +
                                [D_ep[0, 0, 0, 0], D_ep[1, 1, 1, 1], D_ep[2, 2, 2, 2]])
        self.plotCurrentResults()
        self.writeDown()

    def solver(self, deps):
        '''
            The calculation is implemented in the Original Space instead of the Transformation Space (TS)
        '''
        e = self.e - (self.e + 1.) * getVolStrain(deps)
        sig = self.sigma + np.einsum('ijkl, kl->ij', self.D, deps)
        # I123 = getInvariantsSigma(sigma=sig)
        p = getP(sigma=sig)
        q = getQ(sigma=sig)
        '''
            If the loading step is too large, then the trial mean stress may be a negative number.
            
            Then we have to split the loading step.
        '''
        if p < 0.:
            sig = self.sigma
            q = getQ(self.sigma)
            p = getP(self.sigma)
            deps_p = deps
            depsvp = np.trace(deps_p)
            D_ep = np.zeros_like(self.D)
            xi = self.xi
            H = self.H
            yieldValue = self.yieldFunction(q, p, H, self.px0)
            epsvp = self.epsvp+depsvp
            if self.verboseFlag:
                print('Failed element in the elastic trial stage!')
            scence = [sig, e, p, q, xi, yieldValue, H, epsvp]
            return sig,  D_ep, scence
        eta = q / p
        yieldValue = self.yieldFunction(q=q, p=p, H=self.H, px0=self.px0)

        # ------------ Elastic --------------
        if yieldValue < 0.:
            if self.verboseFlag:
                print('\t\t elastic')
            e_eta = self.get_e_eta(eta=eta, p=p)
            xi = e_eta - e
            K, G, lam = self.getElasticModulus(p=p, e=e)
            D = self.getMaterialMatrix(lam=lam, G=G)
            scence = [sig, e, p, q, xi, yieldValue, self.H, self.epsvp]
            return sig, D, scence
        # ------------ Plastic while last step is elastic --------------
        elif yieldValue > 0. and self.yieldValue < -self.yieldTolerance:
            if self.verboseFlag:
                print('\t\t elastic 2 plastic')
            rmid, sig, yieldValue = self.transformSplit(deps=deps)
            e = self.e - (self.e + 1.) * getVolStrain(deps*rmid)
            p, q = getP(sigma=sig), getQ(sigma=sig)
            K, G, lam = self.getElasticModulus(p=p, e=e)
            D = self.getMaterialMatrix(lam=lam, G=G)
            xi_last = self.get_e_eta(eta=q/p, p=p)-e
            Mc_last, Mf_last = self.getM_c(xi=xi_last), self.getM_f(xi=xi_last)
            try:
                sig, D_ep, scence = self.returnMapping(
                    deps=deps * (1. - rmid), sig_last=sig, D_last=D, elast=e, p_last=p, q_last=q,
                    yieldValue_last=yieldValue, xi_last=xi_last, Mc_last=Mc_last, Mf_last=Mf_last)
            except:
                return False
        # ------------ Plastic --------------
        else:
            if self.verboseFlag:
                print('\t\t plastic')
            try:
                sig, D_ep, scence = self.returnMapping(
                    deps=deps, sig_last=self.sigma, D_last=self.D, elast=self.e, p_last=self.p,
                    q_last=self.q, yieldValue_last=self.yieldValue,
                    xi_last=self.xi, Mc_last=self.M_c, Mf_last=self.M_f)
            except:
                return False
        return sig, D_ep, scence

    def transformSplit(self, deps):
        rmin, rmax, rmid = 0., 1., 0.5
        sig = self.sigma + np.einsum('ijkl, kl->ij', self.D, deps * rmid)
        p = getP(sigma=sig)
        q = getQ(sigma=sig)
        yieldValue = self.yieldFunction(q=q, p=p, H=self.H, px0=self.px0)
        while True:
            if yieldValue > 0.:
                rmax = rmid
            elif yieldValue < -self.yieldTolerance:
                rmin = rmid
            else:
                break
            rmid = .5 * (rmax + rmin)
            sig = self.sigma + np.einsum('ijkl, kl->ij', self.D, deps * rmid)
            p = getP(sigma=sig)
            q = getQ(sigma=sig)
            yieldValue = self.yieldFunction(q=q, p=p, H=self.H, px0=self.px0)
        return rmid, sig, yieldValue

    def returnMapping(self, deps, sig_last, D_last,
                      elast, p_last, q_last, yieldValue_last,
                      xi_last, Mc_last, Mf_last):
        e = elast - (1. + elast) * getVolStrain(deps)
        eta_last = q_last / p_last
        if p_last < 0.:
            sig = sig_last
            q = q_last
            p = p_last
            deps_p = deps
            depsvp = np.trace(deps_p)
            D_ep = np.zeros_like(self.D)
            xi = xi_last
            H = self.H
            yieldValue = self.yieldFunction(q, p, H, self.px0)
            epsvp = self.epsvp + depsvp
            scence = [self.sigma, e, self.p, self.q, xi, self.yieldValue, self.H, epsvp]
            if self.verboseFlag:
                print('Failed element in the plastic return mapping stage!')
            return self.sigma, np.zeros_like(self.D), scence
        # return mapping
        returnMapping_iter = 0
        while True:
            dg_dsigma = self.get_dg_dsigma(Mc=Mc_last, eta=eta_last, p=p_last, sigma=sig_last)
            df_dsigma, df_depsvp = self.get_df_dsigma_df_depsp(
                Mf=Mf_last, Mc=Mc_last, sigma=sig_last, p=p_last, q=q_last)
            temp = np.einsum('ij, ijkl, kl->', df_dsigma, D_last, dg_dsigma) - \
                   df_depsvp * np.trace(dg_dsigma)
            A = (np.einsum('ij, ijkl, kl->', df_dsigma, D_last, deps) +
                    yieldValue_last if yieldValue_last < 1e5 else 0.) / temp
            # A = np.einsum('ij, ijkl, kl->', df_dsigma, D_last, deps) / temp
            deps_p = A * dg_dsigma
            depsvp = np.trace(deps_p)
            # Following calculation is better than "sig = sig_last+np.einsum('ijkl, kl->ij', D_ep, deps)"
            # for the nonlinear iteration
            sig = sig_last + np.einsum('ijkl, kl->ij', D_last, deps - deps_p)
            p = getP(sigma=sig)
            q = getQ(sigma=sig)
            if p < 0.:
                sig = sig_last
                q = q_last
                p = p_last
                deps_p = deps
                depsvp = np.trace(deps_p)
                D_ep = np.zeros_like(self.D)
                xi = xi_last
                H = self.H
                yieldValue = self.yieldFunction(q, p, H, self.px0)
                epsvp = self.epsvp+depsvp
                scence = [self.sigma, e, self.p, self.q, xi, self.yieldValue, self.H, epsvp]
                if self.verboseFlag:
                    print('Failed element in the plastic return mapping stage!')
                return self.sigma, np.zeros_like(self.D), scence
                # raise ValueError('The mean stress after return mapping got negtive (p = %.3e)' % p)
            # calculate the updated state variables
            eta= q / p
            xi = self.get_e_eta(eta=eta, p=p) - e
            epsvp = self.epsvp + depsvp
            Mf_last = 0.5 * (Mf_last + self.getM_f(xi))
            Mc_last = 0.5 * (Mc_last + self.getM_c(xi))
            eta_last = .5 * (eta_last + eta)
            D_ep = D_last - \
                   np.einsum('ijmn, mn, st, stkl->ijkl', D_last, dg_dsigma, df_dsigma, D_last) / temp
            H = self.H + self.get_dH(
                Mf=Mf_last,
                Mc=Mc_last,
                eta=eta_last, depsvp=depsvp)
            yieldValue = self.yieldFunction(q=q, p=p, H=H, px0=self.px0)
            if np.abs(yieldValue) < self.yieldTolerance:
                if self.verboseFlag:
                    print('converged!')
                break
            else:
                if self.verboseFlag:
                    print('\t\t\t Yield value: %.3e' % yieldValue)

            returnMapping_iter += 1
            break
        scence = [sig, e, p, q, xi, yieldValue, H, epsvp]
        return sig, D_ep, scence

    def update(self, sig, e, p, q, xi, yieldValue, H, epsvp):
        '''
            sig, e, p, q, xi, yieldValue, H, epsvp
             0   1  2  3   4     5        6    7
        '''
        self.sigma = sig
        self.e = e
        self.p = p
        self.q =q
        self.xi =xi
        self.yieldValue = yieldValue
        self.H = H
        self.epsvp = epsvp
        self.K, self.G, self.lam = self.getElasticModulus(p=self.p, e=self.e)
        self.D = self.getMaterialMatrix(lam=self.lam, G=self.G)
        self.M_c = self.getM_c(self.xi)
        self.M_f = self.getM_f(self.xi)

    def plotCurrentResults(self):
        '''
                  0  1  2  3    4    5         6                 7         8-8+3
                 [p, q, e, H, epsvp, xi, self.getM_c(xi), self.getM_f(xi), D_ep]
        '''
        results = np.array(self.results[:-1])
        length = len(results)
        p, q, e, H, epsvp = results[:, 0] / 1e6, results[:, 1] / 1e6, results[:, 2], results[:, 3], results[:, 4]
        xi, M_c, M_f = results[:, 5], results[:, 6], results[:, 7]
        d0000, d1111, d2222 = results[:, 8], results[:, 9], results[:, 10]
        eta = q / p

        # plt.plot(p[:100], q[:100]);plt.axis('equal');plt.tight_layout();plt.show()
        #
        fig = plt.figure(figsize=[12, 6])
        plt.xticks([])
        plt.yticks([])
        plt.title('$e_{0}$=%.3f p0=%.3f kPa px0=%.3f' % (self.e0, self.p0 / 1e3, self.px0/1e3))

        ax = fig.add_subplot(241)
        plt.plot(p, q, label='q-p')
        plt.xlabel('p MPa')
        plt.ylabel('q MPa')
        plt.axis('equal')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(242)
        plt.plot(range(length), H, label='H')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(243)
        plt.plot(range(length), epsvp, label='$\epsilon_v^p$')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(244)
        plt.plot(range(length), q, 'r', label='$q$')
        plt.legend(loc='upper left')
        plt.tight_layout()
        ax1 = ax.twinx()
        plt.plot(p, label='p')
        # plt.plot(range(length), d0000, label='$D_{0000}$')
        # plt.plot(range(length), d1111, label='$D_{1111}$')
        # plt.plot(range(length), d2222, label='$D_{2222}$')
        plt.legend(loc='lower right')
        plt.tight_layout()

        ax = fig.add_subplot(245)
        plt.plot(range(length), M_f, label=r'$M_{f}$')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(246)
        plt.plot(range(length), M_c, label='$M_{c}$')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(247)
        plt.plot(range(length), eta, label='$\eta$')
        plt.tight_layout()
        plt.legend()

        ax = fig.add_subplot(248)
        plt.plot(eta, M_c, label=r'$M_{c}-\eta$')
        plt.plot(eta, eta, 'r-.')
        plt.axis('equal')
        plt.tight_layout()
        plt.legend(loc='upper left')
        ax1 = ax.twinx()
        plt.plot(eta, M_f, 'r', label=r'$M_{f}-\eta$')
        plt.axis('equal')
        plt.tight_layout()
        plt.legend(loc='lower right')

        fig_name = os.path.join('CSUHresults', 'Toyoura_e0_%.3f_p0_%.3fkPa.png' % (self.e0, self.p0 / 1e3))
        plt.show()
        # plt.savefig(fig_name)
        plt.close()
        return

    def writeDown(self):
        txt_name = os.path.join('CSUHresults', 'Toyoura_e0_%.3f_p0_%.3fkPa.dat' % (self.e0, self.p0 / 1e3))
        np.savetxt(fname=txt_name, X=np.array(self.results))

    def get_e_eta(self, eta, p):
        p = p / 1e3
        ''' UH model '''
        # e_eta = self.N-self.lambdaa*np.log(self.p)-(self.lambdaa-self.kappa)*np.log(1.+self.eta**2./self.M**2.)
        ''' CSUH model'''
        e_eta = self.Z - self.lambdaa * np.log((p + self.ps) / (1. + self.ps)) - \
                (self.lambdaa - self.kappa) * \
                np.log(((self.M ** 2 + eta ** 2) / (self.M ** 2 - self.chi * eta ** 2) * p + self.ps) / (p + self.ps))
        return e_eta

    def getM_c(self, xi):
        ''' Eq. (33) '''
        return self.M * np.exp(-self.m * xi)

    def getM_f(self, xi):
        # R = np.exp(-xi/(self.lambdaa-self.kappa))
        # k = self.M**2/(12.*(3.-self.M))
        # temp = k/R
        # mf1 = 6.*(np.sqrt(temp*(1.+temp))-temp)
        ''' Eq. (12) '''
        mf = 6. / (np.sqrt(12. * (3. - self.M) / self.M ** 2 *
                           np.exp(-xi / (self.lambdaa - self.kappa)) + 1.) + 1.)
        return mf

    def get_dH(self, Mf, Mc, eta, depsvp):
        return (Mf ** 4 - eta ** 4) / (Mc ** 4 - eta ** 4) * depsvp

    def yieldFunction(self, q, p, H, px0):
        p /= 1e3
        q /= 1e3
        temp = self.M ** 2 * p ** 2 - self.chi * q ** 2
        if temp < 0.:
            f = 1e32
            return f
        f = np.log((1. + (1 + self.chi) * q ** 2. / temp) * p + self.ps) - \
            np.log(px0 + self.ps) - \
            H / self.c_p
        return f

    def getElasticModulus(self, p, e):
        K = (1. + e) / self.kappa * (p + self.ps*1e3)
        G = 3. * (1 - 2 * self.poisson) * K / 2. / (1. + self.poisson)
        lam = K - 2. / 3. * G
        return K, G, lam

    def getMaterialMatrix(self, lam, G):
        matrix = np.zeros(shape=[3, 3, 3, 3])
        for i in range(3):
            for j in range(3):
                matrix[i, i, j, j] += lam
        for i in range(3):
            matrix[i, i, i, i] += 2. * G
            matrix[i, (i + 1) % 3, i, (i + 1) % 3] = \
                matrix[i, (i + 1) % 3, (i + 1) % 3, i] = \
                matrix[(i + 1) % 3, i, (i + 1) % 3, i] = \
                matrix[(i + 1) % 3, i, i, (i + 1) % 3] = G
        return matrix

    # def get_q_c(self, I1, I2, I3):
    #     ''' Reference eq. (50) in paper Unified hardening (UH) model for clays and sands '''
    #     q_c = 2. * I1 / (3. * np.sqrt((I1 * I2 - I3) / (I1 * I2 - 9. * I3)) - 1.) if I1 * I2 - 9. * I3 > 0. else 0.
    #     return q_c

    def get_dg_dsigma(self, Mc, eta, p, sigma):
        term1 = (Mc ** 2. - eta ** 2.) * np.eye(3) / 3.
        term2 = 3. * (sigma - p * np.eye(3)) / p
        v = (term1 + term2) / p/ (Mc ** 2. + eta ** 2.)
        return v

    def get_df_dsigma_df_depsp(self, Mf, Mc, sigma, p, q):
        '''  sigma = np.random.random(size=[3, 3])
             sigma = 0.5*(sigma+sigma.T)
        '''
        eta = q / p
        dfdp_up = self.M ** 4 - (1. + 3. * self.chi) * self.M ** 2. * eta ** 2. - self.chi * eta ** 4.
        dfdp_low = p * (self.M ** 2. - self.chi * eta ** 2.) * \
                   (self.M ** 2. + eta ** 2. + (self.M ** 2. - self.chi * eta ** 2.) * self.ps / p)
        dfdq_up = 2. * self.M ** 2. * (1 + self.chi) * eta
        dfdp = dfdp_up / dfdp_low
        dfdq = dfdq_up / dfdp_low
        '''
         NOTE: calculation in sympy is very slow!!! So the Sympy package is not used in this version
        '''
        dp_dsigma, dq_dsigma = get_dpdsig_dqdsigma(sigma=sigma)
        dfdsigma = dfdp * dp_dsigma + dfdq * dq_dsigma
        df_depsvp = -(Mf ** 4 - eta ** 4) / (Mc ** 4 - eta ** 4) / self.c_p
        return dfdsigma, df_depsvp


if __name__ == '__main__':
    # baseline simulation
    # csuh = CSUH(p0=1e6, e0=0.7)
    # csuh.forward()
    # csuh =CSUH(p0=1e5, e0=0.833)
    # csuh.forward()

    # comparison of the void ratio
    # for e0 in np.linspace(0.7, 0.935, 6):
    csuh = CSUH(Z=None, p0=1e5, e0=0.85,
             e_c0=0.934, lambdaa=0.135, kappa=0.04,
             poisson=0.3, N=1.973, chi=0.0, m=1.8, verboseFlag=False)
    csuh.forward()

    # comparison of the initial pressure
    # for p0 in [0.1 * 1e6, 1.0 * 1e6, 2.0 * 1e6, 3.0 * 1e6]:
    #     csuh = CSUH(p0=p0)
    #     csuh.forward()
