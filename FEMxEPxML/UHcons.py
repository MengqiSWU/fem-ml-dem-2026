import multiprocessing
import numpy as np
from FEMxEPxML.constitutive import ConstitutiveMask
from FEMxEPxML.utils_constitutive import tensor2_tensor3, returnedDatasDecode, \
    get_elasticMatrix, getVolStrain, getP, getQ, get_M
from utilSelf.general import echo, mapMask


class uhConstitutive(ConstitutiveMask):
    def __init__(self, explicitFlag, numg, pool: multiprocessing.Pool,save_path, rho,
                 p0=1e5, e0=1.3513020248916077, theta_degree=30, lambdaa=0.135, kappa=0.01, ocr=1.2, # 0.04
                 poisson=0.3, N=1.973,
                 verboseFlag=False, ndim=3):
        self.cons = [
            uh_single(p0=p0, e0=e0, ocr=ocr, lambdaa=lambdaa, kappa=kappa,
                        poisson=poisson, N=N,  verboseFlag=verboseFlag, theta_degree=theta_degree,
                        explicitFlag=explicitFlag) for _ in range(numg)]
        ConstitutiveMask.__init__(self, save_path=save_path, p0=p0, cons=self.cons, pool=pool, ndim=ndim, explicitFlag=explicitFlag,
                                  numg=numg, rho=rho)


class uh_single:
    def __init__(self, p0, lambdaa, kappa, # 0.04
                 poisson, N, ocr, e0=None, theta_degree=30,
                 verboseFlag=True, explicitFlag=False):
        # -----------------Parameters (fundamental)------------------
        self.M = get_M(theta_degree=theta_degree)  # ratio at critical state
        self.lambdaa = lambdaa
        self.kappa = kappa
        self.poisson = poisson
        self.N = N
        self.p0 = p0  # CAUTION: in unit of 1kPa
        if ocr:
            self.ocr = ocr
            self.R = 1./self.ocr
            self.p_bar = self.p0*self.ocr
            self.e_eta = self.get_e_eta(eta=0, p=self.p0)
            self.e0 = self.N-self.lambdaa*np.log(self.p0/1e3)-(self.lambdaa-self.kappa)*np.log(self.ocr)
            print(self.e0)
            print(self.ocr)
        elif e0:
            self.e0 = e0
            self.e_eta = self.get_e_eta(eta=0., p=self.p0)
            self.xi = self.e_eta-self.e0
            self.ocr = np.exp(self.xi/(self.lambdaa-self.kappa))
            self.p_bar = self.p0*self.ocr
            self.R = 1./self.ocr
        self.c_p = (self.lambdaa - self.kappa) / (1. + self.e0)

        # -----------------States (calculated)------------------
        self.sig = np.eye(3, dtype=float) * self.p0
        self.eps = np.zeros(shape=[3, 3], dtype=float)
        # according to the current stress and void ratio state
        self.q, self.p = 0., self.p0
        self.lam, self.G = self.get_lam_G(self.p, self.e0)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)
        self.e = self.e0
        self.eta = self.q / self.p
        self.e_eta = self.get_e_eta(eta=self.eta, p=self.p)
        self.xi = self.e_eta - self.e
        # self.over_overconsolidation_ratio = np.exp(-self.xi / (self.lambdaa - self.kappa))
        self.px0 = self.p0*(1.+self.eta**2/self.M**2)   # CAUTION: the px0 is in unit of kPa
        self.M_f = self.getM_f(xi=self.xi)

        # -----------------Reference yield surface (calculated)------------------
        self.epsvp = 0.

        # -----------------Calculation parameters-----------------
        self.yieldTolerance = 0.05

        # -----------------Current yield surface (calculated)------------------
        self.H = 0.
        self.yieldValue = self.yieldFunction(q=self.q, p=self.p, H=0., px0=self.px0)

        # ==================== initialization ended ======================
        self.verboseFlag = verboseFlag
        self.explicitFlag = explicitFlag

    def solver(self, deps):
        e = self.e - (self.e + 1.) * getVolStrain(deps)
        sig_trial = self.sig + np.einsum('ijkl, kl->ij', self.D, deps)
        p, q = getP(sig_trial), getQ(sig_trial)
        eta = q/p
        # ------ failure ckeck() ------
        if p < 0:  # failed
            echo('111 Failure with the mean stress %.3e Pa' % (p))
            scences = self.failure_scences(deps=deps, e=e)
            if self.explicitFlag:
                return self.sig, scences
            else:
                return self.sig, np.zeros_like(self.D), scences

        yieldValue = self.yieldFunction(q=q, p=p, H=self.H, px0=self.px0)
        if yieldValue < 0:  # Elastic
            e_eta = self.get_e_eta(eta=eta, p=p)
            xi = e_eta - e
            scence = [sig_trial, self.eps + deps,
                      [yieldValue, p, q, e, xi, self.epsvp, self.H]]
            if self.explicitFlag:
                return sig_trial, scence
            else:
                lam, G =self.get_lam_G(p=p, e=e)
                D = get_elasticMatrix(lam=lam, G=G)
                return sig_trial, D, scence
        elif self.yieldValue < -self.yieldTolerance:
            rmid, sig_last, yieldValue_last = self.transformSplit(deps=deps, D=self.D)
            e_last = self.e - (self.e + 1.) * getVolStrain(deps * rmid)
            p_last, q_last = getP(sigma=sig_last), getQ(sigma=sig_last)
            eta_last = q_last/p_last
            lam_last, G_last = self.get_lam_G(p=p_last, e=e_last)
            D_last = get_elasticMatrix(lam=lam_last, G=G_last)
            xi_last = self.get_e_eta(eta=eta_last, p=p_last) - e_last
            # Mc_last, Mf_last = self.getM_c(xi=xi_last), self.getM_f(xi=xi_last)
            Mf_last =  self.getM_f(xi=xi_last)
            deps_left = deps * (1 - rmid)
            eps_last = self.eps + deps * rmid
        else:
            sig_last = self.sig
            deps_left = deps
            D_last = self.D
            e_last = self.e
            p_last, q_last = self.p, self.q
            yieldValue_last = self.yieldValue
            xi_last = self.xi
            # Mc_last, Mf_last = self.M_c, self.M_f
            Mf_last =  self.M_f
            eps_last = self.eps
        return self.plasticReturnMapping(
            deps=deps_left, sig_last=sig_last, D_last=D_last, e_last=e_last, p_last=p_last,
            q_last=q_last, yieldValue_last=yieldValue_last,
            xi_last=xi_last, Mf_last=Mf_last, eps_last=eps_last)

    def plasticReturnMapping(self, deps, sig_last, D_last,
                             e_last, p_last, q_last, yieldValue_last,
                             xi_last, Mf_last, eps_last):
        e = e_last - (1. + e_last) * getVolStrain(deps)
        eta_last = q_last / p_last
        dfdsig = dgdsig = self.dgdsig(sigma=sig_last, p=p_last, eta=eta_last)
        df_depsvp = self.get_df_depsvp(Mf=Mf_last, eta=eta_last)
        temp = np.einsum('ij, ijkl, kl->', dfdsig, D_last, dgdsig)-\
               df_depsvp * np.trace(dgdsig)
        dlam = (np.einsum('ij, ijkl, kl->', dfdsig, D_last, deps)
                + yieldValue_last if yieldValue_last < 1e5 else 0.) / temp

        deps_p = dlam * dgdsig
        deps_vp = np.trace(deps_p)
        sig = sig_last + np.einsum('ijkl, kl->ij', D_last, deps - deps_p)
        p, q = getP(sig), getQ(sig)

        if p < 0.:  # failed
            echo('222 Failure with the mean stress %.3e Pa' % (p))
            scences = self.failure_scences(deps=deps, e=e)
            if self.explicitFlag:
                return self.sig, scences
            else:
                return self.sig, np.zeros_like(self.D), scences

        eta = q / p
        xi = self.get_e_eta(eta=eta, p=p) - e
        epsvp = self.epsvp + deps_vp
        H = self.H + self.get_dH(Mf=Mf_last, eta=eta_last, deps_vp=deps_vp)
        yieldValue = self.yieldFunction(q=q, p=p, H=H, px0=self.px0)
        scence = [sig, eps_last + deps,
                  [yieldValue, p, q, e, xi, epsvp, H]]
        if self.explicitFlag:
            return sig, scence
        else:
            D_ep = D_last - np.einsum('ijmn, mn, st, stkl->ijkl', D_last, dgdsig, dfdsig, D_last) / \
                   temp
            return sig, D_ep, scence

    def dfdsig(self, sigma, p, q):
        '''
            sigma = np.random.random(size=[3, 3])
            sigma = 0.5*(sigma+sigma.T)
        '''
        dfdsig = self.dgdsig(eta=q / p, p=p, sigma=sigma)
        return dfdsig

    def dgdsig(self, eta, p, sigma):
        term1 = (self.M ** 2. - eta ** 2.) * np.eye(3) / 3.
        term2 = 3. * (sigma - p * np.eye(3)) / p
        v = (term1 + term2) / p/ (self.M ** 2. + eta ** 2.)
        return v

    def get_dH(self, Mf, eta, deps_vp):
        # return (self.M**4/Mf**4)*(Mf ** 4 - eta ** 4) / (self.M ** 4 - eta ** 4) * deps_vp
        return (Mf ** 4 - eta ** 4) / (self.M ** 4 - eta ** 4) * deps_vp

    def get_df_depsvp(self, Mf, eta):
        # return -(self.M**4/Mf**4)*(Mf ** 4 - eta ** 4) / (self.M ** 4 - eta ** 4)/self.c_p
        return -(Mf ** 4 - eta ** 4) / (self.M ** 4 - eta ** 4)/self.c_p

    def get_lam_G(self, p, e):
        K = (1. + e) * p / self.kappa
        G = 3. * (1 - 2 * self.poisson) * K / 2. / (1. + self.poisson)
        lam = 3.*K*self.poisson/(1+self.poisson)
        return lam, G

    def get_e_eta(self, eta, p):
        e_eta = self.N-self.lambdaa*np.log(p/1e3)-\
                (self.lambdaa-self.kappa)*np.log(1+(eta/self.M)**2)
        return e_eta
    #
    # def getM_c(self, xi):
    #     ''' Eq. (33) '''
    #     return self.M * np.exp(-self.m * xi)

    def getM_f(self, xi):
        k = self.M**2./12./(3.-self.M)
        R = np.exp(-xi/(self.lambdaa-self.kappa))
        # mf = np.sqrt(36*k/R*(1+k/R))-6*k
        temp = k/R
        mf = 6*(np.sqrt(temp*(1+temp))-temp)
        # xi = np.linspace(-0.5, 0.5, 100); import matplotlib.pyplot as plt ; plt.plot(xi, mf); plt.show()
        return mf

    def yieldFunction(self, q, p, H, px0):
        # p /= 1e3
        # q /= 1e3
        temp = q**2/(self.M ** 2 * p ** 2)
        f = np.log(p/self.px0)+np.log(1+temp)-H/self.c_p
        return f

    def transformSplit(self, deps, D):
        rmin, rmax, rmid = 0., 1., 0.5
        sig = self.sig + np.einsum('ijkl, kl->ij', D, deps * rmid)
        p = getP(sigma=sig)
        q = getQ(sigma=sig)
        yieldValue = self.yieldFunction(q=q, p=p, H=self.H, px0=self.px0)
        split_num = 1
        while True:
            if yieldValue > 0.:
                rmax = rmid
            elif yieldValue < -self.yieldTolerance:
                rmin = rmid
            else:
                break
            rmid = .5 * (rmax + rmin)
            sig = self.sig + np.einsum('ijkl, kl->ij', D, deps * rmid)
            p = getP(sigma=sig)
            q = getQ(sigma=sig)
            yieldValue = self.yieldFunction(q=q, p=p, H=self.H, px0=self.px0)
            split_num += 1
            if split_num > 100:
                echo('Split num:\t%d yieldValue:\t %.3e last_yieldValue:\t %.3e rmid:\t %.3e' %
                     (split_num, yieldValue, self.yieldValue, rmid))
                raise RuntimeError
        return rmid, sig, yieldValue

    def failure_scences(self, deps, e):
        '''
            scence:
                        sig_geo, eps, [ yieldValue, p, q, e, xi, epsvp, H]
                           0      1       0         1  2  3   4    5    6
        '''
        deps_vp = np.trace(deps)
        # e_eta = self.get_e_eta(eta=self.q / self.p, p=self.p)
        # xi = e_eta - e
        # H = self.H + self.get_dH(Mf=self.M_f, Mc=self.M_c, eta=self.q / self.p, deps_vp=deps_vp)
        xi, H = self.xi, self.H
        scences = [self.sig, self.eps + deps,
                   [self.yieldValue, self.p, self.q, e, xi, self.epsvp + deps_vp, H]]
        return scences

    def update(self, sig_geo, eps, internal):
        self.sig = sig_geo
        self.eps = eps
        self.update_internal(*internal)

    def update_internal(self, yieldValue, p, q, e, xi, epsvp, H):
        'yieldValue, p, q, e, xi, epsvp, H'
        self.yieldValue = yieldValue
        self.p, self.q, self.e, self.xi, self.epsvp, self.H = p, q, e, xi, epsvp, H
        # self.M_c = self.getM_c(self.xi)
        self.M_f = self.getM_f(self.xi)
        self.lam, self.G = self.get_lam_G(p=self.p, e=self.e)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)


def getM_f(xi, M=1.25, lambdaa=0.135, kappa=0.01):
    k = M**2./12./(3.-M)
    R = np.exp(-xi/(lambdaa-kappa))
    # mf = np.sqrt(36*k/R*(1+k/R))-6*k
    temp = k/R
    mf = 6*(np.sqrt(temp*(1+temp))-temp)
    # xi = np.linspace(-0.5, 0.5, 100); import matplotlib.pyplot as plt ; plt.plot(xi, mf); plt.show()
    return mf


if __name__ == '__main__':
    object_axial_strain = 0.3
    load_step = 200
    stress_total, xi_total = [], []
    ocr_list = [0.2, 0.5, 1.0, 2.0]
    for ocr in ocr_list:
        stress = []
        xi = []
        # deps_axial = object_axial_strain / load_step
        axialStrainArray = np.linspace(0., object_axial_strain, 1000)
        uh_single_object = uh_single(p0=1e6, ocr=ocr, theta_degree=30, lambdaa=0.135, kappa=0.01,
                                         poisson=0.3, N=1.973,  explicitFlag=False)
        for i in range(1, load_step):
            deps_axial =axialStrainArray[i]- axialStrainArray[i-1]
            deps = np.diag([-0.5 * deps_axial, -0.5 * deps_axial, deps_axial])
            sig_trial, D, scence = uh_single_object.solver(deps=deps)
            uh_single_object.update(*scence)
            print('%d p: %.3e q: %.3e xi: %.3e eps_v:%.3e' %
                  (i + 1, getP(sig_trial), getQ(sig_trial), scence[2][4], scence[2][5]))
            stress.append(sig_trial)
            xi.append(scence[2][4])
        stress_total.append(np.array(stress))
        xi_total.append(xi)

    import matplotlib.pyplot as plt

    for i in range(len(stress_total)):
        p = np.array([getP(i) for i in stress_total[i]])
        q = np.array([getQ(i) for i in stress_total[i]])
        plt.plot(p/1e6, q/1e6, label=ocr_list[i])
    plt.axis('equal')
    plt.title('q-p')
    plt.tight_layout()
    plt.legend()
    plt.show()
    #
    for i in range(len(stress_total)):
        p = np.array([getP(i) for i in stress_total[i]])
        q = np.array([getQ(i) for i in stress_total[i]])
        plt.plot(q / p, label=ocr_list[i])
    # plt.title('q-p')
    plt.legend()
    plt.title(r'$\eta$')
    plt.tight_layout()
    plt.show()
    #
    for i in range(len(stress_total)):
        p = np.array([getP(i) for i in stress_total[i]])
        q = np.array([getQ(i) for i in stress_total[i]])
        plt.plot(q /1e3, label=ocr_list[i])
    # plt.title('q-p')
    plt.legend()
    plt.title(r'$q (kPa)$')
    plt.tight_layout()
    plt.show()
    #
    for i in range(len(stress_total)):
        plt.plot(xi_total[i], label=ocr_list[i])
    # plt.title('q-p')
    plt.legend()
    plt.title(r'$\xi$')
    plt.tight_layout()
    plt.show()

    for i in range(len(stress_total)):
        mf = [getM_f(xi = xi_total[i][j]) for j in range(len(xi_total[i]))]
        plt.plot(mf, label=ocr_list[i])
    # plt.title('q-p')
    plt.legend()
    plt.title(r'$M_f$')
    plt.tight_layout()
    plt.show()
