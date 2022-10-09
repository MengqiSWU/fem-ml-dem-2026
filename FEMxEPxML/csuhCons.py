import multiprocessing
import numpy as np
from FEMxEPxML.constitutive import ConstitutiveMask, constitutiveSingle
from FEMxEPxML.utils_constitutive import tensor2_tensor3,tensor2d_to_3d_single, returnedDatasDecode, \
    get_elasticMatrix, getVolStrain, get_M, getP, getQ, get_dpdsig_dqdsigma
from utilSelf.general import echo, mapMask, get_dic_from_string

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


class csuhConstitutive(ConstitutiveMask):
    def __init__(self, explicitFlag, numg, pool: multiprocessing.Pool, save_path, rho,
                 # parameters inversed from the footing-dem simulations
                 # kappa:5.212e-02 	 lambdaa:1.488e-01 	 N:1.791e+00 	 Z:9.759e-01 	 ocr:3.599e+01 	 theta_degree:2.359e+01
                 kappa=5.212e-02, lambdaa=1.488e-01, N=1.791e+00, Z=9.759e-01, ocr=3.599e+01, m=1.8, theta_degree=2.359e+01,
                 p0=1e5, nu=0.2,

                 # parameters to generate datasets familiar with the datasets in vonmises simulation
                 # kappa=0.1, lambdaa=0.1689, N=2.021, Z=0.9358, ocr=120., m=1.8, theta_degree=30,

                 # original parameters
                 # p0=1e5, ocr=120., theta_degree=30.,
                 # lambdaa=0.135, kappa=0.04,
                 # nu=0.3, N=1.973, m=1.8, Z=0.933938655,
                 verboseFlag=False, ndim=2, save_flag=False):
        csuhs = [
            csuh_single(p0=p0, ocr=ocr, Z=Z, theta_degree=theta_degree, lambdaa=lambdaa, kappa=kappa,
                        nu=nu, N=N, m=m, verboseFlag=verboseFlag,
                        explicitFlag=explicitFlag, ndim=ndim) for _ in range(numg)]
        ConstitutiveMask.__init__(
            self, name='csuh', save_path=save_path, p0=p0, ndim=ndim, cons=csuhs, pool=pool,
            explicitFlag=explicitFlag, numg=numg, rho=rho, save_flag=save_flag)


class csuh_single(constitutiveSingle):
    def __init__(self,
                 kappa, lambdaa, N, Z, ocr,  theta_degree=None, M=1.25,
                 nu=0.2, m=1.8, p0=1e5, ndim=2,
                 verboseFlag=False, explicitFlag=True):
        constitutiveSingle.__init__(self, p0=p0, ndim=ndim)
        # -----------------Parameters (fundamental)------------------
        self.M = get_M(theta_degree=theta_degree) if theta_degree else M   # ratio at critical state
        self.lambdaa = lambdaa
        self.kappa = kappa
        self.nu = nu
        self.N = N  # location in normal consolidation on e-lnp space, where p = 1kPa
        self.Z = Z  # location in normal consolidation on e-lnp space, where p = 1kPa
        self.m = m
        self.p0 = p0
        self.ps = np.exp((self.N-self.Z)/self.lambdaa)-1.0

        self.ocr = ocr
        self.R = 1./self.ocr
        self.p_bar = self.p0*self.ocr
        self.e_eta = self.get_e_eta(eta=0, p=self.p0)
        self.e0 = self.e_eta-(self.lambdaa-self.kappa) * \
                  np.log((self.p0*self.ocr+self.ps*1e3)/(self.p0+self.ps*1e3))
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
        self.px0 = self.p0*(1+(self.eta/self.M)**2)  # CAUTION: the px0 is in unit of Pa
        self.M_c = self.getM_c(xi=self.xi)
        self.M_f = self.getM_f(xi=self.xi)

        # -----------------Reference yield surface (calculated)------------------
        self.epsvp = 0.

        # -----------------Calculation parameters-----------------
        self.yieldTolerance = 0.1

        # -----------------Current yield surface (calculated)------------------
        self.H = 0.
        self.yieldValue = self.yieldFunction(q=self.q, p=self.p, H=0.)

        # ==================== initialization ended ======================
        self.mini_p = 1e1
        self.verboseFlag = verboseFlag
        self.explicitFlag = explicitFlag
        # echo('e0: %.3e xi: %.3e ocr: %.3e lam: %.3e G: %.3e lambdaa: %.3e kappa: %.3e m: %.3e Z: %.3e N: %.3e' %
        #      (self.e0, self.xi, self.ocr, self.lam, self.G, self.lambdaa, self.kappa, self.m, self.Z, self.N))

    def solver(self, deps):
        deps_norm = np.linalg.norm(deps)
        step_num = int(deps_norm / 0.0002)+1
        if step_num < 1:
            step_num = 1
        # elif self.explicitFlag:
        #     if step_num > 100:
        #         echo('The strain is too large (deps_norm = %.2e)' % deps_norm)
        #         raise()
        # else:
        #     if step_num > 1000:
        #         echo('The strain is too large (deps_norm = %.2e)' % deps_norm)
        #         raise()
        step_size = 1./step_num
        remain, split_num = 1.0, 0
        scece_safe = self.get_current_scene()
        while remain > 0. and split_num < 10:
            try:
                if self.explicitFlag:
                    sig, scene = self.solver_1(deps=deps*step_size)
                else:
                    sig, D, scene = self.solver_1(deps=deps*step_size)
                remain -= step_size
                self.update(*scene)
            except:
                break
        if remain == 1.0:
            sig, scene = scece_safe[0], scece_safe
            if not self.explicitFlag:
                D = self.D*0.1
        self.update(*scece_safe)
        if self.explicitFlag:
            return sig, scene
        else:
            return sig, D, scene

    def solver_1(self, deps):
        e = self.e - (self.e + 1.) * getVolStrain(deps)
        sig_trial = self.sig + np.einsum('ijkl, kl->ij', self.D, deps)
        p, q = getP(sig_trial), getQ(sig_trial)
        eta = q/p
        # ------ failure ckeck() ------
        if p < self.mini_p:  # failed
            if self.verboseFlag:
                echo('111 Failure with the mean stress %.3e Pa' % p)
            # scences = self.failure_scences(deps=deps, e=e)
            # if self.explicitFlag:
            #     return scences[0], scences
            # else:
            #     return scences[0], self.D, scences
            raise

        yieldValue = self.yieldFunction(q=q, p=p, H=self.H)
        if yieldValue < 0:  # Elastic
            e_eta = self.get_e_eta(eta=eta, p=p)
            xi = e_eta - e
            scence = [sig_trial, self.eps + deps, self.eps_abs+np.abs(deps),
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
            Mc_last, Mf_last = self.getM_c(xi=xi_last), self.getM_f(xi=xi_last)
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
            Mc_last, Mf_last = self.M_c, self.M_f
            eps_last = self.eps
        return self.plasticReturnMapping(
            deps=deps_left, sig_last=sig_last, D_last=D_last, e_last=e_last, p_last=p_last,
            q_last=q_last, yieldValue_last=yieldValue_last,
            xi_last=xi_last, Mc_last=Mc_last, Mf_last=Mf_last, eps_last=eps_last)

    def plasticReturnMapping(self, deps, sig_last, D_last,
                             e_last, p_last, q_last, yieldValue_last,
                             xi_last, Mc_last, Mf_last, eps_last):

        e = e_last - (1. + e_last) * getVolStrain(deps)
        eta_last = q_last / p_last
        dgdsig = self.dgdsig(Mc=Mc_last, sigma=sig_last, p=p_last, eta=eta_last)
        dfdsig = self.dfdsig(sigma=sig_last, p=p_last, q=q_last, M_f=Mf_last)
        df_depsvp = - (Mf_last ** 4 - eta_last ** 4) / (Mc_last ** 4 - eta_last ** 4) / self.c_p
        temp = np.einsum('ij, ijkl, kl->', dfdsig, D_last, dgdsig)-\
               df_depsvp * np.trace(dgdsig)
        dlam = (np.einsum('ij, ijkl, kl->', dfdsig, D_last, deps)
                + yieldValue_last if yieldValue_last < 10. else 0.) / temp
        deps_p = dlam * dgdsig
        deps_vp = np.trace(deps_p)
        sig = sig_last + np.einsum('ijkl, kl->ij', D_last, deps - deps_p)
        p, q = getP(sig), getQ(sig)

        if p < self.mini_p:  # failed
            if self.verboseFlag:
                echo('222 Failure with the mean stress %.3e Pa' % (p))
            # scences = self.failure_scences(deps=deps, e=e, e_last=e_last, sig_last=sig_last)
            # if self.explicitFlag:
            #     return scences[0], scences
            # else:
            #     return scences[0], self.D, scences
            raise

        eta = q / p
        xi = self.get_e_eta(eta=eta, p=p) - e
        epsvp = self.epsvp + deps_vp
        H = self.H + self.get_dH(Mf=Mf_last, Mc=Mc_last, eta=eta_last, deps_vp=deps_vp)
        yieldValue = self.yieldFunction(q=q, p=p, H=H)
        if yieldValue > self.yieldTolerance:
            if self.verboseFlag:
                echo('Yield value is %.2e > tolerance (%.2e)' % (yieldValue, self.yieldTolerance))
            raise
        scence = [sig, eps_last + deps, self.eps_abs+np.abs(deps),
                  [yieldValue, p, q, e, xi, epsvp, H]]
        if self.explicitFlag:
            return sig, scence
        else:
            D_ep = D_last - np.einsum('ijmn, mn, st, stkl->ijkl', D_last, dgdsig, dfdsig, D_last) / \
                   temp
            return sig, D_ep, scence

    def get_dg_dsigma(self, Mc, eta, p, sigma):
        term1 = (Mc ** 2. - eta ** 2.) * np.eye(3) / 3.
        term2 = 3. * (sigma - p * np.eye(3)) / p
        v = (term1 + term2) / p / (Mc ** 2. + eta ** 2.)
        return v

    def dfdsig(self, sigma, p, q, M_f):
        '''
            sigma = np.random.random(size=[3, 3])
            sigma = 0.5*(sigma+sigma.T)
        '''
        eta = q / p
        # dfdp_up = self.M**2.-eta**2.
        # dfdp_low = self.M**2.*(p+self.ps*1e3)+q**2./p
        dfdp_up = M_f**2.-eta**2.
        dfdp_low = M_f**2.*(p+self.ps*1e3)+q**2./p
        dfdq_up = 2. * eta
        dfdp = dfdp_up / dfdp_low
        dfdq = dfdq_up / dfdp_low
        '''
         NOTE: calculation in sympy is very slow!!! So the Sympy package is not used in this version
        '''
        dp_dsigma, dq_dsigma = get_dpdsig_dqdsigma(sigma=sigma)
        dfdsigma = dfdp*dp_dsigma+dfdq*dq_dsigma
        return dfdsigma

    def dgdsig(self, Mc, eta, p, sigma):  # checked
        term1 = (Mc ** 2. - eta ** 2.) * np.eye(3) / 3.
        term2 = 3. * (sigma - p * np.eye(3)) / p
        v = (term1 + term2) / p / (Mc ** 2. + eta ** 2.)
        return v

    def get_dH(self, Mf, Mc, eta, deps_vp):
        return (Mf ** 4 - eta ** 4) / (Mc ** 4 - eta ** 4) * deps_vp

    def get_lam_G(self, p, e):
        K = (1. + e)*(p+self.ps*1e3) / self.kappa
        G = 3. * (1 - 2 * self.nu) * K / 2. / (1. + self.nu)
        lam = K - 2. / 3. * G
        return lam, G

    def get_e_eta(self, eta, p):
        p = p / 1e3
        ''' UH model '''
        # e_eta = self.N-self.lambdaa*np.log(p)-(self.lambdaa-self.kappa)*np.log(1.+eta**2./self.M**2.)
        ''' CSUH model'''
        e_eta = self.Z - self.lambdaa * np.log((p + self.ps) / (1. + self.ps)) -\
                (self.lambdaa - self.kappa) * \
                np.log(((1.0+eta ** 2/self.M ** 2) * p + self.ps) / (p + self.ps))
        return e_eta

    def getM_c(self, xi):
        ''' Eq. (33) '''
        if abs(xi) > 0.5:
            return self.M
        return self.M * np.exp(-self.m * xi)

    def getM_f(self, xi):
        # R = np.exp(-xi/(self.lambdaa-self.kappa))
        # k = self.M**2/(12.*(3.-self.M))
        # temp = k/R
        # mf = 6.*(np.sqrt(temp*(1.+temp))-temp)
        # if abs(xi) > 0.5:
        #     print()
        if abs(xi) > 0.5:
            return self.M
        else:
            mf = 6. / (np.sqrt(12. * (3. - self.M) / self.M ** 2 *
                               np.exp(-xi / (self.lambdaa - self.kappa)) + 1.) + 1.)
            return mf

    def yieldFunction(self, q, p, H):
        # p /= 1e3
        # q /= 1e3
        # f = np.log(((1.+q**2/self.M**2./p**2.)*p+self.ps*1e3)/(self.ps*1e3+px0))-H/self.c_p  # function in the paper
        f = np.log(((1.+q**2/self.M_f**2./p**2.)*p+self.ps*1e3)/(self.ps*1e3+self.px0))-H/self.c_p
        return f

    def transformSplit(self, deps, D):
        rmin, rmax, rmid = 0., 1., 0.5
        sig = self.sig + np.einsum('ijkl, kl->ij', D, deps * rmid)
        p = getP(sigma=sig)
        q = getQ(sigma=sig)
        yieldValue = self.yieldFunction(q=q, p=p, H=self.H)
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
            yieldValue = self.yieldFunction(q=q, p=p, H=self.H)
            split_num += 1
            if split_num > 20:
                echo('Split num:\t%d yieldValue:\t %.3e last_yieldValue:\t %.3e rmid:\t %.3e' %
                     (split_num, yieldValue, self.yieldValue, rmid))
                rmid, sig = 0., self.sig
                return rmid, sig, self.yieldValue
                # break
        return rmid, sig, yieldValue

    def failure_scences(self, deps, e, e_last=None, sig_last=None):
        '''
            scence:
                        sig_geo, eps, [ yieldValue, p, q, e, xi, epsvp, H]
                           0      1       0         1  2  3   4    5    6
        '''
        deps_vp = np.trace(deps)
        # e_eta = self.get_e_eta(eta=self.q / self.p, p=self.p)
        # xi = e_eta - e
        # H = self.H + self.get_dH(Mf=self.M_f, Mc=self.M_c, eta=self.q / self.p, deps_vp=deps_vp)
        # xi, H = self.xi, self.H
        # sig = self.sig + np.einsum('ijkl, kl->ij', self.D*0.1, deps)
        if e_last:
            scences = [sig_last, self.eps + deps, self.eps_abs + np.abs(deps),
                       [self.yieldValue, getP(sig_last), getQ(sig_last), e_last, self.xi, self.epsvp + deps_vp, self.H]]
        else:
            scences = [self.sig, self.eps + deps, self.eps_abs+np.abs(deps),
                   [self.yieldValue, self.p, self.q, self.e, self.xi, self.epsvp + deps_vp, self.H]]
        return scences

    def get_current_scene(self):
        return [self.sig, self.eps, self.eps_abs,
                   [self.yieldValue, self.p, self.q, self.e, self.xi, self.epsvp, self.H]]

    def update_internal(self, yieldValue, p, q, e, xi, epsvp, H):
        'yieldValue, p, q, e, xi, epsvp, H'
        self.yieldValue = yieldValue
        self.p, self.q, self.e, self.xi, self.epsvp, self.H = p, q, e, xi, epsvp, H
        self.M_c, self.M_f = self.getM_c(self.xi), self.getM_f(self.xi)
        self.lam, self.G = self.get_lam_G(p=self.p, e=self.e)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)

    def prediction(self, deps_s):
        sig_pre = []
        # sig_pre.append(-self.sig[:2, :2])
        for num, i in enumerate(deps_s):
            sig_temp, scenes_temp = self.solver(deps=-tensor2d_to_3d_single(tensor2d=i))
            # sig_pre = torch.cat((sig_pre, -sig_temp[:2, :2]), dim=0)
            sig_pre.append(-sig_temp[:2, :2])
            self.update(*scenes_temp)
        prediction = np.array(sig_pre)
        return prediction

    def return2initial(self):
        constitutiveSingle.__init__(self, p0=self.p0, ndim=self.ndim)

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
        self.px0 = self.p0 * (1 + (self.eta / self.M) ** 2)  # CAUTION: the px0 is in unit of kPa
        self.M_c = self.getM_c(xi=self.xi)
        self.M_f = self.getM_f(xi=self.xi)

        # -----------------Reference yield surface (calculated)------------------
        self.epsvp = 0.
        # -----------------Current yield surface (calculated)------------------
        self.H = 0.
        self.yieldValue = self.yieldFunction(q=self.q, p=self.p, H=0.)


if __name__ == '__main__':
    stress = []
    xi = []
    object_axial_strain = 0.1
    load_step = 1000
    ocr = 2
    # deps_axial = object_axial_strain / load_step
    stress_total, xi_total = [], []
    ocr_list = [0.2]
    # ocr_list = [0.2, 0.5, 1.0, 10]
    for ocr in ocr_list:
        echo('loading ocr: $%.2f' % ocr)
        stress = []
        xi = []
        axialStrainArray = np.linspace(0., object_axial_strain, load_step)
        csuh_single_object = csuh_single(ocr=ocr, explicitFlag=False, ndim=3,
    **get_dic_from_string(
        s='kappa:1.906e-01 	 lambdaa:2.142e-01 	 N:1.931e+00 	 Z:2.743e-01 	 theta_degree:1.329e+01'
        # s='theta_degree:30. \t lambdaa:0.135 \t kappa:0.04 \t N:1.973 \t Z:0.933938655'
    ))
        for i in range(1, load_step):
            deps_axial =axialStrainArray[i]- axialStrainArray[i-1]
            deps = np.diag([-0.6 * deps_axial, -0.5 * deps_axial, deps_axial])
            sig_trial, D, scence = csuh_single_object.solver(deps=deps)
            csuh_single_object.update(*scence)
            print('%d yield_value: %.3e p: %.3e q: %.3e xi: %.3e eps_v:%.3e' %
                  (i + 1, scence[3][0], getP(sig_trial), getQ(sig_trial), scence[3][4], scence[3][5]))
            stress.append(sig_trial)
            xi.append(scence[3][4])
        stress_total.append(stress)
        xi_total.append(xi)

    import matplotlib.pyplot as plt
    for i in range(len(stress_total)):
        p = np.array([getP(i) for i in stress_total[i]])
        q = np.array([getQ(i) for i in stress_total[i]])
        plt.plot(p / 1e6, q / 1e6, label=ocr_list[i])
    plt.axis('equal')
    plt.xlabel('p MPa')
    plt.ylabel('q MPa')
    plt.title('q-p')
    plt.tight_layout()
    plt.legend()
    plt.show()
    for i in range(len(stress_total)):
        plt.plot(xi_total[i], label=ocr_list[i])
    plt.axis('equal')
    plt.xlabel('step')
    plt.ylabel(r'$\xi$')
    plt.title(r'$\xi$')
    plt.tight_layout()
    plt.legend()
    plt.show()
    #
