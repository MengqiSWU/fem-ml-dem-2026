import multiprocessing
import numpy as np
import torch
import copy

from FEMxEPxML.constitutive import ConstitutiveMask, constitutiveSingle
from FEMxEPxML import utils_constitutive
from FEMxEPxML.utils_constitutive_ml import get_elasticMatrix, getVolStrain, get_M, getP, getQ, get_dpdsig_dqdsigma
from FEMxEPxML.utils_constitutive import tensor2d_to_3d_single
from utilSelf.general import echo

'''
    Author: Shaoheng Gun
            Wuhan University & Swansea University
    Email: shaohengguan@gmail.com

    Critical state theory involved Unified Hardening model for both sand and clay
    
    Complemented in torch version for parameters inversion

        in tensor notion

        Reference:
        [1] Yao, Y. P., Liu, L., Luo, T., Tian, Y., & Zhang, J. M. (2019).
            Unified hardening (UH) model for clays and sands. Computers and Geotechnics,
            110(March), 326–343. https://doi.org/10.1016/j.compgeo.2019.02.024

'''


class mlcsuhConstitutive(ConstitutiveMask):
    def __init__(self, explicitFlag, numg, pool: multiprocessing.Pool, save_path, rho,
                 p0=1e5, ocr=120., theta_degree=30.,
                 lambdaa=0.135, kappa=0.04,
                 poisson=0.3, N=1.973, m=1.8, Z=0.933938655,
                 verboseFlag=False, ndim=2, save_flag=False):
        self.csuhs = [
            mlcsuh_single(p0=p0, ocr=ocr, Z=Z, theta_degree=theta_degree, lambdaa=lambdaa, kappa=kappa,
                        poisson=poisson, N=N, m=m, verboseFlag=verboseFlag,
                        explicitFlag=explicitFlag, ndim=ndim) for _ in range(numg)]
        ConstitutiveMask.__init__(
            self, name='csuh', save_path=save_path, p0=p0, ndim=ndim,cons=self.csuhs, pool=pool,
            explicitFlag=explicitFlag, numg=numg, rho=rho, save_flag=save_flag)


class mlcsuh_single(constitutiveSingle):
    def __init__(self,  p0, ndim,
                 ocr=120., theta_degree=30.,
                 lambdaa=0.135, kappa=0.04,
                 poisson=0.3, N=1.973, m=1.8, Z=0.933938655,
                 verboseFlag=False, explicitFlag=True):
        constitutiveSingle.__init__(self, p0=p0, ndim=ndim)
        # -----------------Parameters (fundamental)------------------
        self.theta_degree = torch.tensor(data=theta_degree, dtype=torch.float32, requires_grad=True)
        self.M = get_M(theta_degree=self.theta_degree)  # ratio at critical state
        self.lambdaa = torch.tensor(lambdaa, dtype=torch.float32, requires_grad=True)
        self.kappa = torch.tensor(kappa, dtype=torch.float32, requires_grad=True)
        self.poisson = torch.tensor(poisson, dtype=torch.float32, requires_grad=True)
        # location in normal consolidation on e-lnp space, where p = 1kPa
        self.N = torch.tensor(N, dtype=torch.float32, requires_grad=True)
        # location in normal consolidation on e-lnp space, where p = 1kPa
        self.Z = torch.tensor(Z, dtype=torch.float32, requires_grad=True)
        self.m = torch.tensor(m, dtype=torch.float32, requires_grad=True)

        self.p0 = torch.tensor(p0, dtype=torch.float32, requires_grad=False)
        self.ps = torch.exp((self.N-self.Z)/self.lambdaa)-1.0
        self.ocr = torch.tensor(ocr, dtype=torch.float32, requires_grad=True)
        self.R = 1./self.ocr
        self.e_eta = self.get_e_eta(eta=0, p=self.p0)
        self.e0 = self.e_eta-(self.lambdaa-self.kappa) * \
                  torch.log((self.p0*self.ocr+self.ps*1e3)/(self.p0+self.ps*1e3))
        self.c_p = (self.lambdaa - self.kappa) / (1. + self.e0)

        # -----------------States (calculated)------------------
        self.sig = torch.tensor(data=np.eye(3), dtype=torch.float32) * self.p0
        self.eps = torch.zeros(size=[3, 3], dtype=torch.float32)
        # according to the current stress and void ratio state
        self.q, self.p = torch.tensor(data=0., dtype=torch.float32, requires_grad=False), self.p0
        self.lam, self.G = self.get_lam_G(self.p, self.e0)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)

        self.e = self.e0
        self.eta = self.q / self.p
        self.e_eta = self.get_e_eta(eta=self.eta, p=self.p)
        self.xi = self.e_eta - self.e
        # self.over_overconsolidation_ratio = np.exp(-self.xi / (self.lambdaa - self.kappa))
        self.px0 = self.p0*(1+(self.eta/self.M)**2) # CAUTION: the px0 is in unit of kPa
        self.M_c = self.getM_c(xi=self.xi)
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
        deps = torch.tensor(deps, dtype=torch.float32)
        e = self.e - (self.e + 1.) * getVolStrain(deps)
        sig_trial = self.sig + torch.einsum('ijkl, kl->ij', self.D, deps)
        p, q = getP(sig_trial), getQ(sig_trial)
        eta = q/p
        # ------ failure ckeck() ------
        if p < 0:  # failed
            if self.verboseFlag:
                echo('111 Failure with the mean stress %.3e Pa' % (p))
            scences = self.failure_scences(deps=deps, e=e)
            if self.explicitFlag:
                return self.sig, scences
            else:
                return self.sig, torch.zeros_like(self.D), scences

        yieldValue = self.yieldFunction(q=q, p=p, H=self.H, px0=self.px0)
        if yieldValue < 0:  # Elastic
            e_eta = self.get_e_eta(eta=eta, p=p)
            xi = e_eta - e
            scence = [sig_trial, self.eps + deps, self.eps_abs+np.abs(deps.numpy()),
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
        dfdsig = self.dfdsig(sigma=sig_last, p=p_last, q=q_last)
        if Mc_last == eta_last:
            df_depsvp = - (Mf_last ** 4 - eta_last ** 4) /torch.tensor(1e-8)/ self.c_p
        else:
            df_depsvp = - (Mf_last ** 4 - eta_last ** 4) / (Mc_last ** 4 - eta_last ** 4) / self.c_p
        temp = torch.einsum('ij, ijkl, kl->', dfdsig, D_last, dgdsig)-\
               df_depsvp * torch.trace(dgdsig)
        dlam = (torch.einsum('ij, ijkl, kl->', dfdsig, D_last, deps)
                + yieldValue_last if yieldValue_last < 1e5 else 0.) / temp
        deps_p = dlam * dgdsig
        deps_vp = torch.trace(deps_p)
        sig = sig_last + torch.einsum('ijkl, kl->ij', D_last, deps - deps_p)
        p, q = getP(sig), getQ(sig)

        if p < 0.:  # failed
            if self.verboseFlag:
                echo('222 Failure with the mean stress %.3e Pa' % (p))
            scences = self.failure_scences(deps=deps, e=e)
            if self.explicitFlag:
                return self.sig, scences
            else:
                return self.sig, torch.zeros_like(self.D), scences

        eta = q / p
        xi = self.get_e_eta(eta=eta, p=p) - e
        epsvp = self.epsvp + deps_vp
        H = self.H + self.get_dH(Mf=Mf_last, Mc=Mc_last, eta=eta_last, deps_vp=deps_vp)
        yieldValue = self.yieldFunction(q=q, p=p, H=H, px0=self.px0)
        scence = [sig, eps_last + deps, self.eps_abs+np.abs(deps.numpy()),
                  [yieldValue, p, q, e, xi, epsvp, H]]
        if self.explicitFlag:
            return sig, scence
        else:
            D_ep = D_last - torch.einsum('ijmn, mn, st, stkl->ijkl', D_last, dgdsig, dfdsig, D_last) / \
                   temp
            return sig, D_ep, scence

    def get_dg_dsigma(self, Mc, eta, p, sigma):
        term1 = (Mc ** 2. - eta ** 2.) * torch.eye(3) / 3.
        term2 = 3. * (sigma - p * torch.eye(3)) / p
        v = (term1 + term2) / p/ (Mc ** 2. + eta ** 2.)
        return v

    def dfdsig(self, sigma, p, q):
        '''
            sigma = torch.random.random(size=[3, 3])
            sigma = 0.5*(sigma+sigma.T)
        '''
        eta = q / p
        dfdp_up = self.M**2.-eta**2.
        dfdp_low = self.M**2.*(p+self.ps*1e3)+q**2./p
        dfdq_up = 2. * eta
        dfdp = dfdp_up / dfdp_low
        dfdq = dfdq_up / dfdp_low
        '''
         NOTE: calculation in sympy is very slow!!! So the Sympy package is not used in this version
        '''
        dp_dsigma, dq_dsigma = get_dpdsig_dqdsigma(sigma=sigma)
        dfdsigma = dfdp*dp_dsigma+dfdq*dq_dsigma
        return dfdsigma

    def dgdsig(self, Mc, eta, p, sigma): # checked
        term1 = (Mc ** 2. - eta ** 2.) * torch.eye(3) / 3.
        term2 = 3. * (sigma - p * torch.eye(3)) / p
        v = (term1 + term2) / p/ (Mc ** 2. + eta ** 2.)
        return v

    def get_dH(self, Mf, Mc, eta, deps_vp):
        if Mc == eta:
            return (Mf ** 4 - eta ** 4)/torch.tensor(1e-8)* deps_vp
        else:
            return (Mf ** 4 - eta ** 4) / (Mc ** 4 - eta ** 4) * deps_vp

    def get_lam_G(self, p:torch.Tensor, e: torch.Tensor):
        K = (1. + e)*(p+self.ps*1e3) / self.kappa
        G = 3. * (1 - 2 * self.poisson) * K / 2. / (1. + self.poisson)
        lam = K - 2. / 3. * G
        return lam, G

    def get_e_eta(self, eta, p):
        p = p / 1e3
        ''' UH model '''
        # e_eta = self.N-self.lambdaa*np.log(p)-(self.lambdaa-self.kappa)*np.log(1.+eta**2./self.M**2.)
        ''' CSUH model'''
        e_eta = self.Z - self.lambdaa * torch.log((p + self.ps) / (1. + self.ps)) -\
                (self.lambdaa - self.kappa) * \
                torch.log(((1.0+eta ** 2/self.M ** 2) * p + self.ps) / (p + self.ps))
        return e_eta

    def getM_c(self, xi):
        ''' Eq. (33) '''
        return self.M * torch.exp(-self.m * xi)

    def getM_f(self, xi):
        # R = np.exp(-xi/(self.lambdaa-self.kappa))
        # k = self.M**2/(12.*(3.-self.M))
        # temp = k/R
        # mf = 6.*(np.sqrt(temp*(1.+temp))-temp)
        mf = 6. / (torch.sqrt(12. * (3. - self.M) / self.M ** 2 *
                           torch.exp(-xi / (self.lambdaa - self.kappa)) + 1.) + 1.)
        # mf = np.sqrt(36.*temp*(1.+temp))-6.*temp
        # ''' Eq. (12) '''
        # mf = 6. / (np.sqrt(12. * (3. - self.M) / self.M ** 2 *
        #                    np.exp(-xi / (self.lambdaa - self.kappa)) + 1.) + 1.)
        return mf

    def yieldFunction(self, q, p, H, px0):
        # p /= 1e3
        # q /= 1e3
        # temp = self.M ** 2 * p ** 2 - self.chi * q ** 2
        # if temp < 0.:
        #     f = 1e32
        #     return f
        # f = np.log((1. + (1 + self.chi) * q ** 2. / temp) * p + self.ps) - \
        #     np.log(px0 + self.ps) - \
        #     H / self.c_p
        # f = np.log(p/1e3/px0)+np.log(1+(q/p)**2/self.M**2)-H / self.c_p
        f = torch.log(((1.+q**2/self.M**2./p**2.)*p+self.ps*1e3)/(self.ps*1e3+px0))-H/self.c_p
        return f

    def transformSplit(self, deps, D):
        rmin, rmax, rmid = 0., 1., 0.5
        sig = self.sig + torch.einsum('ijkl, kl->ij', D, deps * rmid)
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
            sig = self.sig + torch.einsum('ijkl, kl->ij', D, deps * rmid)
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
        deps_vp = torch.trace(deps)
        # e_eta = self.get_e_eta(eta=self.q / self.p, p=self.p)
        # xi = e_eta - e
        # H = self.H + self.get_dH(Mf=self.M_f, Mc=self.M_c, eta=self.q / self.p, deps_vp=deps_vp)
        xi, H = self.xi, self.H
        scences = [self.sig, self.eps + deps, self.eps_abs+np.abs(deps.numpy()),
                   [self.yieldValue, self.p, self.q, e, xi, self.epsvp + deps_vp, H]]
        return scences

    # def update(self, sig, eps, eps_abs, internals):
    #     self.sig = sig.detach() if type(sig) == torch.Tensor else sig
    #     self.eps = eps.detach() if type(eps) == torch.Tensor else eps
    #     self.eps_abs = eps_abs.detach() if type(eps_abs) == torch.Tensor else eps_abs
    #     self.update_internal(*internals)

    def update_internal(self, yieldValue, p, q, e, xi, epsvp, H):
        'yieldValue, p, q, e, xi, epsvp, H'
        # yieldValue, p, q, e, xi, epsvp, H = [i.item() if type(i) == torch.Tensor else i for i in [yieldValue, p, q, e, xi, epsvp, H]]
        self.yieldValue = yieldValue
        self.p, self.q, self.e, self.xi, self.epsvp, self.H = p, q, e, xi, epsvp, H
        self.M_c, self.M_f = self.getM_c(self.xi), self.getM_f(self.xi)
        self.lam, self.G = self.get_lam_G(p=self.p, e=self.e)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)

    def prediction(self, deps_numg):
        sig_numg=[]
        # sig_pre.append(-self.sig[:2, :2])
        for deps in deps_numg:
            sig_pre = []
            for num, i in enumerate(deps):
                sig_temp, scenes_temp = self.solver(deps=-tensor2d_to_3d_single(tensor2d=i))
                # sig_pre = torch.cat((sig_pre, -sig_temp[:2, :2]), dim=0)
                sig_pre.append(-sig_temp[:2, :2])
                self.update(*scenes_temp)
            self.return2initial()
            sig_numg.append(torch.stack(sig_pre))
        prediction = torch.stack(sig_numg)
        return prediction

    def return2initial(self):
        # constitutiveSingle.__init__(self, p0=1e5, ndim=2)
        # -----------------Parameters (fundamental)------------------
        self.M = get_M(theta_degree=self.theta_degree)  # ratio at critical state

        self.ps = torch.exp((self.N - self.Z) / self.lambdaa) - 1.0
        self.R = 1. / self.ocr
        self.e_eta = self.get_e_eta(eta=0, p=self.p0)
        self.e0 = self.e_eta - (self.lambdaa - self.kappa) * \
                  torch.log((self.p0 * self.ocr + self.ps * 1e3) / (self.p0 + self.ps * 1e3))
        self.c_p = (self.lambdaa - self.kappa) / (1. + self.e0)

        self.lam, self.G = self.get_lam_G(p=self.p0, e=self.e0)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)

        # -----------------States (calculated)------------------
        self.sig = torch.tensor(data=np.eye(3), dtype=torch.float32) * self.p0
        self.eps = torch.zeros(size=[3, 3], dtype=torch.float32)
        # according to the current stress and void ratio state
        self.q, self.p = torch.tensor(data=0., dtype=torch.float32, requires_grad=False), self.p0
        self.lam, self.G = self.get_lam_G(self.p, self.e0)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)

        self.e = self.e0
        self.eta = self.q / self.p
        self.xi = self.e_eta - self.e
        # self.over_overconsolidation_ratio = np.exp(-self.xi / (self.lambdaa - self.kappa))
        self.px0 = self.p0 * (1 + (self.eta / self.M) ** 2)  # CAUTION: the px0 is in unit of kPa
        self.M_c = self.getM_c(xi=self.xi)
        self.M_f = self.getM_f(xi=self.xi)

        # -----------------Reference yield surface (calculated)------------------
        self.epsvp = 0.

        # -----------------Current yield surface (calculated)------------------
        self.H = 0.
        self.yieldValue = self.yieldFunction(q=self.q, p=self.p, H=0., px0=self.px0)


if __name__ == '__main__':
    stress = []
    xi = []
    object_axial_strain = 0.2
    load_step = 400
    ocr = 2
    # deps_axial = object_axial_strain / load_step
    stress_total, xi_total = [], []
    # ocr_list = [0.2, 0.5, 1.0, 2.0]
    ocr_list = [1.0]
    for ocr in ocr_list:
        echo('loading ocr: %.2f' % ocr)
        stress = []
        xi = []
        axialStrainArray = np.linspace(0., object_axial_strain, 1000)
        csuh_single_object = mlcsuh_single(p0=1e5, theta_degree=45, ocr=ocr, explicitFlag=False, ndim=3)
        for i in range(1, load_step):
            deps_axial =axialStrainArray[i]- axialStrainArray[i-1]
            deps = np.diag([-0.5 * deps_axial, -0.5 * deps_axial, deps_axial])
            sig_trial, D, scence = csuh_single_object.solver(deps=deps)
            csuh_single_object.update(*scence)
            print('Loading step %d yield_value: %.3e p: %.3e q: %.3e xi: %.3e eps_v:%.3e' %
                  (i + 1, scence[3][0].item(), getP(sig_trial).item(), getQ(sig_trial).item(), scence[3][4].item(), scence[3][5].item()))
            stress.append(sig_trial.detach().numpy())
            xi.append(scence[3][4].item())
        stress_total.append(stress)
        xi_total.append(xi)

    import matplotlib.pyplot as plt
    for i in range(len(stress_total)):
        p = np.array([utils_constitutive.getP(i) for i in stress_total[i]])
        q = np.array([utils_constitutive.getQ(i) for i in stress_total[i]])
        plt.plot(p / 1e6, q / 1e6, label=ocr_list[i])
    plt.axis('equal')
    plt.title('q-p')
    plt.tight_layout()
    plt.legend()
    plt.show()
    #
