import torch
import numpy as np
from FEMxML.torch_net import Net_Simple
from FEMxEPxML.constitutive import ConstitutiveMask, constitutiveSingle
from FEMxEPxML.utils_constitutive_ml import get_elasticMatrix, getVolStrain, get_M, getP, getQ, get_dpdsig_dqdsigma
from FEMxEPxML.utils_constitutive import tensor2d_to_3d_single
from utilSelf.general import echo


class mlCons_base_single(constitutiveSingle):
    def __init__(self, p0=1e5, ndim=2, device=torch.device('cpu')):
        constitutiveSingle.__init__(self, p0=p0, ndim=ndim)
        self.device = device
        self.ndim = ndim
        self.p0 = torch.tensor(p0, dtype=torch.float32, device=self.device)
        self.sig = torch.tensor(data=np.eye(3), dtype=torch.float32, device=self.device) * self.p0
        self.eps = torch.zeros(size=[3, 3], dtype=torch.float32, device=self.device)
        self.eps_abs = torch.zeros(size=[3, 3], dtype=torch.float32, device=self.device)
        self.p, self.q = self.p0, getQ(self.sig)
        self.epsvp = torch.tensor(data=0., dtype=torch.float32, device = self.device)

    def solver(self, deps):
        sig = deps
        return sig

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
        pass


class mlCons_single(mlCons_base_single):
    def __init__(self,
                 # lambdaa=0.135, kappa=0.04, N=1.973, Z=0.93393, nu=0.3, m=1.8, ocr=120., theta_degree=30,
                 lambdaa, kappa, N, Z, ocr, M = 1.25,
                 nu=0.2, m=1.8, p0=1e5, ndim=2,
                 layer_list='dmd', node=10,
                 verbose_flag=False, device=torch.device('cpu')):
        mlCons_base_single.__init__(self, p0=p0, ndim=ndim, device=device)
        # NN for yield function (p, q, H) -> yield_value
        # self.NN_f = Net_Simple(inputNum=3, outputNum=1, fourier_features=True, node=node, layerList=layer_list)

        # NN for hardening function (depsvp, xi) -> dH
        # self.NN_h = Net_Simple(inputNum=2, outputNum=1, fourier_features=True, node=node, layerList=layer_list)

        self.verbose_flag = verbose_flag
        self.ndim = ndim
        self.yieldTolerance = 0.05
        # parameters
        # self.theta_degree_log = torch.tensor(data=theta_degree, dtype=torch.float32, requires_grad=True)
        # self.ocr_log = torch.tensor(data=ocr, dtype=torch.float32, requires_grad=True)
        #
        # self.theta_degree = torch.exp(self.theta_degree_log)
        # self.ocr = torch.exp(self.ocr_log)

        self.ocr_log = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.M_log = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.lambdaa_log = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.kappa_log = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.N_log = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.Z_log = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.nu_log = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=self.device)
        self.m_log = torch.tensor(1.0, dtype=torch.float32, requires_grad=True, device=self.device)

        self.ocr_start = np.log(ocr)
        self.M_start = M
        self.lambdaa_start = lambdaa
        self.kappa_start = kappa
        self.N_start =N
        self.Z_start =Z
        self.nu_start =nu
        self.m_start =m

        self.ocr = torch.exp(self.ocr_log*self.ocr_start)
        self.M = self.M_log*self.M_start
        self.lambdaa = self.lambdaa_log*self.lambdaa_start
        self.kappa = self.kappa_log*self.kappa_start
        self.N = self.N_log*self.N_start
        # location in normal consolidation on e-lnp space, where p = 1kPa
        self.Z = self.Z_log*self.Z_start
        self.nu = self.nu_log*self.nu_start
        self.m = self.m_log*self.m_start

        # self.lambdaa = torch.tensor(lambdaa, dtype=torch.float32, requires_grad=True)
        # self.kappa = torch.tensor(kappa, dtype=torch.float32, requires_grad=True)
        # self.ocr = torch.tensor(ocr, dtype=torch.float32, requires_grad=True)
        # self.N = torch.tensor(N, dtype=torch.float32, requires_grad=True)
        # # location in normal consolidation on e-lnp space, where p = 1kPa
        # self.Z = torch.tensor(Z, dtype=torch.float32, requires_grad=True)
        # self.m = torch.tensor(m, dtype=torch.float32, requires_grad=True)
        # self.nu = torch.tensor(nu, dtype=torch.float32, requires_grad=True)

        self.p0 = torch.tensor(p0, dtype=torch.float32, device=self.device)
        self.ps = torch.exp((self.N-self.Z)/self.lambdaa)-1.0

        self.e_eta = self.get_e_eta(eta=0., p=self.p0)
        self.e0 = self.e_eta-(self.lambdaa-self.kappa) * \
                  torch.log((self.p0*self.ocr+self.ps*1e3)/(self.p0+self.ps*1e3))
        self.c_p = (self.lambdaa - self.kappa) / (1. + self.e0)

        self.lam, self.G = self.get_lam_G(p=self.p0, e=self.e0)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)

        # state
        # self.sig = torch.tensor(data=np.eye(3), dtype=torch.float32, device=self.device) * self.p0
        # self.eps = torch.zeros(size=[3, 3], dtype=torch.float32, device=self.device)
        # self.eps_abs = torch.zeros(size=[3, 3], dtype=torch.float32, device=self.device)
        # self.p, self.q= self.p0, getQ(self.sig)
        # self.epsvp = 0.
        self.e = self.e0
        self.xi = self.e_eta-self.e0
        self.eta = self.q/self.p
        self.px0 = self.p0*(1+(self.eta/self.M)**2)  # CAUTION: the px0 is in unit of kPa
        self.H = torch.tensor(data=0., dtype=torch.float32, device=self.device)
        self.yieldValue = self.yieldFunction(q=self.q, p=self.p, H=self.H)
        self.m_c, self.m_f = self.get_m_c(self.xi), self.get_m_f(self.xi)

    def get_lam_G(self, p: torch.Tensor, e: torch.Tensor):
        K = (1. + e) * (p + self.ps * 1e3) / self.kappa
        G = 3. * (1 - 2 * self.nu) * K / 2. / (1. + self.nu)
        lam = K - 2. / 3. * G
        return lam, G

    # def yieldFunction_ml(self, p, q, H):
    #     in_tensor = torch.tensor(
    #         data=[p, q, H], requires_grad=True).reshape([-1, 3])
    #     return self.NN_f(in_tensor)[0, 0]
    #
    # def get_df_dpdqdH_ml(self, p, q, H):
    #     in_tensor = torch.tensor(
    #         data=[p, q, H],
    #         requires_grad=True, dtype=torch.float32).reshape([-1, 3])
    #     df_dpdqdH = self.NN_f.get_dy(x=in_tensor)
    #     return df_dpdqdH[0]

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
        dp_dsigma, dq_dsigma = get_dpdsig_dqdsigma(sigma=sigma)
        dfdsigma = dfdp*dp_dsigma+dfdq*dq_dsigma
        return dfdsigma

    def yieldFunction(self, p, q, H):
        f = torch.log(((1.+q**2/self.M**2./p**2.)*p+self.ps*1e3)/(self.ps*1e3+self.px0))-H/self.c_p
        return f

    def get_df_dpdqdH(self, p, q):
        eta = q / p
        dfdp_up = self.M ** 2. - eta ** 2.
        dfdp_low = self.M ** 2. * (p + self.ps * 1e3) + q ** 2. / p
        dfdq_up = 2. * eta
        dfdp = dfdp_up / dfdp_low
        dfdq = dfdq_up / dfdp_low
        df_dH = -self.c_p
        return torch.stack([dfdp, dfdq, df_dH])

    def dgdsig(self, m_c, eta, p, sigma): # checked
        term1 = (m_c ** 2. - eta ** 2.) * torch.eye(3, device=self.device) / 3.
        term2 = 3. * (sigma - p * torch.eye(3, device=self.device)) / p
        v = (term1 + term2) / p/ (m_c ** 2. + eta ** 2.)
        return v

    def get_dH(self, deps_vp, eta, m_f, m_c):
        if m_c == eta:
            return (m_f ** 4 - eta ** 4) / torch.tensor(1e-8) * deps_vp
        else:
            return (m_f ** 4 - eta ** 4) / (m_c ** 4 - eta ** 4) * deps_vp

    # def get_dH_depsvpdxi(self, epsvp,xi):
    #     in_tensor = torch.concat((epsvp, xi))
    #     dH_depsvpdxi = self.NN_f.get_dy(x=in_tensor)
    #     return dH_depsvpdxi

    def get_e_eta(self, p, eta):
        p = p / 1e3
        ''' UH model '''
        # e_eta = self.N-self.lambdaa*np.log(p)-(self.lambdaa-self.kappa)*np.log(1.+eta**2./self.M**2.)
        ''' CSUH model'''
        e_eta = self.Z - self.lambdaa * torch.log((p + self.ps) / (1. + self.ps)) - \
                (self.lambdaa - self.kappa) * \
                torch.log(((1.0 + eta ** 2 / self.M ** 2) * p + self.ps) / (p + self.ps))
        return e_eta

    def solver(self, deps):
        deps = torch.tensor(deps, dtype=torch.float32, device=self.device)
        e = self.e - (self.e + 1.) * getVolStrain(deps)
        sig_trial = self.sig + torch.einsum('ijkl, kl->ij', self.D, deps)
        p, q = getP(sig_trial), getQ(sig_trial)
        eta = q/p
        # ------ failure ckeck() ------
        if p < 0:  # failed
            if self.verbose_flag:
                echo('111 Failure with the mean stress %.3e Pa' % (p))
            scences = self.failure_scences(deps=deps)
            return self.sig, scences

        yieldValue = self.yieldFunction(q=q, p=p, H=self.H)
        if yieldValue < 0:  # Elastic
            e_eta = self.get_e_eta(eta=eta, p=p)
            xi = e_eta - e
            try:
                scence = [sig_trial, self.eps + deps, self.eps_abs+torch.abs(deps),
                          [yieldValue, p, q, e, xi, self.epsvp, self.H]]
            except:
                print()
                raise
            return sig_trial, scence
        elif self.yieldValue < -self.yieldTolerance:
            rmid, sig_last, yieldValue_last = self.transformSplit(deps=deps, D=self.D)
            e_last = self.e - (self.e + 1.) * getVolStrain(deps * rmid)
            p_last, q_last = getP(sigma=sig_last), getQ(sigma=sig_last)
            eta_last = q_last / p_last
            lam_last, G_last = self.get_lam_G(p=p_last, e=e_last)
            D_last = get_elasticMatrix(lam=lam_last, G=G_last)
            xi_last = self.get_e_eta(eta=eta_last, p=p_last) - e_last
            m_c_last, m_f_last = self.get_m_c(xi=xi_last), self.get_m_f(xi=xi_last)
            deps_left = deps * (1 - rmid)
            eps_last = self.eps + deps * rmid
            eps_abs_last = self.eps_abs + torch.abs(deps * rmid)
        else:
            sig_last = self.sig
            deps_left = deps
            D_last = self.D
            e_last = self.e
            p_last, q_last = self.p, self.q
            yieldValue_last = self.yieldValue
            xi_last = self.xi
            m_c_last, m_f_last = self.m_c, self.m_f
            eps_last = self.eps
            eps_abs_last = self.eps_abs
        return self.plasticReturnMapping(
            deps=deps_left, sig_last=sig_last, D_last=D_last, e_last=e_last, p_last=p_last,
            q_last=q_last, yieldValue_last=yieldValue_last,
            xi_last=xi_last, m_c_last=m_c_last, m_f_last=m_f_last, eps_last=eps_last, eps_abs_last=eps_abs_last)

    def plasticReturnMapping(
            self, deps: torch.Tensor,
            sig_last, D_last, e_last, p_last,
            q_last, yieldValue_last,
            xi_last, m_c_last, m_f_last, eps_last, eps_abs_last,
    ):
        eta_last = q_last/p_last
        e = e_last - (1. + e_last) * getVolStrain(deps)
        dfdsig = self.dfdsig(sigma=sig_last, p=p_last, q=q_last)
        if m_c_last == eta_last:
            df_depsvp = - (m_f_last ** 4 - eta_last ** 4) / torch.tensor(1e-8)/self.c_p
        else:
            df_depsvp = - (m_f_last ** 4 - eta_last ** 4) / (m_c_last ** 4 - eta_last ** 4)/self.c_p
        dgdsig = self.dgdsig(m_c=m_c_last, p=p_last, eta=eta_last, sigma=sig_last)
        temp = torch.einsum('ij, ijkl, kl->', dfdsig, D_last, dgdsig) - \
               df_depsvp * torch.trace(dgdsig)
        dlam = (torch.einsum('ij, ijkl, kl->', dfdsig, D_last, deps)
                + yieldValue_last if yieldValue_last < 1e5 else 0.) / temp
        deps_p = dlam * dgdsig
        deps_vp = torch.trace(deps_p)
        sig = sig_last + torch.einsum('ijkl, kl->ij', D_last, deps - deps_p)
        p, q = getP(sig), getQ(sig)

        if p < 0.:  # failed
            if self.verbose_flag:
                echo('222 Failure with the mean stress %.3e Pa' % (p))
            scences = self.failure_scences(sig=sig_last, deps=deps)
            return sig_last, scences

        eta = q / p
        xi = self.get_e_eta(eta=eta, p=p) - e
        epsvp = self.epsvp + deps_vp
        H = self.H + self.get_dH(m_f=m_f_last, m_c=m_c_last, eta=eta_last, deps_vp=deps_vp)
        yieldValue = self.yieldFunction(q=q, p=p, H=H)
        try:
            scence = [sig, eps_last + deps, eps_abs_last + torch.abs(deps),
                  [yieldValue, p, q, e, xi, epsvp, H]]
        except:
            print()
            raise
        return sig, scence

    def failure_scences(self, deps, sig=None):
        e = self.e - (1. + self.e) * getVolStrain(deps)
        deps_vp = torch.trace(deps)
        xi, H = self.xi, self.H
        sig = self.sig if sig is None else sig
        try:
            scences = [sig, self.eps + deps, self.eps_abs + torch.abs(deps),
                   [self.yieldValue, getP(sig), getQ(sig), e, xi, self.epsvp + deps_vp, H]]
        except:
            print()
            raise
        return scences

    def get_m_c(self, xi):
        ''' Eq. (33) '''
        return self.M * torch.exp(-self.m * xi)

    def get_m_f(self, xi):
        mf = 6. / (torch.sqrt(12. * (3. - self.M) / self.M ** 2 *
                           torch.exp(-xi / (self.lambdaa - self.kappa)) + 1.) + 1.)
        return mf

    def transformSplit(self, deps, D):
        rmin, rmax, rmid = 0., 1., 0.5
        sig = self.sig + torch.einsum('ijkl, kl->ij', D, deps * rmid)
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
            sig = self.sig + torch.einsum('ijkl, kl->ij', D, deps * rmid)
            p = getP(sigma=sig)
            q = getQ(sigma=sig)
            yieldValue = self.yieldFunction(q=q, p=p, H=self.H)
            split_num += 1
            if split_num > 100:
                echo('Split num:\t%d yieldValue:\t %.3e last_yieldValue:\t %.3e rmid:\t %.3e' %
                     (split_num, yieldValue, self.yieldValue, rmid))
                raise RuntimeError
        return rmid, sig, yieldValue


    def update_internal(self, yieldValue, p, q, e, xi, epsvp, H):
        'yieldValue, p, q, e, xi, epsvp, H'
        # yieldValue, p, q, e, xi, epsvp, H = [i.item() if type(i) == torch.Tensor else i for i in [yieldValue, p, q, e, xi, epsvp, H]]
        self.yieldValue = yieldValue
        self.p, self.q, self.e, self.xi, self.epsvp, self.H = p, q, e, xi, epsvp, H
        self.m_c, self.m_f = self.get_m_c(self.xi), self.get_m_f(self.xi)
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
        # parameters
        # self.p0 = torch.tensor(p0, dtype=torch.float32)
        # self.theta_degree = torch.tensor(data=theta_degree, dtype=torch.float32, requires_grad=True)

        self.ocr = torch.exp(self.ocr_log * self.ocr_start)
        self.M = self.M_log * self.M_start
        self.lambdaa = self.lambdaa_log * self.lambdaa_start
        self.kappa = self.kappa_log * self.kappa_start
        self.N = self.N_log * self.N_start
        # location in normal consolidation on e-lnp space, where p = 1kPa
        self.Z = self.Z_log * self.Z_start
        self.nu = self.nu_log * self.nu_start
        self.m = self.m_log * self.m_start

        self.ps = torch.exp((self.N-self.Z)/self.lambdaa)-1.0
        # self.nu = torch.tensor(nu, dtype=torch.float32, requires_grad=True)

        self.e_eta = self.get_e_eta(eta=0., p=self.p0)
        self.e0 = self.e_eta-(self.lambdaa-self.kappa) * \
                  torch.log((self.p0*self.ocr+self.ps*1e3)/(self.p0+self.ps*1e3))
        self.c_p = (self.lambdaa - self.kappa) / (1. + self.e0)

        self.lam, self.G = self.get_lam_G(p=self.p0, e=self.e0)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)

        # state
        self.sig = torch.tensor(data=np.eye(3), dtype=torch.float32, device = self.device) * self.p0
        self.eps = torch.zeros(size=[3, 3], dtype=torch.float32, device = self.device)
        self.eps_abs = torch.zeros(size=[3, 3], dtype=torch.float32, device = self.device)
        self.p, self.q= self.p0, torch.tensor(data=0., dtype=torch.float32, requires_grad=False, device = self.device)
        self.epsvp = torch.tensor(data=0., dtype=torch.float32, device = self.device)
        self.e = self.e0
        self.xi = self.e_eta-self.e0
        self.m_c, self.m_f = self.get_m_c(self.xi), self.get_m_f(self.xi)
        self.eta = self.q/self.p
        self.px0 = self.p0*(1+(self.eta/self.M)**2)  # CAUTION: the px0 is in unit of kPa
        self.H = torch.tensor(data=0., dtype=torch.float32, device = self.device)
        self.yieldValue = self.yieldFunction(q=self.q, p=self.p, H=self.H)
