import multiprocessing
import numpy as np
from FEMxEPxML.constitutive import ConstitutiveMask, constitutiveSingle
from FEMxEPxML.utils_constitutive import tensor2_tensor3, returnedDatasDecode, \
    get_elasticMatrix, getVolStrain, getP, getQ, get_M, get_dpdsig_dqdsigma, get_deps_s_deps, getQEps
from utilSelf.general import echo, mapMask


class vonmisesConstitutive(ConstitutiveMask):
    def __init__(self, explicitFlag, numg,  save_path:str, rho: float,
                 p0=1e5, nu=0.2, E=2e7, A=3e5, B=0.2, epsilon0=0.02, yield_stress0=1e3,

                 verboseFlag=False, ndim=3, save_flag=False, nump=1):  # parameters used in the ML work
    # def __init__(self, explicitFlag, numg, pool: multiprocessing.Pool, save_path:str, rho: float,
    #              p0=1e5, nu=0.2, E=2e7, A=3e5, B=0.05, epsilon0=0.00, yield_stress0=0.,
    #              verboseFlag=False, ndim=3, save_flag=False):
        self.cons = [
            vonmisesSingle(explicitFlag=explicitFlag,
                 p0=p0, nu=nu, E=E, A=A, B=B, epsilon0=epsilon0, yield_stress0=yield_stress0,
                 verboseFlag=verboseFlag, ndim=ndim) for _ in range(numg)]
        ConstitutiveMask.__init__(
            self, p0=p0, ndim=ndim, cons=self.cons, rho=rho,
            explicitFlag=explicitFlag, nump=nump,  numg=numg, name='vonmises', save_path=save_path, save_flag=save_flag)


class vonmisesSingle(constitutiveSingle):
    def __init__(self,
                 p0,
                 nu, E, A, B, epsilon0, yield_stress0,
                 # dilation_coefficient:8.246e-02 	 yield_p_c:8.175e-02 	 C:3.504e+05 	 D:2.320e-01 	 epsilon0_p:1.668e-02 	 harden_E
                 dilation_coefficient=0., yield_p_c=0., C=0., D=0., epsilon0_p=0., harden_E=0.,
                 verboseFlag=False, ndim=2, explicitFlag=True):
        constitutiveSingle.__init__(self, p0=p0, ndim=ndim)
        # calculation settings
        self.ndim = ndim
        self.verboseFlag = verboseFlag
        self.explicitFlag = explicitFlag

        # material constants
        self.dilation_coefficient, self.yield_p_c, self.C, self.D, self.epsilon0_p, self.harden_E = \
            dilation_coefficient, yield_p_c, C, D, epsilon0_p, harden_E
        self.E_tangential = E
        self.E = self.get_E(eps_p_p=0.)
        self.nu = self.nu = nu
        self.lam = self.E * self.nu / (1. + self.nu) / (1 - 2. * self.nu)
        self.G = self.E / 2. / (1. + self.nu)
        self.K = self.E/3./(1.-2.*self.nu)
        self.De = get_elasticMatrix(lam=self.lam, G=self.G)
        self.A = A
        self.B = B
        self.epsilon0 = epsilon0
        self.yield_stress0 = yield_stress0

        # state variables
        self.eps_p = np.zeros(shape=[3, 3])
        self.eps_s = 0.
        self.eps_s_p = 0.
        self.eps_p_p = 0.
        self.H = self.hardeningFunction(eps_p_p=self.eps_p_p, eps_s_p=self.eps_s_p)
        # if set the initial yieldValue as 0, the optimization will results in nan
        # that's why we do not set the value as 0.
        # self.f0 = self.H + self.yield_stress0 - (self.p*self.yield_p_c + self.q)
        self.yieldValue = self.yieldFunction(p=self.p, q=getQ(self.sig), H=self.H)

    def get_E(self, eps_p_p):
        E = self.E_tangential * (1. + eps_p_p) ** self.harden_E
        return E

    def yieldFunction(self, p, q, H):
        f = p*self.yield_p_c + q-H - self.yield_stress0
        return f

    def get_dfdp_dfdq(self, ):
        dfdp, dfdq = self.yield_p_c, 1.
        dfdH = -1.
        return dfdp, dfdq, dfdH

    def get_dgdp_dgdq(self, dfdp, dfdq):
        dgdp = self.dilation_coefficient + dfdp
        dgdq = dfdq
        return dgdp, dgdq

    def hardeningFunction(self, eps_p_p, eps_s_p):
        H = self.A * (self.epsilon0 + eps_s_p) ** self.B + \
                self.C * (self.epsilon0_p + eps_p_p) ** self.D
        return H

    def get_dHdepsp_dHdepsq(self, ):
        dHdepsp = self.C * (self.D) * (self.epsilon0_p + self.eps_p_p) ** (self.D - 1.) if self.C != 0 else 0.
        dHdepsq = self.A*(self.B)*(self.epsilon0+self.eps_s_p) ** (self.B-1.)
        return dHdepsp, dHdepsq

    def solver(self, deps):
        deps_norm = np.linalg.norm(deps)
        step_num = int(deps_norm / 0.0002) + 1
        if step_num < 1:
            step_num = 1
        step_size = 1. / step_num
        remain, split_num = 1.0, 0
        scece_safe = self.get_current_scene()
        while remain > 1e-5 and split_num < 10:
            if self.explicitFlag:
                sig, scene = self.solver_single(deps=deps * step_size)
            else:
                sig, D, scene = self.solver_single(deps=deps * step_size)
            remain -= step_size
            self.update(*scene)
        if remain == 1.0:
            # raise
            sig, scene = scece_safe[0], scece_safe
            if not self.explicitFlag:
                D = self.De * 0.1
        self.update(*scece_safe)
        if self.explicitFlag:
            return sig, scene
        else:
            return sig, D, scene

    def solver_single(self, deps):
        # deps = np.tensor(deps, dtype=np.float32)
        sig_trial = self.sig + np.einsum('ijkl, kl', self.De, deps)
        p_trial, q_trial = getP(sig_trial), getQ(sig_trial)
        yieldValue = self.yieldFunction(p=p_trial, q=q_trial, H=self.H)
        if yieldValue < 0:  # Elastic
            sig_trial = self.sig + np.einsum('ijkl, kl', self.De, deps)
            scene = [sig_trial, self.eps + deps, self.eps_abs + np.abs(deps),
                     [yieldValue, self.eps_p, self.eps_s_p, self.H, p_trial, q_trial, self.eps_p_p]]
            if self.explicitFlag:
                return sig_trial, scene
            else:
                return sig_trial, self.De, scene
        else:  # Plastic
            return self.plasticReturnMapping(deps=deps)

    def plasticReturnMapping(self, deps):
        # deps_dev = deps-np.trace(deps)/3.*np.eye(3)
        # eps_dev = self.eps - np.trace(self.eps)/3.*np.eye(3)

        dpdsig, dqdsig = get_dpdsig_dqdsigma(self.sig)
        dfdp, dfdq, dfdH = self.get_dfdp_dfdq()
        dHdepsp, dHdepsq = self.get_dHdepsp_dHdepsq()
        dgdp, dgdq = self.get_dgdp_dgdq(dfdp=dfdp, dfdq=dfdq)  # associated flow
        dfdsig = dfdp * dpdsig + dfdq * dqdsig
        dgdsig = dgdp * dpdsig + dgdq * dqdsig

        temp0 = np.einsum('ij, ijkl->kl', dfdsig, self.De)
        temp1 = np.einsum('kl, kl->', temp0, deps) + self.yieldValue
        temp2 = np.einsum('kl, kl->', temp0, dgdsig)
        temp3 = dfdH * (dHdepsp * dgdp + dHdepsq * dgdq)
        dlam = temp1 / (temp2 - temp3)

        deps_p = dlam * dgdsig
        sig = self.sig + np.einsum('ijkl, kl->ij', self.De, deps - deps_p)
        p, q = getP(sig), getQ(sig)

        eps_p_p = max(np.trace(self.eps_p + deps_p), 0.)
        eps_s_p = getQEps(self.eps_p + deps_p)

        H = self.hardeningFunction(eps_p_p=eps_p_p, eps_s_p=eps_s_p)
        yieldValue = self.yieldFunction(p=p, q=q, H=H)
        scene = [sig, self.eps + deps, self.eps_abs + np.abs(deps),
                 [yieldValue, self.eps_p + deps_p, eps_s_p, H, p, q, eps_p_p]]
        if self.explicitFlag:
            return sig, scene
        else:
            dfdsig = dfdp * dpdsig + dfdq * dqdsig
            Dep = self.De - np.einsum('ijmn, mn, st, stkl', self.De, dfdsig, dfdsig, self.De) / temp2
            return sig, Dep, scene

    def update_internal(self, yieldValue, eps_p, eps_s_p, H, p, q, eps_p_p):
        """
        [sig, self.eps + deps,
                      yieldValue, eps_p, eps_s_p, H]
        """
        self.yieldValue = yieldValue
        self.eps_p = eps_p
        self.eps_s_p = eps_s_p
        self.eps_p_p = eps_p_p
        self.H = H
        self.p, self.q = getP(self.sig), getQ(self.sig)

        self.E = self.get_E(eps_p_p=eps_p_p)
        self.lam = self.E * self.nu / (1. + self.nu) / (1 - 2. * self.nu)
        self.G = self.E / 2. / (1. + self.nu)
        self.K = self.E / 3. / (1. - 2. * self.nu)
        self.De = get_elasticMatrix(lam=self.lam, G=self.G)

    def get_current_scene(self, ):
        scene = [self.sig, self.eps, self.eps_abs,
         [self.yieldValue, self.eps_p, self.eps_s_p, self.H, self.p, self.q, self.eps_p_p]]
        return scene

    def return2initial(self):
        constitutiveSingle.__init__(self, p0=self.p0, ndim=self.ndim)
        # state variables
        self.eps_s = 0.
        self.eps_p = np.zeros(shape=[3, 3])  # plstic strain tensor
        self.eps_s_p, self.eps_p_p= 0., 0.
        self.H = self.hardeningFunction(eps_p_p=self.eps_s_p, eps_s_p=self.eps_s_p)
        self.yieldValue = self.yieldFunction(p=self.p, q=getQ(self.sig), H=self.H)


    def plastic_in_p_q(self):
        # depsq = getQEps(self.eps+deps)-self.eps_s
        # depsp = np.trace(deps)
        # dfdp, dfdq, dfdH = self.get_dfdp_dfdq()
        # dHdepsp, dHdepsq = self.get_dHdepsp_dHdepsq()
        # dgdp, dgdq = dfdp, dfdq  # associated flow
        # temp1 = (dfdp * depsp * self.K + dfdq * depsq * 3. * self.G)
        # temp2 = (dfdp * dgdp * self.K + dfdq * dgdq * 3. * self.G - dfdH * (dHdepsp * dgdp + dHdepsq * dgdq))
        # dlam = temp1/temp2
        # depspp, depspq = dlam*dgdp, dlam*dgdq
        # eps_s_p = self.eps_s_p+depspq
        # eps_p_p = self.eps_p_p+depspp
        # H = self.hardeningFunction(eps_p_p=eps_p_p, eps_s_p=eps_s_p)
        # p, q = self.p+self.K*(depsp-depspp), self.q + 3.*self.G*(depsq-depspq)
        # yieldValue = self.yieldFunction(p=p, q=q, H=H)
        #
        # dpdsig, dqdsig = get_dpdsig_dqdsigma(self.sig)
        # dgdsig = dgdp*dpdsig+dgdq*dqdsig
        # deps_p = dlam * (dgdsig)
        # sig = self.sig + np.einsum('ijkl, kl->ij', self.De, deps-deps_p)
        # scene = [sig, self.eps + deps, self.eps_abs+np.abs(deps),
        #               [yieldValue, self.eps_p+deps_p, eps_s_p, H, p, q, eps_p_p]]
        pass