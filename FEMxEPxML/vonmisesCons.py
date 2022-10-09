import multiprocessing
import numpy as np
from FEMxEPxML.constitutive import ConstitutiveMask, constitutiveSingle
from FEMxEPxML.utils_constitutive import tensor2_tensor3, returnedDatasDecode, \
    get_elasticMatrix, getVolStrain, getP, getQ, get_M, get_dpdsig_dqdsigma, get_deps_s_deps, getQEps
from utilSelf.general import echo, mapMask


class vonmisesConstitutive(ConstitutiveMask):
    def __init__(self, explicitFlag, numg, pool: multiprocessing.Pool, save_path:str, rho: float,
                 p0=1e5, poisson=0.2, E=2e7, A=3e5, B=0.2, epsilon0=0.02, yield_stress0=1e3,
                 verboseFlag=False, ndim=3, save_flag=False):  # parameters used in the ML work
    # def __init__(self, explicitFlag, numg, pool: multiprocessing.Pool, save_path:str, rho: float,
    #              p0=1e5, poisson=0.2, E=2e7, A=3e5, B=0.05, epsilon0=0.00, yield_stress0=0.,
    #              verboseFlag=False, ndim=3, save_flag=False):
        self.cons = [
            vonmisesSingle(explicitFlag,
                 p0, poisson, E, A, B, epsilon0, yield_stress0,
                 verboseFlag, ndim) for _ in range(numg)]
        ConstitutiveMask.__init__(
            self, p0=p0, ndim=ndim, cons=self.cons, rho=rho,
            explicitFlag=explicitFlag, pool=pool, numg=numg, name='vonmises', save_path=save_path, save_flag=save_flag)


class vonmisesSingle(constitutiveSingle):
    def __init__(self, explicitFlag,
                 p0, poisson, E, A, B, epsilon0, yield_stress0,
                 verboseFlag, ndim):
        constitutiveSingle.__init__(self, p0=p0, ndim=ndim)
        # calculation settings
        self.ndim = ndim
        self.verboseFlag = verboseFlag
        self.explicitFlag = explicitFlag

        # material constants
        self.E = E
        self.poisson = poisson
        self.lam = self.E * self.poisson / (1. + self.poisson) / (1 - 2. * self.poisson)
        self.G = self.E / 2. / (1. + self.poisson)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)
        self.A = A
        self.B = B
        self.epsilon0 = epsilon0
        self.yield_stress0 = yield_stress0

        # state variables
        self.eps_p = np.zeros(shape=[3, 3])
        self.eps_s_p = 0.
        self.H = self.hardeningFunction(self.eps_s_p)
        self.yieldValue = self.yieldFunction(q=getQ(self.sig), H=self.H)

    def yieldFunction(self, q, H):
        f = q-H-self.yield_stress0
        return f

    def hardeningFunction(self, eps_s_p):
        H = self.A * (self.epsilon0 + eps_s_p) ** self.B
        return H

    def solver(self, deps):
        sig_trial = self.sig + np.einsum('ijkl, kl->ij', self.D, deps)
        q = getQ(sig_trial)
        yieldValue = self.yieldFunction(q=q, H=self.H)
        if yieldValue < 0:  # Elastic
            scene = [sig_trial, self.eps + deps, self.eps_abs+np.abs(deps),
                      [yieldValue, self.eps_p, self.eps_s_p, self.H]]
            if self.explicitFlag:
                return sig_trial, scene
            else:
                return sig_trial, self.D, scene
        else:  # Plastic
            return self.plasticReturnMapping(deps=deps)

    def plasticReturnMapping(self, deps):
        dfdq = 1.
        dpdsig, dqdsig = get_dpdsig_dqdsigma(self.sig)
        dfdsig = dfdq*dqdsig
        dfdH = -1.
        dH_deps_p = self.get_dH_deps_s()
        dfdeps_p = dfdH*dH_deps_p
        temp1  = np.einsum('ij, ijkl, kl->', dfdsig, self.D, deps)
        temp2 = np.einsum('ij, ijkl, kl->', dfdsig,self.D,dfdsig)
        temp3 = np.einsum('ij, ij->', dfdeps_p, dfdsig)
        dlam = (self.yieldValue+temp1)/(temp2-temp3)
        deps_p = dlam*dfdsig
        eps_p  =self.eps_p+deps_p
        eps_s_p = self.eps_s_p+getQEps(deps_p)
        H = self.hardeningFunction(eps_s_p=eps_s_p)
        sig = self.sig + np.einsum('ijkl, kl->ij', self.D, deps-deps_p)
        q = getQ(sigma=sig)
        yieldValue = self.yieldFunction(q=q, H=H)
        scene = [sig, self.eps + deps, self.eps_abs+np.abs(deps),
                      [yieldValue, eps_p, eps_s_p, H]]
        if self.explicitFlag:
            return sig, scene
        else:
            Dep = self.D - np.einsum('ijmn, mn, st, stkl', self.D, dfdsig, dfdsig, self.D)/(temp2-temp3)
            return sig, Dep, scene

    def get_dH_deps_s(self):
        temp = self.epsilon0+self.eps_s_p
        if temp <= 0.:
            dH_deps_s = self.A*self.B*1.
        else:
            dH_deps_s = self.A*self.B*(temp)**(self.B-1.)
        deps_s_deps_p = get_deps_s_deps(self.eps_p)
        dH_deps_p = dH_deps_s*deps_s_deps_p
        return dH_deps_p

    def update_internal(self, yieldValue, eps_p, eps_s_p, H):
        """
        [sig, self.eps + deps,
                      yieldValue, eps_p, eps_s_p, H]
        """
        self.yieldValue = yieldValue
        self.eps_p = eps_p
        self.eps_s_p = eps_s_p
        self.H = H

    def return2initial(self):
        constitutiveSingle.__init__(self, p0=self.p0, ndim=self.ndim)
        # state variables
        self.eps_p = np.zeros(shape=[3, 3])
        self.eps_s_p = 0.
        self.H = self.hardeningFunction(self.eps_s_p)
        self.yieldValue = self.yieldFunction(q=getQ(self.sig), H=self.H)