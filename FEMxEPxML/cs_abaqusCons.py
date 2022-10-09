import copy

from constitutive import ConstitutiveMask, constitutiveSingle
import numpy as np
from utils_constitutive import getP, getQ, getI3, get_elasticMatrix, tensor2_tensor3


class cs_abaqusConSingle(constitutiveSingle):
    """
        https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/usb/default.htm?startat=pt05ch18s03abm29.html#usb-mat-ccapplastic
    """
    def __init__(self, p0=1e5, ndim=2, lambdaa=0.135, kappa=0.07, e1=1.0, e0=0.7, K=0.8, M=1.2, beta=0.8, nu=0.3):
        constitutiveSingle.__init__(self,  p0=p0, ndim=ndim)
        # parameters
        self.lambdaa, self.kappa, self.e1, self.e0, self.K, self.M = lambdaa, kappa, e1, e0, K, M
        self.beta = beta
        self.nu = nu

        # state variables
        self.p, self.q, self.r = getP(self.sig), getQ(self.sig), getI3(self.sig)
        self.t = self.get_t(q=self.q, r=self.r)
        self.a0 = self.get_a0()
        self.a = copy.deepcopy(self.a0)
        lam, G = self.get_lam_G(self.p)
        self.D = get_elasticMatrix(lam=lam, G=G)

        # configuration
        self.yield_tol = 0.05

    def yieldFunction(self, t, p):
        f = (p/self.a-1.0)**2/self.beta**2+(t/self.M/self.a)**2.-1.
        return f

    def get_t(self, q, r):
        return 0.5 * q * (1. + 1. / self.K - (1. - 1. / self.K) * (r / q) ** 3.)

    def renew_a0(self, deps_v_p):
        a = self.a0*np.exp((1+self.e0)*(1.-deps_v_p)/(self.lambdaa-self.kappa*deps_v_p))
        return a

    def get_a0(self, ):
        a0 = 0.5*np.exp((self.e1-self.e0-self.kappa*np.log(self.p0))/(self.lambdaa-self.kappa))
        return a0

    def get_lam_G(self, p):
        K = (1.+self.e0)*p/self.kappa
        lam, G = 3.*K*self.nu/(1.+self.nu), 3.*K*(1.-2.0*self.nu)/2./(1.+self.nu)
        return lam, G

    def solver(self, deps):
        deps = tensor2_tensor3(deps)
        sig = self.sig + np.einsum('ijkl, kl->ij', self.D, deps)
        p, q, r = getP(sig), getQ(sig), getI3(sig)
        t = self.get_t(q=q, r=r)
        f = self.yieldFunction(t=t, p=p)

        if f < 0:
            return sig, self.D
        else:
            return self.return_mapping()

    def return_mapping(self, deps):
        return
