import torch
import numpy as np
from FEMxML.torch_net import Net_Simple
from FEMxEPxML.mlCons import mlCons_base_single
from FEMxEPxML.utils_constitutive_ml import get_elasticMatrix, getVolStrain, get_M, \
    getP, getQ, get_dpdsig_dqdsigma, getQEps, get_deps_s_deps
from FEMxEPxML.utils_constitutive import tensor2d_to_3d_single
from utilSelf.general import echo
from FEMxML.torch_net import Net_Simple


class mlvonmises_single(mlCons_base_single):
    def __init__(self, explicitFlag=True,
                 p0=1e5, nu=0.2, E=np.log(2e7), A=np.log(3e5), B=0.2, epsilon0=0.02, yield_stress0=np.log(1e3),
                 verboseFlag=False, ndim=2, node=10, layer_list='dmd', NN_flag=True):
        mlCons_base_single.__init__(self, p0=p0, ndim=2)
        # Net
        self.NN_flag = NN_flag
        self.NN_h = Net_Simple(inputNum=1, outputNum=1, fourier_features=True, node=node, layerList=layer_list)

        # calculation settings
        self.ndim = ndim
        self.verboseFlag = verboseFlag
        self.explicitFlag = explicitFlag

        # material constants
        self.E_log = torch.tensor(E, dtype=torch.float32, requires_grad=True)
        self.A_log = torch.tensor(A, dtype=torch.float32, requires_grad=True)
        self.yield_stress0_log = torch.tensor(yield_stress0, dtype=torch.float32, requires_grad=True)
        self.E = torch.exp(self.E_log)
        self.A = torch.exp(self.A_log)
        self.yield_stress0 = torch.exp(self.yield_stress0_log)  #
        # self.E = torch.tensor(E, dtype=torch.float32, requires_grad=True)  #
        # self.A = torch.tensor(A, dtype=torch.float32, requires_grad=True)  #
        # self.yield_stress0 = torch.tensor(yield_stress0, dtype=torch.float32, requires_grad=True)  #
        self.nu = torch.tensor(nu, dtype=torch.float32, requires_grad=True)
        self.B = torch.tensor(B, dtype=torch.float32, requires_grad=True)
        self.epsilon0 = torch.tensor(epsilon0, dtype=torch.float32, requires_grad=True)
        self.lam = self.E * self.nu / (1. + self.nu) / (1 - 2. * self.nu)
        self.G = self.E / 2. / (1. + self.nu)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)

        # state variables
        self.p0 = torch.tensor(p0, dtype=torch.float32)
        self.eps_p = torch.zeros([3, 3], dtype=torch.float)
        self.eps_s_p = torch.tensor(0., dtype=torch.float32)
        self.H = self.hardeningFunction(self.eps_s_p)
        self.yieldValue = self.yieldFunction(q=getQ(self.sig), H=self.H)

    def yieldFunction(self, q, H):
        f = q-H-self.yield_stress0
        return f

    def hardeningFunction(self, eps_s_p):
        if self.NN_flag:
            H = self.NN_h(eps_s_p.reshape([-1, 1]))[0, 0]
        else:
            H = self.A * (self.epsilon0 + eps_s_p) ** self.B
        return H

    def solver(self, deps):
        deps = torch.tensor(deps, dtype=torch.float32)
        sig_trial = self.sig + torch.einsum('ijkl, kl->ij', self.D, deps)
        q = getQ(sig_trial)
        yieldValue = self.yieldFunction(q=q, H=self.H)
        if yieldValue < 0:  # Elastic
            scene = [sig_trial, self.eps + deps, self.eps_abs+torch.abs(deps),
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
        temp1  = torch.einsum('ij, ijkl, kl->', dfdsig, self.D, deps)
        temp2 = torch.einsum('ij, ijkl, kl->', dfdsig,self.D,dfdsig)
        temp3 = torch.einsum('ij, ij->', dfdeps_p, dfdsig)
        dlam = (self.yieldValue+temp1)/(temp2-temp3)
        deps_p = dlam*dfdsig
        eps_p  =self.eps_p+deps_p
        eps_s_p = self.eps_s_p+getQEps(deps_p)
        H = self.hardeningFunction(eps_s_p=eps_s_p)
        sig = self.sig + torch.einsum('ijkl, kl->ij', self.D, deps-deps_p)
        q = getQ(sigma=sig)
        yieldValue = self.yieldFunction(q=q, H=H)
        scene = [sig, self.eps + deps, self.eps_abs+torch.abs(deps),
                      [yieldValue, eps_p, eps_s_p, H]]
        if self.explicitFlag:
            return sig, scene
        else:
            Dep = self.D - torch.einsum('ijmn, mn, st, stkl', self.D, dfdsig, dfdsig, self.D)/(temp2-temp3)
            return sig, Dep, scene

    def get_dH_deps_s(self):
        if self.NN_flag:
            x = self.eps_s_p.reshape([-1, 1]).clone().detach().requires_grad_(True)
            # self.eps_s_p.reshape([-1, 1])
            dH_deps_s = self.NN_h.get_dy(x=x)[0, 0]
        else:
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
        mlCons_base_single.__init__(self, p0=1e5, ndim=2)
        # calculation settings

        # material constants
        self.E = torch.exp(self.E_log)
        self.A = torch.exp(self.A_log)
        self.yield_stress0 = torch.exp(self.yield_stress0_log)  #
        self.lam = self.E * self.nu / (1. + self.nu) / (1 - 2. * self.nu)
        self.G = self.E / 2. / (1. + self.nu)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)

        # state variables
        self.p0 = torch.tensor(1e5, dtype=torch.float32)
        self.eps_p = torch.zeros([3, 3], dtype=torch.float)
        self.eps_s_p = torch.tensor(0., dtype=torch.float32)
        self.H = self.hardeningFunction(self.eps_s_p)
        self.yieldValue = self.yieldFunction(q=getQ(self.sig), H=self.H)
