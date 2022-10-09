import numpy as np
from FEMxEPxML.constitutive import ConstitutiveMask, tensor2_tensor3
from FEMxEPxML.utils_constitutive import get_elasticMatrix


class elasticConstitutive(ConstitutiveMask):
    def __init__(self, save_path, numg, rho, explicitFlag=False, E=1e8, poisson=0.2, p0=1e5, ndim=2):
        ConstitutiveMask.__init__(self, p0=p0, ndim=ndim, explicitFlag=explicitFlag, numg=numg, name='elastic', rho=rho, save_path=save_path)
        '''
            https://en.wikipedia.org/wiki/Lam%C3%A9_parameters
        '''
        self.lam = E*poisson/(1+poisson)/(1-2.*poisson)
        self.G = E/2./(1+poisson)
        self.D = np.array([get_elasticMatrix(lam=self.lam, G=self.G) for _ in range(numg)])

    def solver(self, deps):
        if len(deps[0]) ==2:
            deps = tensor2_tensor3(t2=deps)
        sig = self.sig + np.einsum('nijkl, nkl->nij', self.D, deps)
        scene = [sig, self.eps+deps]
        if self.explicitFlag:
            # Caution: find out why we cannot update the self.sig?
            # Because the multi-threading in python will copy the
            # variables to calculate instead of using the original
            # variable via points
            return sig, scene
        else:
            return sig, self.D, scene

    def update(self, scenes):
        self.sig= scenes[0]
        self.eps= scenes[1]
