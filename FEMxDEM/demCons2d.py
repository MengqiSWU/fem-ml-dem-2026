import copy
import numpy as np
from yadeimport import *
from FEMxDEM.simDEM import *
from FEMxEPxML.constitutive import ConstitutiveMask
from utilSelf.saveGauss import saveGauss2D
from multiprocessing import Pool


class demConstitutive(ConstitutiveMask):
    def __init__(self, p0, ndim, explicitFlag, numg, nump, save_path, save_flag):
        pool=None
        ConstitutiveMask.__init__(self, p0=p0, ndim=ndim, explicitFlag=explicitFlag, numg=numg, save_path=save_path,
             name='dem', cons=None, pool=pool, save_flag=save_flag)
        self.numg = numg
        self.pool = pool
        self.nump = nump
        if self.ndim == 2:
            self.initLoad = initLoad
            self.getStressAndTangent = getStressAndTangent2D
            self.shear = shear2D
        else:  # ndim=3
            self.initLoad = initLoad3D
            self.getStressAndTangent = getStressAndTangent3D
            self.shear = shear3D
        self.scenes = self.pool.map(self.initLoad, list(range(self.numg))) if self.pool else\
            [self.initLoad(i) for i in range(self.numg)]
        self.sig = np.zeros([self.numg, self.ndim, self.ndim])
        self.eps_abs = np.zeros([self.numg, self.ndim, self.ndim])
        self.D = np.zeros([self.numg, self.ndim, self.ndim, self.ndim, self.ndim])
        st = self.pool.map(self.getStressAndTangent, self.scenes) if self.pool else \
            [self.getStressAndTangent(self.scenes[i]) for i in range(self.numg)]
        for i in range(self.numg):
            # Caution: all of the sig and eps saved in the Cons are in geo-mechanical form
            self.sig[i] = -np.array(st[i][0])
            self.D[i] = np.array(st[i][1])

    def solver(self, deps):
        sig = np.zeros([self.numg, self.ndim, self.ndim])
        eps_abs = self.eps_abs + np.abs(deps)
        D = np.zeros([self.numg, self.ndim, self.ndim, self.ndim, self.ndim])
        # NOTE: revert because compression is negative in DEM model
        if self.nump > 1:
            with Pool(processes=self.nump) as pool:
                scenes = pool.map(self.shear, list(zip(self.scenes, -deps.reshape(self.numg, -1))))
                st = pool.map(getStressAndTangent2D, scenes)
        else:
            scenes = []
            st = []
            for i in range(self.numg):
                temp = self.shear([self.scenes[i], -deps[i].reshape(-1)])
                scenes.append(temp)
                st.append(self.getStressAndTangent(temp))
        for i in range(self.numg):
            sig[i] = np.array(st[i][0])
            D[i] = np.array(st[i][1])
        sig_geo = -sig
        if self.explicitFlag:
            self.update([scenes, eps_abs])
            return sig_geo
        else:
            return sig_geo, D, [scenes, eps_abs]

    def update(self, scenes):
        self.scenes= scenes[0]
        self.eps_abs = scenes[1]
