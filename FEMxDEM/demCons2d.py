import copy
import numpy as np
from yadeimport import *
from FEMxDEM.simDEM import *
from FEMxDEM.utiles_H_3f import *
from FEMxEPxML.constitutive import ConstitutiveMask
from utilSelf.saveGauss import saveGauss2D
from multiprocessing import Pool


class demConstitutive(ConstitutiveMask):
    def __init__(self, p0, ndim, explicitFlag, numg, save_path, save_flag, nump):
        ConstitutiveMask.__init__(self, p0=p0, ndim=ndim, explicitFlag=explicitFlag, numg=numg, save_path=save_path,
             name='dem', cons=None, save_flag=save_flag, nump=nump)       #pool=True, 改进后的调用多进程不用pool
        self.numg = numg
        self.nump = nump
        if self.ndim == 2:
            self.initLoad = initLoad
            self.getStressAndTangent = getStressAndTangent2D
            self.shear = shear2D
        else:  # ndim=3
            self.initLoad = initLoad3D
            self.getStressAndTangent = getStressAndTangent3D
            self.shear = shear3D
        self.sig = np.zeros([self.numg, self.ndim, self.ndim])
        self.eps_abs = np.zeros([self.numg, self.ndim, self.ndim])
        self.H_3f = np.zeros([self.numg, 2])
        self.D = np.zeros([self.numg, self.ndim, self.ndim, self.ndim, self.ndim])



        # # # # # # # # # # # # # # 早期可run # # # # # # # # # # # # # #
        # self.scenes = self.pool.map(self.initLoad, list(range(self.numg))) if self.pool else\
        #     [self.initLoad(i) for i in range(self.numg)]
        # self.sig = np.zeros([self.numg, self.ndim, self.ndim])
        # self.eps_abs = np.zeros([self.numg, self.ndim, self.ndim])
        # self.H_3f = np.zeros([self.numg, 2])
        # self.D = np.zeros([self.numg, self.ndim, self.ndim, self.ndim, self.ndim])


        # # # # # # # # # # # # # # 早期可run # # # # # # # # # # # # # #
        # st = self.pool.map(self.getStressAndTangent, self.scenes) if self.pool else \
        #     [self.getStressAndTangent(self.scenes[i]) for i in range(self.numg)]
        # for i in range(self.numg):
        #     # Caution: all of the sig and eps saved in the Cons are in geo-mechanical form
        #     self.sig[i] = -np.array(st[i][0])
        #     self.D[i] = np.array(st[i][1])


        # # # # # # # # # # # # # # 需要修改getcon函数后才能使用 # # # # # # # # # # # # # #
        if self.nump > 1:
            with Pool(processes=self.nump) as pool:
                self.scenes = pool.map(self.initLoad, list(range(self.numg)))
                st = pool.map(self.getStressAndTangent, self.scenes)
        else:
            self.scenes = [self.initLoad(i) for i in range(self.numg)]
            st = [self.getStressAndTangent(self.scenes[i]) for i in range(self.numg)]
        for i in range(self.numg):
            # Caution: all of the sig and eps saved in the Cons are in geo-mechanical form
            self.sig[i] = -np.array(st[i][0])
            self.D[i] = np.array(st[i][1])
        # # # # # # # # # # # # # # 需要修改getcon函数后才能使用 # # # # # # # # # # # # # #


    def solver(self, deps):
        sig = np.zeros([self.numg, self.ndim, self.ndim])
        eps_abs = self.eps_abs + np.abs(deps)
        D = np.zeros([self.numg, self.ndim, self.ndim, self.ndim, self.ndim])
        re_eps_abs = np.delete((eps_abs.reshape(self.numg, -1)), 2, axis=1)
        # NOTE: revert because compression is negative in DEM model
        if self.nump > 1:
            with Pool(processes=self.nump) as pool:
                H1 = pool.map(H_vars_1, list(zip(re_eps_abs[:, 0:1], re_eps_abs[:, 2:3])))
                H2 = pool.map(H_vars_2, list(zip(re_eps_abs[:, 0:1], re_eps_abs[:, 2:3])))
                scenes = pool.map(self.shear, list(zip(self.scenes, -deps.reshape(self.numg, -1))))
                st = pool.map(getStressAndTangent2D, scenes)
                H_3f = np.array(list(zip(H1, H2)))
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


        # for i in range(self.numg):
        #     history_3f[i][0] = np.linalg.norm([eps_abs[i][0][0], eps_abs[i][0][1], eps_abs[i][1][0]])
        #     history_3f[i][1] = np.mean([eps_abs[i][0][0], eps_abs[i][0][1], eps_abs[i][1][0]])
        #     history_3f[i][2] = eps_abs[i][0][0] * eps_abs[i][0][1] * eps_abs[i][1][0]

        if self.explicitFlag:
            self.update([scenes, eps_abs])
            return sig_geo
        else:
            # return sig_geo, D, [scenes, eps_abs]
            # return sig_geo, D, [scenes, eps_abs], hist_varibles
            return sig_geo, D, [scenes, eps_abs, H_3f]



    def update(self, scenes):
        self.scenes= scenes[0]
        self.eps_abs = scenes[1]
