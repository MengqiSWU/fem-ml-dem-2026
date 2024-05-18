import copy
import numpy as np
from yadeimport import *
# from FEMxDEM.simDEM import *
from FEMxDEM.simDEM import initLoad, initLoad3D, getStressAndTangent3D, shear3D
from FEMxDEM.utiles_H_3D import *
from FEMxEPxML.constitutive import ConstitutiveMask
from utilSelf.saveGauss import saveGauss2D
from multiprocessing import Pool

class demConstitutive(ConstitutiveMask):
    def __init__(self, p0, ndim, explicitFlag, numg, save_path, save_flag, nump):
        ConstitutiveMask.__init__(self, p0=p0, ndim=ndim, explicitFlag=explicitFlag, numg=numg, save_path=save_path,
             name='dem3d', cons=None, save_flag=save_flag, nump=nump)
        self.numg = numg
        self.nump = nump
        self.initLoad = initLoad3D
        self.getStressAndTangent = getStressAndTangent3D
        self.shear = shear3D
        self.sig = np.zeros([self.numg, self.ndim, self.ndim])
        self.D = np.zeros([self.numg, self.ndim, self.ndim, self.ndim, self.ndim])
        self.H_3D = np.zeros([self.numg, 3])



        # # # # # # # # # # # # # # 需要修改getcon函数后才能使用 # # # # # # # # # # # # # #
        # if self.nump > 1:
        #     with Pool(processes=self.nump) as pool:
        #         self.scenes = pool.map(self.initLoad, list(range(self.numg)))
        #         st = pool.map(self.getStressAndTangent, self.scenes)
        #         # scenes = pool.map(initLoad3D, list(range(self.numg)))
        #         # st = pool.map(getStressAndTangent3D, scenes)
        #         # self.scenes = scenes
        # else:
        #     self.scenes = [self.initLoad(i) for i in range(self.numg)]
        #     st = [self.getStressAndTangent(self.scenes[i]) for i in range(self.numg)]
        # for i in range(self.numg):
        # # Caution: all of the sig and eps saved in the Cons are in geo-mechanical form
        #     self.sig[i] = -np.array(st[i][0])
        #     self.D[i] = np.array(st[i][1])


        with Pool(processes=self.nump) as pool:
            scenes_initial = pool.map(self.initLoad, list(range(self.numg)))
            st = pool.map(getStressAndTangent3D, scenes_initial)
            # pool.close()
            # pool.join()
        self.scenes = scenes_initial

        for i in range(self.numg):
        # Caution: all of the sig and eps saved in the Cons are in geo-mechanical form
            self.sig[i] = -np.array(st[i][0])
            self.D[i] = np.array(st[i][1])
        # # # # # # # # # # # # # # 需要修改getcon函数后才能使用 # # # # # # # # # # # # # #



    # def solver(self, deps):
        # sig = np.zeros([self.numg, self.ndim, self.ndim])
        # deps_abs = np.abs(deps)
        # D = np.zeros([self.numg, self.ndim, self.ndim, self.ndim, self.ndim])
        # r_deps = np.delete((deps_abs.reshape(self.numg, -1)), [3,6,7], axis=1)
        # scenes = self.scenes

        # NOTE: revert because compression is negative in DEM model
        # if self.nump > 1:
        #     with Pool(processes=self.nump) as pool:
        #         # Scenes = pool.map(self.shear, list(zip(self.scenes, -deps.reshape(self.numg, -1))))
        #         # St = pool.map(getStressAndTangent3D, Scenes)
        #
        #         # Scenes = pool.map(shear3D, list(zip(self.scenes, -deps.reshape(self.numg, -1))))
        #         # St = pool.map(getStressAndTangent3D, Scenes)
        #
        #         H1 = pool.map(H_vars_1, list(zip(r_deps[:, 0:1],r_deps[:, 1:2],r_deps[:, 2:3],r_deps[:, 3:4],r_deps[:, 4:5],r_deps[:, 5:6])))
        #         H2 = pool.map(H_vars_2, list(zip(r_deps[:, 0:1],r_deps[:, 3:4],r_deps[:, 5:6])))
        #         H3 = pool.map(H_vars_3, list(zip(r_deps[:, 0:1],r_deps[:, 1:2],r_deps[:, 2:3],r_deps[:, 3:4],r_deps[:, 4:5],r_deps[:, 5:6])))
        #         # H_3d = np.array(list(zip(H1, H2, H3)))
        #         pool.close()
        #         pool.join()
        #
        # else:
        #     Scenes = []
        #     St = []
        #     for i in range(self.numg):
        #         temp = self.shear([self.scenes[i], -deps[i].reshape(-1)])
        #         Scenes.append(temp)
        #         St.append(self.getStressAndTangent(temp))
        # His_3D = self.H_3D + np.array(list(zip(H1, H2, H3)))

        # for i in range(self.numg):
        #     sig[i] = np.array(St[i][0])
        #     D[i] = np.array(St[i][1])
        # sig_geo = -sig
        #
        # if self.explicitFlag:
        #     # self.update([scenes, eps_abs])  #2D
        #     self.update([Scenes])  #3D
        #     return sig_geo
        # else:
        #     # return sig_geo, D, [scenes, eps_abs, H_3f]  #2D
        #     # return sig_geo, D, [scenes, eps_abs, H_3d]
        #     return sig_geo, D, [Scenes, His_3D]



    def solver(self, deps):
        sig = np.zeros([self.numg, self.ndim, self.ndim])
        deps_abs = np.abs(deps)
        D = np.zeros([self.numg, self.ndim, self.ndim, self.ndim, self.ndim])
        r_deps = np.delete((deps_abs.reshape(self.numg, -1)), [3,6,7], axis=1)
        scenes = self.scenes

        with Pool(processes=self.nump) as pool:
            Scenes = pool.map(self.shear, list(zip(scenes, -deps.reshape(self.numg, -1))))
            St = pool.map(getStressAndTangent3D, Scenes)
            H1 = pool.map(H_vars_1, list(zip(r_deps[:, 0:1],r_deps[:, 1:2],r_deps[:, 2:3],r_deps[:, 3:4],r_deps[:, 4:5],r_deps[:, 5:6])))
            H2 = pool.map(H_vars_2, list(zip(r_deps[:, 0:1],r_deps[:, 3:4],r_deps[:, 5:6])))
            H3 = pool.map(H_vars_3, list(zip(r_deps[:, 0:1],r_deps[:, 1:2],r_deps[:, 2:3],r_deps[:, 3:4],r_deps[:, 4:5],r_deps[:, 5:6])))
            pool.close()
            pool.join()

        His_3D = self.H_3D + np.array(list(zip(H1, H2, H3)))

        for i in range(self.numg):
            sig[i] = np.array(St[i][0])
            D[i] = np.array(St[i][1])
        sig_geo = -sig

        if self.explicitFlag:
            # self.update([scenes, eps_abs])  #2D
            self.update([Scenes])  #3D
            return sig_geo
        else:
            return sig_geo, D, [Scenes, His_3D]


    def update(self, s_data):
        self.scenes= s_data[0]
        self.H_3D = s_data[1]
