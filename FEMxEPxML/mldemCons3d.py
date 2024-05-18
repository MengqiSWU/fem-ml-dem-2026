import numpy as np
import torch
from FEMxDEM.simDEM import *
from FEMxEPxML.constitutive import ConstitutiveMask
from FEMxML.utils_ml import get_q_2d, get_p_2d
from utilSelf.general import echo
from FEMxDEM.utiles_H_3D import *
from multiprocessing import Pool

class MlDemConstitutive(ConstitutiveMask):
    def __init__(self, p0, numg, nump, NN_sig, save_path, rho, NN_D=None,  ndim=3, explicitFlag=False,
                 input_features='epsANDplast', H_initial = 137191.5155781979):
        ConstitutiveMask.__init__(self, p0=p0,save_path=save_path, ndim=ndim, explicitFlag=explicitFlag,
            name='mldem3d', numg=numg, nump=nump, cons=None, rho=rho)

        self.numg = numg
        self.nump = nump
        self.voigt_len = 3 if self.ndim == 2 else 6
        self.NN_sig, self.NN_D = NN_sig, NN_D
        self.eps = np.zeros([self.numg, self.voigt_len])
        self.eps_abs = np.zeros([self.numg, self.voigt_len])
        self.H_3d = np.zeros([self.numg, 3])
        self.input_features = input_features

        # self.sig_vector = np.array([[-p0, 1000, 1000, -p0, 1000, -p0] for _ in range(self.numg)])
        # self.sig_vector = np.array([[-p0, 0, 0, -p0, 0, -p0] for _ in range(self.numg)])
        # self.sig_vector = np.array([[-p0, -1239, 1316, -p0, -1201, -p0] for _ in range(self.numg)])

        self.initLoad = initLoad3D
        self.getStressAndTangent = getStressAndTangent3D
        self.shear = shear3D
        self.sig = np.zeros([self.numg, self.ndim, self.ndim])
        self.D = np.zeros([self.numg, self.ndim, self.ndim, self.ndim, self.ndim])

        with Pool(processes=self.nump) as pool:
            self.scenes = pool.map(self.initLoad, list(range(self.numg)))
            st = pool.map(self.getStressAndTangent, self.scenes)
        for i in range(self.numg):
        # Caution: all of the sig and eps saved in the Cons are in geo-mechanical form
            self.sig[i] = -np.array(st[i][0])
            self.D[i] = np.array(st[i][1])



        # if 'epsANDH' in self.input_features or 'epsANDqH' in self.input_features or self.input_features=='epsANDpqH':
        #     self.H = np.array([H_initial]*self.numg).reshape([self.numg, 1])
        # if self.explicitFlag:
        #     self.sig = self.solver(deps=np.zeros([self.numg, self.ndim, self.ndim]))
        # else:
        #     self.sig, self.D, _ = self.solver(deps=np.zeros([self.numg, self.ndim, self.ndim]))





    def solver(self, deps):
        '''
            deps_s: in shape of [numg, 2, 2]
            return: sig_geo
        '''
        # NOTE: revert cause compression is negative in ML model


        deps_s_voigt = -np.delete((deps.reshape(self.numg, -1)), [3, 6, 7], axis=1)
        eps_s = self.eps + deps_s_voigt
        eps_abs = self.eps_abs + np.abs(deps_s_voigt)
        r_deps = eps_abs.reshape(self.numg, -1)

        # eps_s = -np.delete((deps.reshape(self.numg, -1)), [3, 6, 7], axis=1)
        # eps_abs = self.eps_abs + np.abs(eps_s)
        # r_deps = eps_abs.reshape(self.numg, -1)

        if self.nump > 1:
            with Pool(processes=self.nump) as pool:
                H1 = pool.map(H_vars_1, list(zip(r_deps[:, 0:1],r_deps[:, 1:2],r_deps[:, 2:3],r_deps[:, 3:4],r_deps[:, 4:5],r_deps[:, 5:6])))
                H2 = pool.map(H_vars_2, list(zip(r_deps[:, 0:1],r_deps[:, 3:4],r_deps[:, 5:6])))
                H3 = pool.map(H_vars_3, list(zip(r_deps[:, 0:1],r_deps[:, 1:2],r_deps[:, 2:3],r_deps[:, 3:4],r_deps[:, 4:5],r_deps[:, 5:6])))
                # H_3D = np.array(list(zip(H1, H2, H3)))
                H_3d = np.array(list(zip(H1, H2, H3)))


        if self.input_features == 'eps':
            input_vector = eps_s
        elif self.input_features=='epsANDabsxy':
            input_vector = np.concatenate((eps_s, self.eps_abs[:, 0:1], self.eps_abs[:, 2:3]), axis=1)
        elif self.input_features == 'epsANDH':
            input_vector = np.concatenate((eps_s, self.H), axis=1)
        elif self.input_features == 'epsANDqH':
            input_vector = np.concatenate((eps_s, get_q_2d(self.sig_vector), self.H), axis=1)
        elif self.input_features == 'epsANDpqH':
            input_vector = np.concatenate((eps_s, get_p_2d(self.sig_vector), get_q_2d(self.sig_vector), self.H), axis=1)

        elif self.input_features == 'epsAND3d':
            input_vector = np.concatenate((eps_s, H_3d,), axis=1)

        else:
            echo('No input_features as: %s' % self.input_features)
            raise
        if self.input_features == 'epsANDH' or self.input_features == 'epsANDqH' or self.input_features=='epsANDpqH':
            input_normed = self.NN_sig.normalization(torch.tensor(input_vector, dtype=torch.float))
            temp_normed = self.NN_sig(input_normed)
            temp = self.NN_sig.re_normalization(temp_normed).detach().numpy()
            sig_vector, H_1 = temp[:, :3], temp[:, 3:]
            sences = [eps_s, self.eps_abs + np.abs(deps_s_voigt), sig_vector, H_1]

        else:
            # hist_3d = input_vector[:, 6:]
            input_normed = self.NN_sig.normalization(torch.tensor(input_vector, dtype=torch.float))
            temp_normed = self.NN_sig(input_normed)
            sig_vector = self.NN_sig.re_normalization(temp_normed).detach().numpy()
            sences = [eps_s, self.eps_abs + np.abs(deps_s_voigt), H_3d, sig_vector]

        sig_geo = - self.assemble_sig_ml_3d(sig_vector=sig_vector)

        if self.explicitFlag:
            self.update(scenes=sences)
            return sig_geo
        else:
            input_normed = self.NN_D.normalization(torch.tensor(input_vector, dtype=torch.float))
            temp_normed = self.NN_D(input_normed)
            D_vecror = self.NN_D.re_normalization(temp_normed).detach().numpy()
            D = self.assemble_D_ml_3d(D_vector=D_vecror)
            return sig_geo, D, sences

    def assemble_sig_ml_3d(self, sig_vector):
        sig_tensor = np.zeros([self.numg, 3, 3])
        for i in range(self.numg):
            sig_tensor[i, 0, 0] = sig_vector[i, 0]
            sig_tensor[i, 0, 1] = sig_tensor[i, 1, 0] = sig_vector[i, 1]
            sig_tensor[i, 0, 2] = sig_tensor[i, 2, 0] = sig_vector[i, 2]
            sig_tensor[i, 1, 1] = sig_vector[i, 3]
            sig_tensor[i, 1, 2] = sig_tensor[i, 2, 1] = sig_vector[i, 4]
            sig_tensor[i, 2, 2] = sig_vector[i, 5]
        return sig_tensor

    def assemble_D_ml_3d(self, D_vector):
        D_tensor = np.zeros([self.numg, 3, 3, 3, 3])
        for i in range(self.numg):
            D_tensor[i,0,0,0,0] = D_vector[i, 0]
            D_tensor[i,0,0,1,1] = D_tensor[i,1,1,0,0] = D_vector[i, 2]
            D_tensor[i,0,0,2,2] = D_tensor[i,2,2,0,0] = D_vector[i, 5]
            D_tensor[i,0,0,1,2] = D_tensor[i,0,0,2,1] = D_tensor[i,1,2,0,0] = D_tensor[i,2,1,0,0] = D_vector[i, 3]
            D_tensor[i,0,0,0,2] = D_tensor[i,0,0,2,0] = D_tensor[i,0,2,0,0] = D_tensor[i,2,0,0,0] = D_vector[i, 4]
            D_tensor[i,0,0,0,1] = D_tensor[i,0,0,1,0] = D_tensor[i,0,1,0,0] = D_tensor[i,1,0,0,0] = D_vector[i, 1]

            D_tensor[i,1,1,1,1] = D_vector[i, 11]
            D_tensor[i,1,1,2,2] = D_tensor[i,2,2,1,1] = D_vector[i, 14]
            D_tensor[i,1,1,1,2] = D_tensor[i,1,1,2,1] = D_tensor[i,1,2,1,1] = D_tensor[i,2,1,1,1] = D_vector[i, 12]
            D_tensor[i,1,1,0,2] = D_tensor[i,1,1,2,0] = D_tensor[i,0,2,1,1] = D_tensor[i,2,0,1,1] = D_vector[i, 13]
            D_tensor[i,1,1,0,1] = D_tensor[i,1,1,1,0] = D_tensor[i,0,1,1,1] = D_tensor[i,1,0,1,1] = D_vector[i, 7]

            D_tensor[i,2,2,2,2] = D_vector[i, 20]
            D_tensor[i,2,2,1,2] = D_tensor[i,2,2,2,1] = D_tensor[i,1,2,2,2] = D_tensor[i,2,1,2,2] = D_vector[i, 17]
            D_tensor[i,2,2,0,2] = D_tensor[i,2,2,2,0] = D_tensor[i,0,2,2,2] = D_tensor[i,2,0,2,2] = D_vector[i, 19]
            D_tensor[i,2,2,0,1] = D_tensor[i,2,2,1,0] = D_tensor[i,0,1,2,2] = D_tensor[i,1,0,2,2] = D_vector[i, 10]

            D_tensor[i,1,2,1,2] = D_tensor[i,1,2,2,1] = D_tensor[i,2,1,1,2] = D_tensor[i,2,1,2,1] = D_vector[i, 15]
            D_tensor[i,1,2,0,2] = D_tensor[i,1,2,2,0] = D_tensor[i,2,1,0,2] = D_tensor[i,2,1,2,0] = D_tensor[i,0,2,1,2] = D_tensor[i,2,0,1,2] \
                = D_tensor[i,0,2,2,1] = D_tensor[i,2,0,2,1] = D_vector[i, 16]

            D_tensor[i,1,2,0,1] = D_tensor[i,1,2,1,0] = D_tensor[i,2,1,0,1] = D_tensor[i,2,1,1,0] = D_tensor[i,0,1,1,2] = D_tensor[i,1,0,1,2] \
                = D_tensor[i,0,1,2,1] = D_tensor[i,1,0,2,1] = D_vector[i, 8]

            D_tensor[i,0,2,0,2] = D_tensor[i,2,0,0,2] = D_tensor[i,0,2,2,0] = D_tensor[i,2,0,2,0] = D_vector[i, 18]
            D_tensor[i,0,2,0,1] = D_tensor[i,0,2,1,0] = D_tensor[i,2,0,0,1] = D_tensor[i,2,0,1,0] = D_tensor[i,0,1,0,2] = D_tensor[i,1,0,0,2] \
                = D_tensor[i,0,1,2,0] = D_tensor[i,1,0,2,0] = D_vector[i, 9]

            D_tensor[i,0,1,0,1] = D_tensor[i,0,1,1,0] = D_tensor[i,1,0,0,1] = D_tensor[i,1,0,1,0] = D_vector[i, 6]

        return D_tensor

    def update(self, scenes):
        self.eps = scenes[0]
        self.eps_abs = scenes[1]
        self.sig_vector = scenes[3]
        if self.input_features == 'epsANDH' or self.input_features == 'epsANDqH' or self.input_features=='epsANDpqH':
            self.H = scenes[3]

    def return2initial(self):
        self.eps = np.zeros([self.numg, self.voigt_len])
        self.eps_abs = np.zeros([self.numg, self.voigt_len])
        self.sig_vector = np.array([[-self.p0, 0, -self.p0] for _ in range(self.numg)])