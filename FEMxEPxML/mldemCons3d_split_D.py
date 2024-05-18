import numpy as np
import torch
from FEMxDEM.simDEM import *
from FEMxEPxML.constitutive import ConstitutiveMask
from FEMxML.utils_ml import get_q_2d, get_p_2d
from utilSelf.general import echo
from FEMxDEM.utiles_H_3D import *
from multiprocessing import Pool

class MlDemConstitutive(ConstitutiveMask):
    def __init__(self, p0, numg, nump, NN_sig, save_path, rho,  NN_Dv=None, NN_Dr=None, ndim=3, explicitFlag=False,
                 input_features='epsANDplast', H_initial = 137191.5155781979):

        ConstitutiveMask.__init__(self, p0=p0,save_path=save_path, ndim=ndim, explicitFlag=explicitFlag,
            name='mldem3d', numg=numg, nump=nump, cons=None, rho=rho)

        self.numg = numg
        self.nump = nump
        self.voigt_len = 3 if self.ndim == 2 else 6
        self.NN_sig = NN_sig

        self.NN_Dv = NN_Dv
        self.NN_Dr = NN_Dr
        self.eps = np.zeros([self.numg, self.voigt_len])
        self.eps_abs = np.zeros([self.numg, self.voigt_len])
        self.H_3D = np.zeros([self.numg, 3])
        # self.sig_vector = np.array([[-p0, 1000, 1000, -p0, 1000, -p0] for _ in range(self.numg)])
        self.sig_vector = np.array([[-p0, 0, 0, -p0, 0, -p0] for _ in range(self.numg)])
        self.input_features = input_features
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
        # deps_s_voigt = -np.delete((deps.reshape(self.numg, -1)), [3, 6, 7], axis=1)
        # eps_s = self.eps + deps_s_voigt
        # eps_abs = self.eps_abs + np.abs(deps_s_voigt)
        # r_deps = eps_abs.reshape(self.numg, -1)

        deps_s_voigt = -np.delete((deps.reshape(self.numg, -1)), [3, 6, 7], axis=1)
        eps_s = self.eps + deps_s_voigt
        deps_abs = np.abs(deps_s_voigt)
        r_deps = deps_abs.reshape(self.numg, -1)


        if self.nump > 1:
            with Pool(processes=self.nump) as pool:
                H1 = pool.map(H_vars_1, list(zip(r_deps[:, 0:1],r_deps[:, 1:2],r_deps[:, 2:3],r_deps[:, 3:4],r_deps[:, 4:5],r_deps[:, 5:6])))
                H2 = pool.map(H_vars_2, list(zip(r_deps[:, 0:1],r_deps[:, 3:4],r_deps[:, 5:6])))
                H3 = pool.map(H_vars_3, list(zip(r_deps[:, 0:1],r_deps[:, 1:2],r_deps[:, 2:3],r_deps[:, 3:4],r_deps[:, 4:5],r_deps[:, 5:6])))
                # H_3D = np.array(list(zip(H1, H2, H3)))
                # H_3d = np.array(list(zip(H1, H2, H3)))
                # H_2d = np.array(list(zip(H1, H2)))
                His_3D = self.H_3D + np.array(list(zip(H1, H2, H3)))

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
            input_vector = np.concatenate((eps_s, His_3D,), axis=1)
        elif self.input_features == 'epsAND2d':
            input_vector = np.concatenate((eps_s, His_3D,), axis=1)

        else:
            echo('No input_features as: %s' % self.input_features)
            raise
        if self.input_features == 'epsANDH' or self.input_features == 'epsANDqH' or self.input_features=='epsANDpqH':
            input_normed = self.NN_sig.normalization(torch.tensor(input_vector, dtype=torch.float))
            temp_normed = self.NN_sig(input_normed)
            temp = self.NN_sig.re_normalization(temp_normed).detach().numpy()
            sig_vector, H_1 = temp[:, :3], temp[:, 3:]
            Sences = [eps_s, self.eps_abs + np.abs(deps_s_voigt), sig_vector, H_1]

        else:
            # hist_3d = input_vector[:, 6:]
            input_normed = self.NN_sig.normalization(torch.tensor(input_vector, dtype=torch.float))
            temp_normed = self.NN_sig(input_normed)
            sig_vector = self.NN_sig.re_normalization(temp_normed).detach().numpy()
            # sences = [eps_s, self.eps_abs + np.abs(deps_s_voigt), H_3d, sig_vector]
            Sences = [eps_s, His_3D, self.eps_abs + np.abs(deps_s_voigt),  sig_vector]

        sig_geo = - self.assemble_sig_ml_3d(sig_vector=sig_vector)

        if self.explicitFlag:
            self.update(s_data=Sences)
            return sig_geo
        else:
            input_normed_Dv = self.NN_Dv.normalization(torch.tensor(input_vector, dtype=torch.float))
            temp_normed_Dv = self.NN_Dv(input_normed_Dv)
            Dv_vector = self.NN_Dv.re_normalization(temp_normed_Dv).detach().numpy()

            input_normed_Dr = self.NN_Dr.normalization(torch.tensor(input_vector, dtype=torch.float))
            temp_normed_Dr = self.NN_Dr(input_normed_Dr)
            Dr_vector = self.NN_Dr.re_normalization(temp_normed_Dr).detach().numpy()

            D = self.assemble_D_ml_3d(Dv_vector=Dv_vector, Dr_vector = Dr_vector)
            return sig_geo, D, Sences

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

    def assemble_D_ml_3d(self, Dv_vector, Dr_vector):
        D_tensor = np.zeros([self.numg, 3, 3, 3, 3])
        for i in range(self.numg):
            D_tensor[i,0,0,0,0] = Dv_vector[i, 0]
            D_tensor[i,0,0,1,1] = D_tensor[i,1,1,0,0] = Dv_vector[i, 1]
            D_tensor[i,0,0,2,2] = D_tensor[i,2,2,0,0] = Dv_vector[i, 2]
            D_tensor[i,0,0,1,2] = D_tensor[i,0,0,2,1] = D_tensor[i,1,2,0,0] = D_tensor[i,2,1,0,0] = Dr_vector[i, 1]
            D_tensor[i,0,0,0,2] = D_tensor[i,0,0,2,0] = D_tensor[i,0,2,0,0] = D_tensor[i,2,0,0,0] = Dr_vector[i, 2]
            D_tensor[i,0,0,0,1] = D_tensor[i,0,0,1,0] = D_tensor[i,0,1,0,0] = D_tensor[i,1,0,0,0] = Dr_vector[i, 0]

            D_tensor[i,1,1,1,1] = Dv_vector[i, 4]
            D_tensor[i,1,1,2,2] = D_tensor[i,2,2,1,1] = Dv_vector[i, 5]
            D_tensor[i,1,1,1,2] = D_tensor[i,1,1,2,1] = D_tensor[i,1,2,1,1] = D_tensor[i,2,1,1,1] = Dr_vector[i, 7]
            D_tensor[i,1,1,0,2] = D_tensor[i,1,1,2,0] = D_tensor[i,0,2,1,1] = D_tensor[i,2,0,1,1] = Dr_vector[i, 8]
            D_tensor[i,1,1,0,1] = D_tensor[i,1,1,1,0] = D_tensor[i,0,1,1,1] = D_tensor[i,1,0,1,1] = Dr_vector[i, 3]

            D_tensor[i,2,2,2,2] = Dv_vector[i, 8]
            D_tensor[i,2,2,1,2] = D_tensor[i,2,2,2,1] = D_tensor[i,1,2,2,2] = D_tensor[i,2,1,2,2] = Dr_vector[i, 10]
            D_tensor[i,2,2,0,2] = D_tensor[i,2,2,2,0] = D_tensor[i,0,2,2,2] = D_tensor[i,2,0,2,2] = Dr_vector[i, 11]
            D_tensor[i,2,2,0,1] = D_tensor[i,2,2,1,0] = D_tensor[i,0,1,2,2] = D_tensor[i,1,0,2,2] = Dr_vector[i, 6]

            D_tensor[i,1,2,1,2] = D_tensor[i,1,2,2,1] = D_tensor[i,2,1,1,2] = D_tensor[i,2,1,2,1] = Dv_vector[i, 6]
            D_tensor[i,1,2,0,2] = D_tensor[i,1,2,2,0] = D_tensor[i,2,1,0,2] = D_tensor[i,2,1,2,0] = D_tensor[i,0,2,1,2] = D_tensor[i,2,0,1,2] \
                = D_tensor[i,0,2,2,1] = D_tensor[i,2,0,2,1] = Dr_vector[i, 9]

            D_tensor[i,1,2,0,1] = D_tensor[i,1,2,1,0] = D_tensor[i,2,1,0,1] = D_tensor[i,2,1,1,0] = D_tensor[i,0,1,1,2] = D_tensor[i,1,0,1,2] \
                = D_tensor[i,0,1,2,1] = D_tensor[i,1,0,2,1] = Dr_vector[i, 4]

            D_tensor[i,0,2,0,2] = D_tensor[i,2,0,0,2] = D_tensor[i,0,2,2,0] = D_tensor[i,2,0,2,0] = Dv_vector[i, 7]
            D_tensor[i,0,2,0,1] = D_tensor[i,0,2,1,0] = D_tensor[i,2,0,0,1] = D_tensor[i,2,0,1,0] = D_tensor[i,0,1,0,2] = D_tensor[i,1,0,0,2] \
                = D_tensor[i,0,1,2,0] = D_tensor[i,1,0,2,0] = Dr_vector[i, 5]

            D_tensor[i,0,1,0,1] = D_tensor[i,0,1,1,0] = D_tensor[i,1,0,0,1] = D_tensor[i,1,0,1,0] = Dv_vector[i, 3]

        return D_tensor

    def update(self, s_data):
        self.eps = s_data[0]
        self.H_3D = s_data[1]
        self.sig_vector = s_data[3]
        if self.input_features == 'epsANDH' or self.input_features == 'epsANDqH' or self.input_features=='epsANDpqH':
            self.H = s_data[3]

    def return2initial(self):
        self.eps = np.zeros([self.numg, self.voigt_len])
        self.eps_abs = np.zeros([self.numg, self.voigt_len])
        self.sig_vector = np.array([[-self.p0, 0, -self.p0] for _ in range(self.numg)])