import numpy as np
import torch
from FEMxEPxML.constitutive import ConstitutiveMask
from FEMxML.utils_ml import get_q_2d, get_p_2d
from utilSelf.general import echo


class MlDemConstitutive(ConstitutiveMask):
    def __init__(self, p0, numg, NN_sig, save_path, rho, NN_D=None,  ndim=2, explicitFlag=False,
                 input_features='epsANDplast', H_initial = 137191.5155781979):
        ConstitutiveMask.__init__(
            self, save_path=save_path, p0=p0, ndim=ndim, explicitFlag=explicitFlag,
            name='mldem', numg=numg, cons=None, pool=None, rho=rho)
        self.numg = numg
        self.voigt_len = 3 if self.ndim == 2 else 6
        self.eps = np.zeros([self.numg, self.voigt_len])
        self.eps_abs = np.zeros([self.numg, self.voigt_len])
        self.NN_sig, self.NN_D = NN_sig, NN_D
        self.sig_vector = np.array([[-p0, 0, -p0] for _ in range(self.numg)])
        self.input_features = input_features
        if 'epsANDH' in self.input_features or 'epsANDqH' in self.input_features or self.input_features=='epsANDpqH':
            self.H = np.array([H_initial]*self.numg).reshape([self.numg, 1])
        if self.explicitFlag:
            self.sig = self.solver(deps=np.zeros([self.numg, self.ndim, self.ndim]))
        else:
            self.sig, self.D, _ = self.solver(deps=np.zeros([self.numg, self.ndim, self.ndim]))

    def solver(self, deps):
        '''
            deps_s: in shape of [numg, 2, 2]
            return: sig_geo
        '''
        # NOTE: revert cause compression is negative in ML model
        deps_s_voigt = -np.delete(deps.reshape([self.numg, 4]), [2], axis=1)
        eps_s = self.eps + deps_s_voigt
        if self.input_features=='epsANDabsxy':
            input_vector = np.concatenate((
                eps_s,
                self.eps_abs[:, 0:1], self.eps_abs[:, 2:3]), axis=1)
        elif self.input_features =='epsANDabsxyq':
            input_vector = np.concatenate((
                eps_s,
                self.eps_abs[:, 0:1], self.eps_abs[:, 2:3], get_q_2d(self.sig_vector)), axis=1)
        elif self.input_features =='epsANDabsy':
            input_vector = np.concatenate((
                eps_s,
                self.eps_abs[:, 2:3]), axis=1)
        elif self.input_features =='epsANDsiglast':
            input_vector = np.concatenate((eps_s, self.sig_vector), axis=1)
        elif self.input_features =='epsANDplast':
            input_vector = np.concatenate((eps_s, get_p_2d(self.sig_vector)), axis=1)
        elif self.input_features =='epsANDpqlast':
            input_vector = np.concatenate((eps_s, get_p_2d(self.sig_vector), get_q_2d(self.sig_vector)), axis=1)
        elif self.input_features == 'eps':
            input_vector = eps_s
        elif self.input_features == 'epsANDH':
            input_vector = np.concatenate((eps_s, self.H), axis=1)
        elif self.input_features == 'epsANDqH':
            input_vector = np.concatenate((eps_s, get_q_2d(self.sig_vector), self.H), axis=1)
        elif self.input_features == 'epsANDpqH':
            input_vector = np.concatenate((eps_s, get_p_2d(self.sig_vector), get_q_2d(self.sig_vector), self.H), axis=1)
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
            input_normed = self.NN_sig.normalization(torch.tensor(input_vector, dtype=torch.float))
            temp_normed = self.NN_sig(input_normed)
            sig_vector = self.NN_sig.re_normalization(temp_normed).detach().numpy()
            sences = [eps_s, self.eps_abs + np.abs(deps_s_voigt), sig_vector]
        sig_geo = - self.assemble_sig_ml(sig_vector=sig_vector)
        if self.explicitFlag:
            self.update(scenes=sences)
            return sig_geo
        else:
            D_vecror = self.NN_D(torch.tensor(input_vector, dtype=torch.float)).detach().numpy()
            D = self.assemble_D_ml(D_vector=D_vecror)
            return sig_geo, D, sences

    def assemble_sig_ml(self, sig_vector):
        sig_tensor = np.zeros([self.numg, 2, 2])
        for i in range(self.numg):
            sig_tensor[i, 0, 0] = sig_vector[i, 0]
            sig_tensor[i, 0, 1] = sig_tensor[i, 1, 0] = sig_vector[i, 1]
            sig_tensor[i, 1, 1] = sig_vector[i, 2]
        return sig_tensor

    def assemble_D_ml(self, D_vector):
        D_tensor = np.zeros([self.numg, 2, 2, 2, 2])
        for i in range(self.numg):
            D_tensor[i, 0, 0, 0, 0] = D_vector[i, 0]
            D_tensor[i, 0, 1, 0, 0] = D_tensor[i, 0, 0, 0, 1] = D_tensor[i, 1, 0, 0, 0] = D_tensor[i, 0, 0, 1, 0] = \
            D_vector[i, 1]
            D_tensor[i, 1, 1, 0, 0] = D_tensor[i, 0, 0, 1, 1] = D_vector[i, 2]
            D_tensor[i, 0, 1, 0, 1] = D_tensor[i, 0, 1, 1, 0] = D_tensor[i, 1, 0, 0, 1] = D_tensor[i, 1, 0, 1, 0] = \
            D_vector[i, 3]
            D_tensor[i, 1, 1, 0, 1] = D_tensor[i, 0, 1, 1, 1] = D_tensor[i, 1, 1, 1, 0] = D_tensor[i, 1, 0, 1, 1] = \
            D_vector[i, 4]
            D_tensor[i, 1, 1, 1, 1] = D_vector[i, 5]
        return D_tensor

    def update(self, scenes):
        self.eps = scenes[0]
        self.eps_abs = scenes[1]
        self.sig_vector = scenes[2]
        if self.input_features == 'epsANDH' or self.input_features == 'epsANDqH' or self.input_features=='epsANDpqH':
            self.H = scenes[3]

    def return2initial(self):
        self.eps = np.zeros([self.numg, self.voigt_len])
        self.eps_abs = np.zeros([self.numg, self.voigt_len])
        self.sig_vector = np.array([[-self.p0, 0, -self.p0] for _ in range(self.numg)])