import copy
from multiprocessing import Pool
import numpy as np
from FEMxEPxML.constitutive import ConstitutiveMask
from FEMxEPxML.mldemCons import MlDemConstitutive
from FEMxML.utils_ml import error_evaluate
from utilSelf.saveGauss import save_loading
from utilSelf.general import  echo
from FEMxEPxML.utils_constitutive import voigt_2_tensor
from FEMxML.torch_net import Net
import torch


class MixedConstitutive(ConstitutiveMask):
    def __init__(self, save_path, NN_sig,
                 input_features, p0, explicitFlag, numg, pool: Pool, rho, kwargs,
                 ndim=3, tol=0.5, x_name='vonmises'):
        if x_name == 'mldem':
            name = '2ml'
        else:
            name = 'mixed'
        ConstitutiveMask.__init__(
            self, p0, ndim=ndim, explicitFlag=explicitFlag, numg=numg, cons=None, pool=pool, name=name, save_path=save_path)
        self.ml_model = MlDemConstitutive(numg=numg, NN_sig=NN_sig, p0=p0, explicitFlag=explicitFlag,
                                          input_features=input_features, save_path=save_path, rho=rho)
        if 'vonmises' in x_name:
            from FEMxEPxML.vonmisesCons import vonmisesConstitutive
            self.x_model = vonmisesConstitutive(
                explicitFlag=explicitFlag, numg=numg, pool=pool,
                p0=kwargs['p0'], poisson=kwargs['poisson'], E=kwargs['E'], rho=kwargs['rho'],
                verboseFlag=False, ndim=ndim, save_path=kwargs['save_path'])
        elif 'csuh' in x_name:
            from FEMxEPxML.csuhCons import csuhConstitutive
            self.x_model = csuhConstitutive(
            explicitFlag=explicitFlag, ndim=ndim,  rho=kwargs['rho'],
            p0 = kwargs['p0'], numg=numg, pool=pool, save_path=kwargs['save_path'], **kwargs['csuh_dic'])

        elif 'mldem' in x_name:
            # nn_name = 'X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_2ml'
            nn_name = 'X_epsANDH_Y_sigANDH_dd20_Fourier_noRotate_von_all_2ml'
            NN_sig = torch.load(
            './FEMxML/biax_ml_1e5/%s/entire_model.pt' % nn_name,
            map_location=torch.device('cpu'))
            self.x_model = MlDemConstitutive(NN_sig=NN_sig, NN_D=None, explicitFlag=explicitFlag, numg=numg, rho=kwargs['rho'],
                                 input_features=kwargs['input_features'], save_path=kwargs['save_path'])
        else:
            echo()

        self.tol = tol
        self.save_path = save_path
        self.t = 0
        self.sig, _ = self.solver(deps=np.zeros(shape=[self.numg, 2, 2]))

    def solver(self, deps):
        step_internal = 20
        if self.t % step_internal == 0 or self.t == 1:
            if self.x_model.name != 'mldem':
                sig_geo_0 = np.array([self.x_model.cons[i].sig for i in range(self.numg)])
                eps_abs_0 = np.array([self.x_model.cons[i].eps_abs for i in range(self.numg)])
                if self.x_model.name == 'vonmises':
                    H_0 = np.array([self.x_model.cons[i].H for i in range(self.numg)])
            else:  # mldem
                sig_geo_0 = -self.x_model.sig
                eps_abs_0 = voigt_2_tensor(self.x_model.eps_abs)
        sig_geo_ml = self.ml_model.solver(deps=deps)
        sig_geo_ml_temp = copy.deepcopy(sig_geo_ml)
        sig_geo_x = self.x_model.solver(deps=deps)
        if self.x_model.name != 'mldem':
            index_large_error = []
            for i in range(self.numg):
                relative_error = error_evaluate(t_true=sig_geo_x[i], t_pre=sig_geo_ml[i])
                if np.max(relative_error) > self.tol:
                    sig_geo_ml[i] = sig_geo_x[i, :self.ndim, :self.ndim]
                    index_large_error.append(i)
            if (self.t % step_internal == 0 or self.t == 1) and len(index_large_error) > 0:
                eps = -np.array([self.x_model.cons[i].eps[:self.ndim, :self.ndim] for i in index_large_error])
                kwargs = {
                    "numg_index": index_large_error,
                    'sig_last': -sig_geo_0[index_large_error, :self.ndim, :self.ndim],
                    'eps': eps,
                    'eps_abs': eps_abs_0[index_large_error, :self.ndim, :self.ndim],
                    'sig': -sig_geo_x[index_large_error, :self.ndim, :self.ndim],
                    'sig_err': -sig_geo_ml_temp[index_large_error, :self.ndim, :self.ndim],
                }
                if self.x_model.name == 'vonmises':
                    kwargs['H_0'] = [H_0[i] for i in index_large_error]
                    kwargs['H_1'] = [self.x_model.cons[i].H for i in index_large_error]
                elif self.x_model.name == 'csuh':
                    pass
                save_loading(
                    save_path=self.save_path, t=self.t, special_str='%d' % len(index_large_error), **kwargs)
            self.t += 1
            return sig_geo_ml, np.abs(sig_geo_ml_temp-sig_geo_x[:, :self.ndim, :self.ndim])
        else:
            self.t += 1
            return 0.5*(sig_geo_ml+sig_geo_x), np.abs(sig_geo_ml-sig_geo_x)

