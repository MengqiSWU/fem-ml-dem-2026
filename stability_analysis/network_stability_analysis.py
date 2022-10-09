import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from FEMxEPxML.csuhCons import csuhConstitutive
from FEMxEPxML.mldemCons import MlDemConstitutive
from FEMxEPxML.utils_constitutive import voigt_2_tensor
from FEMxEPxML.vonmisesCons import vonmisesConstitutive
from FEMxML.utils_ml import get_data_series
from sciptes4figures.utils_plot import configurations, get_color_list
from utilSelf.general import check_mkdir
font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
color_list = get_color_list()

"""

    This file is used to display the process of the NN model gradually losing stability which is used 
        to support our opinion that the NN model cannot make sure to give the stable stress once there 
        is a unpredictable disturbance in the input.
        
    The prediction show that once the noise comes to 0.5, the predictions begin to diverite from the 
        vonmises model calculation results.

"""


# -------------------------------------------------------
#           plot prediction
def plot_prediction(
        sig_pre, sig_simu, sig_vonmises, sig_pre_1,
                    numg_list, save_path, step_num, noise=0.):
    for i in numg_list:
        plot_index = np.arange(0, step_num, int(step_num / 20))
        # plt.plot(np.arange(0, step_num), -sig[i, :, 2]/1e3, label='von-mises')
        plt.scatter(plot_index, -sig_simu[i, :, 2][plot_index] / 1e3, c=color_list[0], label='Training data', s=50)
        plt.plot(np.arange(0, step_num), -sig_vonmises[i, :, 2] / 1e3, c=color_list[1], label='IVH', linewidth=2)
        plt.plot(np.arange(0, step_num), -sig_pre[i, :, 2] / 1e3, c=color_list[2], label='$\mathcal{NN}\ 1$', linewidth=2)
        plt.plot(np.arange(0, step_num), -sig_pre_1[i, :, 2] / 1e3, c=color_list[3], label='$\mathcal{NN}\ 2$', linewidth=2)
        plt.ylabel('kPa', fontdict=font_4)
        plt.xlabel('Loading step', fontdict=font_4)
        plt.title('Noise %.2f' % noise, fontdict=font_4)
        plt.legend(**legendDic)
        plt.tick_params(**tickParamsDic)
        plt.tight_layout()
        dir_name = os.path.join(save_path, "Noise_%.2f" % noise)
        check_mkdir(dir_name)
        fname = os.path.join(dir_name, "origin_prediction_numg%d_noise%.2f.png" % (i, noise))
        plt.savefig(fname)
        plt.close()


# -------------------------------------------------------
#               prediction
def prediction_plot(deps, noise):
    if noise > 0:
        deps = deps + noise * np.random.random(size=deps.shape) * deps
    sig_pre = np.zeros_like(sig)
    sig_pre_1 = np.zeros_like(sig)
    # sig_csuh = np.zeros_like(sig)
    sig_vonmises = np.zeros_like(sig)
    for step in range(step_num):
        sig_geo = cons_ml.solver(deps=-voigt_2_tensor(deps[:, step]))
        sig_geo_1 = cons_ml_1.solver(deps=-voigt_2_tensor(deps[:, step]))
        sig_pre[:, step, :] = - np.delete(sig_geo_1.reshape(numg, 4), [2], axis=1)
        sig_pre_1[:, step, :] = - np.delete(sig_geo.reshape(numg, 4), [2], axis=1)
        # sig_geo_csuh = cons_csuh.solver(deps=-voigt_2_tensor(deps[:, step]))
        # sig_csuh[:, step, :] = - np.delete(sig_geo_csuh[:, :2, :2].reshape(numg, 4), [2], axis=1)
        sig_geo_vonmises = cons_vonmises.solver(deps=-voigt_2_tensor(deps[:, step]))
        sig_vonmises[:, step, :] = - np.delete(sig_geo_vonmises[:, :2, :2].reshape(numg, 4), [2], axis=1)

    cons_ml.return2initial()
    cons_ml_1.return2initial()
    cons_csuh.return2initial()
    cons_vonmises.return2initial()
    plot_prediction(
        sig_pre=sig_pre, sig_simu=sig, sig_vonmises=sig_vonmises, sig_pre_1=sig_pre_1,
        numg_list=range(0, numg, numg // 10), save_path=save_path, step_num=step_num,
        noise=noise)
    return


# -------------------------------------------------------
#              read data
data_file_list = [
    # biaixal -> misese
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5',
    # retaining -> mises
    # '../../simu/explicit/retaining/retaining_explicit_vonmises_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5',
    # footing -> mises
    '../../simu/explicit/footing/footing_explicit_vonmises_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5',
]
numg = int(data_file_list[0].split("numg")[1].split("_")[0])
data_dict = get_data_series(
    root_path_list=data_file_list, numg=numg, maxTime=int(1e5), explicit_flag=True, add_flag=False)

sig, eps, eps_abs = data_dict['sig'], data_dict['eps'], data_dict['eps_abs']
deps = np.zeros_like(eps)
deps[:, 0, :] = eps[:, 0, :]
deps[:, 1:, :] = eps[:, 1:, :] - eps[:, 0:-1, :]

step_num = len(sig[0])

# -------------------------------------------------------
#              initialize the model
explicitFlag = True
model_name = "../FEMxML/biax_ml_1e5/X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0/entire_model.pt"
model_name_1 = "../FEMxML/biax_ml_1e5/X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0_repeat4_stability_analysis/entire_model.pt"
input_features = "epsANDabsxy"
save_path = "stability_check%s" % input_features
check_mkdir(save_path)
kwargs = {
    "p0": 1e5,
    'rho': 2650.,
    "input_features": input_features,
    "save_path": save_path,
}
NN_sig = torch.load(model_name, map_location='cpu')
NN_sig_1 = torch.load(model_name_1, map_location='cpu')

cons_ml = MlDemConstitutive(
    p0=kwargs['p0'], NN_sig=NN_sig, NN_D=None, explicitFlag=explicitFlag, numg=numg, rho=kwargs['rho'],
    input_features=kwargs['input_features'], save_path=kwargs['save_path'])
cons_ml_1 = MlDemConstitutive(
    p0=kwargs['p0'], NN_sig=NN_sig_1, NN_D=None, explicitFlag=explicitFlag, numg=numg, rho=kwargs['rho'],
    input_features=kwargs['input_features'], save_path=kwargs['save_path'])

cons_csuh = csuhConstitutive(
    explicitFlag=explicitFlag, numg=numg, pool=None, save_path=save_path, rho=kwargs['rho'],
    p0=kwargs['p0'], ocr=120., theta_degree=30.,
    lambdaa=0.135, kappa=0.04,
    nu=0.3, N=1.973, m=1.8, Z=0.933938655)

cons_vonmises = vonmisesConstitutive(
    explicitFlag=explicitFlag, numg=numg, pool=None, save_path=save_path, rho=kwargs['rho'],
    p0=1e5, poisson=0.2, E=2e7, A=3e5, B=0.2, epsilon0=0.02, yield_stress0=1e3
)

# -------------------------------------------------------
#              prediction & plot
for noise_temp in [0., 0.2, 0.5, 1.0, 2.0]:
    prediction_plot(deps=deps, noise=noise_temp)

