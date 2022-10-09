import os.path

import numpy as np

from FEMxML.torch_restore import modelRestore
from FEMxML.utils_ml import get_data_series
from FEMxML.utils_ml import get_q_2d
from utilSelf.general import check_mkdir, echo
from utils_plot import plot_sig_series

"""
    This script is used to plot thr strain stress curve of the gauss points in the exFEM calculations.
"""

mlsimu_flag = True
pre_flag = True
path_num = 2  # 0 1 2
data_paths = [
    # biaxial
    '../../simu/explicit/biaxial/biaxial_rough_explicit_csuh_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_theta8_timestep2.0e-04',
    # retaining
    '../../simu/explicit/retaining/retaining_explicit_csuh_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_theta8_timestep4.0e-04',
    # footing
    '../../simu/explicit/footing/footing_explicit_csuh_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_theta8_timestep4.0e-04_b0.30',
]
data_paths_mlsimu = [
    # biaxial
    '../../simu/explicit/biaxial/biaxial_rough_explicit_mldem_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_6_timestep2.0e-04',
    # retaining
    '../../simu/explicit/retaining/retaining_explicit_mldem_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_6_timestep4.0e-04',
    # footing
    '../../simu/explicit/footing/footing_explicit_mldem_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_6_timestep4.0e-04_b0.30',
]
numg = int(data_paths[path_num].split('numg')[1].split('_')[0])
returned_dict = get_data_series(root_path_list=data_paths[path_num:path_num + 1], maxTime=int(1e4), numg=numg,
                                explicit_flag=True)
strain_true, stress_true, strain_abs_true, stress_last_true, H_0_true, H_1_true, tangent_true, = \
    returned_dict['eps'], returned_dict['sig'], \
    returned_dict['eps_abs'], \
    returned_dict['sig_last'] if 'sig_last' in returned_dict.keys() else None, \
    returned_dict['H_0'] if 'H_0' in returned_dict.keys() else None, \
    returned_dict['H_1'] if 'H_1' in returned_dict.keys() else None, \
    returned_dict['tangent'] if 'tangent' in returned_dict.keys() else None

save_path = os.path.join(data_paths[path_num], 'series_plot')
# RESTORE THE NETWORK MODEL
if pre_flag:
    model_path = '../FEMxML/biax_ml_1e5/X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_6'
    model = modelRestore(
        savedPath=model_path)
    input_features = model_path.split('X_')[1].split('_')[0]

# ml_simu
if mlsimu_flag:
    returned_dict = get_data_series(root_path_list=data_paths_mlsimu[path_num:path_num + 1], maxTime=int(1e4),
                                    numg=numg, explicit_flag=True)
    strain_mlsimu, stress_mlsimu, strain_abs_mlsimu, stress_last_mlsimu, H_0_mlsimu, H_1_mlsimu, tangent_mlsimu, = \
        returned_dict['eps'], returned_dict['sig'], \
        returned_dict['eps_abs'], \
        returned_dict['sig_last'] if 'sig_last' in returned_dict.keys() else None, \
        returned_dict['H_0'] if 'H_0' in returned_dict.keys() else None, \
        returned_dict['H_1'] if 'H_1' in returned_dict.keys() else None, \
        returned_dict['tangent'] if 'tangent' in returned_dict.keys() else None

check_mkdir(save_path)
for numg_temp in range(0, numg, 5):
    if pre_flag:
        if input_features == 'epsANDabsy':
            input = np.concatenate((strain_true[numg_temp], strain_abs_true[numg_temp][:, 2:3]), axis=1)
        elif input_features == 'epsANDabsxy':
            input = np.concatenate(
                (strain_true[numg_temp], strain_abs_true[numg_temp][:, 0:1],
                 strain_abs_true[numg_temp][:, 2:3]), axis=1)
        elif input_features == 'eps':
            input = strain_true[numg_temp]
        elif input_features == 'epsANDH':
            input = np.concatenate((strain_true[numg_temp], H_0_true[numg_temp]), axis=1)
        elif input_features == 'epsANDqH':
            input = np.concatenate((strain_true[numg_temp], get_q_2d(stress_last_true[numg_temp]), H_0_true[numg_temp]),
                                   axis=1)
        else:
            echo('Input feature %s is not included' % input_features)
            raise ValueError
    if pre_flag and mlsimu_flag:
        plot_sig_series(
            eps=strain_true[numg_temp], sig=stress_true[numg_temp],
            eps_simu=strain_mlsimu[numg_temp], sig_simu=stress_mlsimu[numg_temp],
            sig_pre=model.get_prediction(input),
            numg=numg_temp, prediction_save_path=save_path,
        legend_flag=True if numg_temp == 0 else False)
    elif pre_flag:
        plot_sig_series(
            eps=strain_true[numg_temp], sig=stress_true[numg_temp],
            sig_pre=model.get_prediction(input),
            numg=numg_temp, prediction_save_path=save_path,
        legend_flag=True if numg_temp == 0 else False)
    else:
        plot_sig_series(
            eps=strain_true[numg_temp], sig=stress_true[numg_temp],
            # eps_simu=strain_mlsimu[numg_temp], sig_simu=stress_mlsimu[numg_temp],
            numg=numg_temp, prediction_save_path=save_path,
        legend_flag=True if numg_temp == 0 else False, scatter_flag=False)
