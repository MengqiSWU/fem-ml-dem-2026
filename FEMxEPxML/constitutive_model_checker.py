from csuhCons import csuh_single
from FEMxML.utils_ml import get_data_series
from utilSelf.general import get_dic_from_string, echo
from FEMxEPxML.utils_constitutive import voigt_2_tensor_high
import numpy as np
import matplotlib.pyplot as plt

csuh_dic = get_dic_from_string(s='kappa:1.906e-01 	 lambdaa:2.142e-01 	 N:1.931e+00 	 Z:2.743e-01 	 ocr:3.774e+02 	 theta_degree:1.329e+01')
cons = csuh_single(**csuh_dic, verboseFlag=True)

path_num = 0
data_paths = [
    # footing
    '../../simu/explicit/footing/footing_explicit_csuh_intorder1_numg254_footing303_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_timestep4.0e-04_b0.30',
]
numg = int(data_paths[path_num].split('numg')[1].split('_')[0])
returned_dict = get_data_series(root_path_list=data_paths[path_num:path_num + 1], maxTime=int(1e4), numg=numg,
                                explicit_flag=True)

sig = returned_dict['sig']
eps = returned_dict['eps']
sig_numg = voigt_2_tensor_high(voigt=sig, len_steps=290)
eps_numg = voigt_2_tensor_high(voigt=eps, len_steps=290)
deps_numg = np.zeros(shape=[len(eps_numg), len(eps_numg[0]), 2, 2])
for numg_temp in range(numg):
    deps_numg[numg_temp, 0] = eps_numg[numg_temp, 0]
    for step in range(1, len(eps_numg[0])):
        deps_numg[numg_temp, step] = eps_numg[numg_temp, step] - eps_numg[numg_temp, step - 1]
if np.linalg.norm(deps_numg[0, 0]) == 0.:
    sig_numg = sig_numg[:, 1:]
    eps_numg = eps_numg[:, 1:]
    deps_numg = deps_numg[:, 1:]

# for i in range(0, numg, 5):
#     echo("Gauss point %d" % i)
#     sig_cal = cons.prediction(deps_s=deps_numg[i])
#     # plt.plot(eps_numg[i, :, 1, 1])
#     # plt.show()
#     plt.plot(sig_cal[:, 1, 1])
#     plt.title('%d' % i)
#     plt.show()
#     plt.close()
#     print()

numg_index = 65
plt.plot(eps_numg[numg_index, :,  1, 1])
plt.show()
plt.plot(sig_numg[numg_index, :, 1, 1])
plt.show()
sig_cal = cons.prediction(deps_s=deps_numg[numg_index])
sig_simu = sig_numg[numg_index]
plt.plot(sig_cal[:,  1, 1])
plt.plot(sig_simu[:,  1, 1])
plt.show()
print()

