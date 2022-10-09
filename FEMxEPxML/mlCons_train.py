import os.path
import random
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from FEMxEPxML.csuhCons import csuh_single
from FEMxEPxML.mlCons import mlCons_single  # mlcons
from FEMxEPxML.utils_constitutive import voigt_2_tensor_high
from FEMxML.utils_ml import get_data_series, findDevice
from utilSelf.general import writeLine, check_mkdir, get_dic_from_string

# from FEMxEPxML.mlcsuhCons import mlcsuh_single as mlCons_single  # ml-csuh
# from FEMxEPxML.mlvonmises_single import mlvonmises_single as mlCons_single  # ml-von-mises

start_time = time.time()
out_directory = './classical_model_train'
saved_directory = os.path.join(out_directory, './csuh_dem_train_test')

# -------------------------
# find the device
device = findDevice(useGPU=False)
check_mkdir(out_directory, saved_directory)

data_path_list = [
    # csuh
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_csuh_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_p100kPa_ocr_120.0',
    # von-mises
    # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5',
    # dem explicit
    '../../simu/explicit/footing/footing_explicit_dem_order1_numg480_footing552_vel0.40_damp1.0e+06_safe0.5',
    # dem implicit
    # '../../simu/footing/footing_dem_footing552_2D_order1_numG480_2ndShear',
]
confining = 1e5
if 'numg' in data_path_list[0]:
    numg = int(data_path_list[0].split('numg')[1].split('_')[0])
else:
    numg = int(data_path_list[0].split('numG')[1].split('_')[0])

returned_dict = get_data_series(
    root_path_list=data_path_list,
    maxTime=int(1e5),
    numg=numg, series_flag=True, explicit_flag=True)
sig = returned_dict['sig']
eps = returned_dict['eps']
sig_numg = voigt_2_tensor_high(voigt=sig, len_steps=40)
eps_numg = voigt_2_tensor_high(voigt=eps, len_steps=40)
deps_numg = np.zeros(shape=[len(eps_numg), len(eps_numg[0]), 2, 2])
for numg_temp in range(numg):
    deps_numg[numg_temp, 0] = eps_numg[numg_temp, 0]
    for step in range(1, len(eps_numg[0])):
        deps_numg[numg_temp, step] = eps_numg[numg_temp, step] - eps_numg[numg_temp, step - 1]
if np.linalg.norm(deps_numg[0, 0]) == 0.:
    sig_numg = sig_numg[:, 1:]
    eps_numg = eps_numg[:, 1:]
    deps_numg = deps_numg[:, 1:]

numg_used, deps_use, sig_true = [], [], []
for i in range(0, numg, 48):
    deps_temp = deps_numg[i]
    sig_temp = sig_numg[i]
    use_flag = True
    for sig in sig_temp:
        if np.sum(sig*sig) == 0.:
            use_flag = False
            break
    if use_flag:
        numg_used.append(i)
        deps_use.append(deps_temp)
        sig_true.append(sig_numg[i])
numg_used = np.array(numg_used, dtype=int)
deps_use = np.array(deps_use)
sig_true = torch.tensor(np.array(sig_true), dtype=torch.float32, device=device)

'''
Initial parameters for the csuh-based ml constitutive model
'''
param_dic = get_dic_from_string(
    'kappa:0.08	 lambdaa:0.135 	 N:1.9 	 Z:0.9	 ocr:120. 	 M:1.25'
)

# prediction before the optimization
original_save_path = os.path.join(saved_directory, 'origin_prediction')
check_mkdir(original_save_path)
mlcsuh_object_ = mlCons_single(p0=confining, ndim=2, device=device, **param_dic)
# mlcsuh_object_ = mlCons_single(p0=1e5, ndim=2)
mlprediction = mlcsuh_object_.prediction(deps_numg=deps_use).cpu()
# mlcsuh_object_.return2initial()
csuh_object_ = csuh_single(p0=confining, ndim=2, **param_dic)
for pic_num in range(len(numg_used)):
    plt.plot(sig_true[pic_num, :, 1, 1].cpu().numpy(), marker='o', label='Simu')
    prediction = csuh_object_.prediction(deps_s=deps_use[pic_num])
    csuh_object_ = csuh_single(p0=confining, ndim=2, **param_dic)
    plt.plot(prediction[:, 1, 1], marker='+', label='Cal')
    plt.plot(mlprediction[pic_num, :, 1, 1].detach().numpy(), label='Cal_ml')
    plt.title('Gauss %d' % numg_used[pic_num])
    plt.legend()
    plt.tight_layout()
    fname = 'epoch_%d_gauss_%d.png' % (-1, numg_used[pic_num])
    # plt.show()
    plt.savefig(os.path.join(original_save_path, fname))
    plt.close()

# initialize the ml-cons
mlcsuh_object = mlCons_single(p0=confining, ndim=2, device=device, **param_dic)
params_mlcsuh = [
    # csuh
    mlcsuh_object.kappa_log,
    mlcsuh_object.lambdaa_log,
    mlcsuh_object.N_log,
    mlcsuh_object.Z_log,
    mlcsuh_object.ocr_log,
    mlcsuh_object.M_log,
    # mlcsuh_object.nu_log,
    # mlcsuh_object.theta_degree_log,

    # vonmises
    # mlcsuh_object.E_log,
    # mlcsuh_object.nu,
    # mlcsuh_object.A_log,
    # mlcsuh_object.B,
    # mlcsuh_object.epsilon0,
    # mlcsuh_object.yield_stress0_log,
]
# params_name_list = ['kappa', 'lambda', 'm', 'N', 'Z', 'e0', 'M', 'nu']
params_name_list = [
    # csuh
    'kappa', 'lambdaa', 'N', 'Z', 'ocr', 'M',
    # 'theta_degree',
    # vonmises
    # 'E_log', 'nu', 'A_log', 'B', 'epsilon0', 'yield_stress0_log',
]
line = '#' * 80 + '\n' + '\t\t\tBegining...\n'
for num, param in enumerate(params_name_list):
    if param == 'ocr':
        temp = np.exp(params_mlcsuh[num].item() * np.log(param_dic[param]))
    else:
        temp = params_mlcsuh[num].item() * param_dic[param]
    line += ' %s:%.3e \t' % (param, temp)
line += '\n\n\n'
print(line)
writeLine(fname=os.path.join(saved_directory, 'history.dat'), mode='w', s=line)

optimizer = torch.optim.Adam(
    params=params_mlcsuh)
# optimizer = torch.optim.Adam(
#     params=mlcsuh_object.NN_h.parameters())

loss_operator = torch.nn.MSELoss()

for epoch in range(int(1e6) + 1):
    if len(numg_used) > 1:
        numg_used_index = np.array(random.choices(range(len(numg_used)), k=1))
    else:
        numg_used_index = np.arange(len(numg_used))
    deps_use_temp = deps_use[numg_used_index]

    def closure():
        # NOTE: if the deps = 0 included in the computation, there will return a gradient of NAN
        optimizer.zero_grad()
        prediction = mlcsuh_object.prediction(deps_numg=deps_use_temp[:, :, :, :])
        loss = loss_operator(sig_true[numg_used_index, :, :, :], prediction)/\
               torch.mean(sig_true[numg_used_index, :, :, :]*sig_true[numg_used_index, :, :, :])
        # print('\t loss %.3e' % loss.item())
        loss.backward()
        return loss

    optimizer.step(closure)

    # NOTE: Caution: this is really important !!!!!
    mlcsuh_object.return2initial()

    if epoch % int(100) == 0:
        with torch.no_grad():
            prediction = mlcsuh_object.prediction(deps_numg=deps_use)
            loss = loss_operator(sig_true, prediction)/torch.mean(sig_true*sig_true)
            prediction = prediction.cpu()
            loss = loss.cpu()
            sig_true_cpu = sig_true.cpu()
        line = 'Epoch %d \t Loss :%.3e \t' % (epoch, loss.item())
        for num, param in enumerate(params_name_list):
            if param == 'ocr':
                temp = np.exp(params_mlcsuh[num].item() * np.log(param_dic[param]))
            else:
                temp = params_mlcsuh[num].item() * param_dic[param]
            line += ' %s:%.3e \t' % (param, temp)
        line += ' Consumed time: %.3e mins' % ((time.time() - start_time) / 60.)
        print(line)
        mlcsuh_object.return2initial()
        writeLine(fname=os.path.join(saved_directory, 'history.dat'), mode='a', s='\n' + line)
        if epoch % int(1000) == 0:
            directory_temp = os.path.join(saved_directory, 'epoch_%d' % epoch)
            check_mkdir(directory_temp)
            for i, gauss_num in enumerate(numg_used):
                plt.plot(sig_true_cpu[i, :, 1, 1].numpy(), marker='o', label='Simu')
                loss = loss_operator(prediction[i], sig_true_cpu[i])/torch.mean(sig_true_cpu[i]*sig_true_cpu[i])
                plt.plot(prediction[i, :, 1, 1].detach().numpy(), label='Pre')
                # plt.plot(sig_cal[:, 1, 1], label='cal')
                plt.title('Guass %d Epoch %s loss %.3e' % (
                    gauss_num, epoch, loss.item()))
                plt.legend()
                plt.tight_layout()
                fname = 'epoch_%d_gauss_%d.png' % (epoch, gauss_num)
                plt.savefig(os.path.join(directory_temp, fname))
                plt.close()
