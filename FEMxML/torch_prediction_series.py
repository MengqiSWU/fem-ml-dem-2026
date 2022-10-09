import os.path
import numpy as np
from utilSelf.general import check_mkdir
from torch_restore import modelRestore
from utils_ml import get_data_series, prediction
import matplotlib.pyplot as plt
from sciptes4figures.utils_plot import plot_sig_series



def main(
        root_path_list =['../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_safe0.5'],
        model_path='biax_ml_1e5/X_eps_Y_sig_dmdd40_Fourier_noRotate_vonmises'):
    numg = 128
    datas = get_data_series(root_path_list, maxTime=int(1e6), mixflag=False, numg=numg)
    if len(datas) == 5:
        strain, strain_abs, stress, tangent, stress_last = datas
    elif len(datas) == 3:
        strain, strain_abs, stress = datas
    else:
        strain, strain_abs, stress, tangent = datas
    prediction_save_path = os.path.join(model_path, 'series_prediction')
    check_mkdir(prediction_save_path)
    model = modelRestore(
        savedPath=model_path)
    index = list(range(0, numg, 10))
    for numg_temp in index:
        if 'absy'in os.path.split(model_path)[-1]:
            stress_pre=prediction(
                model=model, eps=np.concatenate((strain[numg_temp], strain_abs[numg_temp, :, 2:3]), axis=1))
        else:
            stress_pre = prediction(model=model, eps=strain[numg_temp])
        plot_sig_series(
            eps=strain[numg_temp], sig=stress[numg_temp], sig_pre=stress_pre,
            numg=numg_temp, prediction_save_path=prediction_save_path)


if __name__ == '__main__':
    main()
