import time

import numpy as np

from FEMxML.torch_main import train_main_def
from FEMxML.utils_ml import data_clean_numg, get_data_series, check_mkdir
from utilSelf.general import echo


def get_index_numg(
        fname="./footing_ml/active_footing_3618_5/X_epsANDabsxy_Y_D_numNN3_dd8/sample_index.txt"):
    f = open(file=fname, mode='r')
    datas = f.readlines()
    f.close()
    numg_list = []
    for i in datas:
        if "added index:" in i[:20]:
            temp_list = i.split('[')[1].split(']')[0].split(' ')
            if len(temp_list) == 1:
                continue
            temp_numg_list = [int(i) for i in temp_list]
            numg_list += temp_numg_list
    return np.array(numg_list)


def main(
        data_index=0,
        max_time=70,
        outer_directory='./footing_ml/active_footing_3618',
        fourier_features=False,
        sample_ratio=1.0,
        special_str='after_resample',
):

    numg_sig = get_index_numg(
        fname="./footing_ml/active_footing_3618_5/X_epsANDabsxy_Y_sig_numNN3_dd5/sample_index.txt")

    numg_D = get_index_numg(
        fname="./footing_ml/active_footing_3618_5/X_epsANDabsxy_Y_D_numNN3_dd8/sample_index.txt")

    check_mkdir(outer_directory)
    echo('\tReading data ...')
    data_paths = [
        '../../simu/footing/footing_dem_footing_1206_2D_order2_numG3618_2ndShear',
        # '../../simu/footing/important footing_dem_footing_Tonming_2D_order2_numG3114_2ndShear',
    ]
    data_paths = [data_paths[data_index]]
    numg = int(data_paths[data_index].split('numG')[1].split('_')[0])
    # datas_dict = get_data(
    #     root_path_list=data_paths, maxTime=int(max_time), series_flag=False)
    datas_dict = get_data_series(root_path_list=data_paths, maxTime=int(max_time), series_flag=False, numg=numg, )
    datas_dict = data_clean_numg(datas_dict)

    datas_dict_sig, datas_dict_D = {}, {}
    for key in datas_dict.keys():
        datas_dict_sig[key] = datas_dict[key][numg_sig].reshape(-1, len(datas_dict[key][0, 0]))
        datas_dict_D[key] = datas_dict[key][numg_D].reshape(-1, len(datas_dict[key][0, 0]))

    # ------------------------- sig ---------------------------
    train_main_def(
        datas=datas_dict_sig,
        input_features='epsANDabsxy', output_features='sig',
        layerList='dd',
        node_num=5,
        fourier_features=fourier_features,
        outer_directory=outer_directory, epoch_max=int(1e5),
        numSamplesUsed=None, sample_ratio=sample_ratio,
        special_str=special_str,
    )

    # ------------------------- D ---------------------------
    train_main_def(
        datas=datas_dict_D,
        input_features='epsANDabsxy', output_features='D',
        layerList='dd',
        node_num=8,
        fourier_features=fourier_features,
        outer_directory=outer_directory, epoch_max=int(1e5),
        numSamplesUsed=None, sample_ratio=sample_ratio,
        special_str=special_str,
    )


if __name__ == '__main__':
    start_time = time.time()
    main()
    echo('Total time consumed: %.2e mins' % ((time.time() - start_time) / 60.))
