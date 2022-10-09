from FEMxML.torch_main import train_main_def
from FEMxML.utils_ml import \
    get_data, check_mkdir, data_clean
from utilSelf.general import echo


def main(
        data_index=0,
        sample_ratio=1.0,
        max_time=70,
        outer_directory='./footing_ml',
        fourier_features=False,
        repeat_num=None,
        series_flag = False,
):
    check_mkdir(outer_directory)

    echo('\tReading data ...')
    data_paths = [
        '../../simu/footing/footing_dem_footing_1206_2D_order2_numG3618_2ndShear',
        '../../simu/footing/important footing_dem_footing_Tonming_2D_order2_numG3114_2ndShear',
        '../../simu/footing/footing_dem_footing552_2D_order1_numG480_2ndShear',
        # '../../simu/footing/whu_footing_dem_footing552_2D_order1_numG480',
        '../../simu/footing/footing_dem_footing303_2D_order1_numG254_2ndShear',
    ]
    data_paths = [data_paths[data_index]]
    special_str = data_paths[0].split('numG')[1].split('_')[0]
    if 'whu' in data_paths[0]:
        special_str += '_whu'
    special_str += '_ratio%.1f' % sample_ratio
    if repeat_num is not  None:
        special_str += '_repeat%d' % repeat_num

    datas_dict = get_data(
        root_path_list=data_paths, maxTime=int(max_time), series_flag=series_flag)
    datas_dict = data_clean(datas_dict)

    # ------------------------- sig ---------------------------
    train_main_def(
        datas=datas_dict,
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
        datas=datas_dict,
        input_features='epsANDabsxy', output_features='D',
        layerList='dd',
        node_num=8,
        fourier_features=fourier_features,
        outer_directory=outer_directory, epoch_max=int(1e5),
        numSamplesUsed=None, sample_ratio=sample_ratio,
        special_str=special_str,
    )


if __name__ == '__main__':
    main(data_index=0, sample_ratio=0.01, series_flag = False)
    # main(data_index=0, sample_ratio=1.)

    """
    1. data_clean                  -> finished
    2. ratio = 1.0 (dense)         -> finished
    3. ratio = 0.1 (dense) then AL -> pending
    """
