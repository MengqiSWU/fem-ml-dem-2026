from FEMxML.torch_active_learning_onthefly import main_active_mask
from FEMxML.utils_ml import get_data, save_scalar, reconstruct_x_y, echo, check_mkdir, findDevice, sampling_index, \
    writeLine, calVariance, remove_from_list


if __name__ =='__main__':
    outer_directory = './biax_ml_1e5'
    check_mkdir(outer_directory)

    data_paths = [
        # biaixal -> misese
        '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5/iteration_gauss',
        '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all/added_points',
        '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_1/added_points',
        '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_2/added_points',
        '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_3/added_points',
        '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_4/added_points',

        # retaining -> mises
        '../../simu/explicit/retaining/retaining_explicit_vonmises_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5/iteration_gauss',
        '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all/added_points',
        '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_1/added_points',
        '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_2/added_points',
        '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_3/added_points',
        '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_4/added_points',

        # footing -> mises
        '../../simu/explicit/footing/footing_explicit_vonmises_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5/iteration_gauss',
        '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all/added_points',
        '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_1/added_points',
        '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_2/added_points',
        '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_3/added_points',
        '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDH_Y_sigANDH_dmdd40_Fourier_noRotate_von_all_4/added_points',
    ]
    returned_dict = get_data(root_path_list=data_paths, maxTime=int(1e4), explicit_flag=True, add_flag=True)
    strain, stress, strain_abs, stress_last, H_0, H_1, tangent = \
        returned_dict['eps'], returned_dict['sig'], \
        returned_dict['eps_abs'], \
        returned_dict['sig_last'], \
        returned_dict['H_0'], returned_dict['H_1'], returned_dict['tangent']

    datas_dict = returned_dict

    # ------------------------- sig ---------------------------
    for node_num in [20]:
        input_features = 'epsANDH'  # 'epsANDH'
        output_features = 'sigANDH'
        main_active_mask(
            datas=datas_dict,
            input_features=input_features, output_features=output_features,
            nodenum=node_num, numNN=3, iter_max=4, ratio_per_iter=0.05, first_train_ratio=0.2,
            epoch_per_iter=int(5e4), layerList='dmdd', fourier_features=True
        )