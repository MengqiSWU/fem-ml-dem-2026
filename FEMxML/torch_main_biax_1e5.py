import os
from FEMxML.torch_main import train_main_def
from FEMxML.utils_ml import get_data
from utilSelf.general import check_mkdir, echo
import sys

if __name__ == '__main__':
    outer_directory = './biax_ml_1e5'
    check_mkdir(outer_directory)
    data_paths = [
        # biaxial
        # '../../simu/biaxial_0.08/biax_rough_implicit_dem_intorder1_numg800_x10_y20_Reld15st/iteration_gauss',
        # '../../simu/biaxial_Reld/biax_rough_implicit_dem_intorder1_numg200_x5_y10_Reld_St15_modified_H/iteration_gauss',
        '../../simu/biaxial_Reld/biax_rough_implicit_dem_intorder1_numg800_x10_y20_Reld_St12_2H/iteration_gauss',

        # csuh
        # '../../simu/explicit/biaxial/biaxial_rough_explicit_csuh_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_theta8_timestep2.0e-04/iteration_gauss',
        # # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep2.0e-04/added_points',
        # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_1_timestep2.0e-04/added_points',
        # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_2_timestep2.0e-04/added_points',
        # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_3_timestep2.0e-04/added_points',
        # # retaining
        # '../../simu/explicit/retaining/retaining_explicit_csuh_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_theta8_timestep4.0e-04/iteration_gauss',
        # '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep4.0e-04/added_points',
        # # #footing
        # '../../simu/explicit/footing/footing_explicit_csuh_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_p100kPa_ocr_377.4_theta8_timestep4.0e-04_b0.30/iteration_gauss',
        # '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_timestep4.0e-04_b0.30/added_points',

        # biaixal -> misese
        # '../../simu/explicit/biaxial/biaxial_rough_explicit_vonmises_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5/iteration_gauss',
        # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0/added_points',
        # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_1/added_points',
        # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_2/added_points',
        # '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_order1_numg128_biaxial_0.1_162_vel0.20_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_3/added_points',

        # retaining -> mises
        # '../../simu/explicit/retaining/retaining_explicit_vonmises_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5/iteration_gauss',
        # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0/added_points',
        # '../../simu/explicit/retaining/retaining_explicit_mixed_order1_numg271_retaining_321_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_1/added_points',

        # footing -> mises
        # '../../simu/explicit/footing/footing_explicit_vonmises_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5/iteration_gauss',
        # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_0/added_points',
        # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_1/added_points',
        # '../../simu/explicit/footing/footing_explicit_mixed_order1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_all_2/added_points',

        # biaxial & retaining datasets included in the model training then evaluate and add the points into the datasets
        # '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_biax_retain_0/added_points',
        # '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg254_footing303_vel0.40_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_biax_retain_addedFooting_1/added_points',
    ]
    input_n = None
    arg_list = sys.argv
    i, n_args = 0, len(arg_list)
    while i < n_args:
        if arg_list[i] == '-n':
            input_n = int(arg_list[i+1])
            echo("Get -n= %d" % input_n)
            i += 1
        else:
            i += 1
    if input_n == None:
        special_str = 'FEM_DEM_sig'
                      # % (len(data_paths)//3)
    else:
        special_str = 'FEM_DEM_sig'
                      # % (input_n)
        data_paths_temp = []
        for i in range(3):
            data_paths_temp.append(data_paths[i*(len(data_paths)//3)])
            data_paths_temp.append(data_paths[i*(len(data_paths)//3)+1])
            for j in range(1, input_n):
                if i == 0:
                    data_paths_temp.append(
        '../../simu/explicit/biaxial/biaxial_rough_explicit_mixed_intorder1_numg484_biaxial_0.05_548_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_%d_timestep2.0e-04/added_points' % j,
                    )
                elif i == 1:
                    data_paths_temp.append(
        '../../simu/explicit/retaining/retaining_explicit_mixed_intorder1_numg271_retaining_321_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_%d_timestep4.0e-04/added_points' % j,
                    )
                elif i==2:
                    data_paths_temp.append(
        '../../simu/explicit/footing/footing_explicit_mixed_intorder1_numg546_footing615_vel0.10_damp1.0e+06_safe0.5_NNX_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_csuh_all_ocr377_%d_timestep4.0e-04_b0.30/added_points' % j,
                    )
        data_paths = data_paths_temp

    echo(os.getcwd())

    returned_dict = get_data(root_path_list=data_paths, maxTime=int(1e4), explicit_flag=True, add_flag=True)
    strain, stress, strain_abs, H_3F, stress_last, H_0, H_1, tangent, = \
        returned_dict['eps'], returned_dict['sig'], \
        returned_dict['eps_abs'], \
        returned_dict['H_3F'], \
        returned_dict['sig_last'], \
        returned_dict['H_0'] if 'H_0' in returned_dict.keys() else None, \
        returned_dict['H_1'] if 'H_1' in returned_dict.keys() else None, \
        returned_dict['tangent'] if 'tangent' in returned_dict.keys() else None

    # ------------------------- sig ---------------------------
    # layer_list = ['ddd', 'dddd', 'dmdd', 'dmmd']
    layer_list = ['ddd']
    for layers_name in layer_list:
        for node_num in [14]:
            for fourier_features in [True]:
                input_features = 'epsAND3f'  # 'epsANDH' epsANDqH epsANDpqH
                output_features = 'sig'

                train_main_def(
                    datas=returned_dict,
                    input_features=input_features,
                    output_features=output_features,
                    rotate_flag=False,
                    layerList=layers_name,
                    node_num=node_num,
                    fourier_features=fourier_features,
                    outer_directory=outer_directory,
                    epoch_max=int(3.5e4),
                    # numSamplesUsed=int(1e5),
                    special_str=special_str,
                    sample_ratio=0.75

                )

    # for node_num in [5, 10, 20, 30, 60, 100]:
    #     trainMain_mask(
    #         strain=strain, strain_abs=strain_abs, stress=stress, tangent=tangent, stress_last=stress_last,
    #         input_features='eps',
    #         output_features='sig',
    #         rotate_flag=False,
    #         layerList='dd',
    #         node_num=node_num,
    #         fourier_features=False, outer_directory=outer_directory,
    #         epoch_max=int(1e5),
    #         numSamplesUsed=int(2e5))

    # ------------------------- D ---------------------------

            # train_main_def(
            #     datas=returned_dict,
            #     input_features=input_features,
            #     output_features=output_features,
            #     rotate_flag=False,
            #     layerList=layers_name,
            #     node_num=node_num,
            #     fourier_features=fourier_features,
            #     outer_directory=outer_directory,
            #     epoch_max=int(1e5),
            #     numSamplesUsed=int(1e5),
            #     special_str=special_str,
            # )
