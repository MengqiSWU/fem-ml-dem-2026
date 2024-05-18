import os
from FEMxML.torch_main_3d import train_main_def
from FEMxML.utils_ml import get_data
from FEMxML.utils_ml_3d import get_data_3d
from utilSelf.general import check_mkdir, echo
import sys

if __name__ == '__main__':
    outer_directory = './triax_ml_1e5'
    check_mkdir(outer_directory)
    data_paths = [
        # biaxial
        # '../../simu/biaxial_0.08/biax_rough_implicit_dem_intorder1_numg800_x10_y20_Reld15st/iteration_gauss',
        # '../../simu/biaxial_Reld/biax_rough_implicit_dem_intorder1_numg200_x5_y10_Reld_St15_modified_H/iteration_gauss',
        # '../../simu/biaxial_3D/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y6e8_fri0.3_p0.2_n1000_Reld/iteration_gauss',  #for x6_y6_z6
        # '../../simu/biaxial_3D/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y6e8_f0.3_p0.2_n600_Reld_Modified/iteration_gauss', #for x8_y8_z16_Reld_bad
        # '../../simu/biaxial_3D/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y4.5e8_fri0.4_p0.2_n600_Reld/iteration_gauss',
        # '../../simu/biaxial_3D/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y4e8_fri0.4_p0.2_n600_Reld/iteration_gauss',
        # '../../simu/biaxial_3D/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y4e8_fri0.35_p0.2_n600_Reld/iteration_gauss',
        # '../../simu/biaxial_3D/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y5e8_fri0.35_p0.2_n600_Reld/iteration_gauss',  #better
        # '../../simu/biaxial_3D/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y6e8_fri0.3_p0.2_n1000_Reld_modified/iteration_gauss',
        # '../../simu/biaxial_3D/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y6e8_fri0.3_p0.2_n600_Reld (copy)/iteration_gauss',
        # '../../simu/biaxial_3D_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y6e8_fri0.3_p0.2_n1200_Reld/iteration_gauss',
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg128_x2_y2_Y3e8_fri0.5_p0.3_n1000_rM005_Reld/iteration_gauss',  #trial6
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y3e8_fri0.5_p0.3_n1000_rM005_Reld/iteration_gauss',  #trial7
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg128_x2_y2_Y3e8_fri0.5_p0.2_n1000_rM01_Reld/iteration_gauss',   #trial8
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y3e8_fri0.5_p0.2_n1000_rM01_Reld/iteration_gauss',    #trial9
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y3e8_fri0.6_p0.2_n1000_rM005_Reld/iteration_gauss',    #trial10
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg128_x2_y2_Y3e8_fri0.52_p0.3_n1000_rM005_Reld_accum/iteration_gauss', #trial11
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y3e8_fri0.52_p0.3_n1000_rM005_Reld/iteration_gauss',  #trial12
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y3e8_fri0.52_p0.3_n1000_rM005_Reld_modified/iteration_gauss', #trial13
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y3e8_fri0.5_p0.3_n1000_rM005_Reld/iteration_gauss',    #trial14
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y3e8_fri0.5_p0.3_n1000_rM005_Reld_modified/iteration_gauss', #trial15
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y3e8_fri0.5_p0.3_n1000_rM005_Reld/iteration_gauss', #trial16
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y2e8_fri0.2_p0.4_n1000_rM005_Reld/iteration_gauss',  #trial_split_sig
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y3e8_fri0.5_p0.3_n1000_rM005_Reld_modified/iteration_gauss',  #trial16
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y6e8_fri0.5_p0.8_n1000_rM005_with_rotation/iteration_gauss',  #trial18
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y4e8_fri0.5_p0.2_n1000_rM01/iteration_gauss',   #trial17  #trial19
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y4e8_fri0.5_p0.2_n1000_rM01_noDense/iteration_gauss'  #trial20
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y5e8_fri0.5_p0.2_n1000_rM01/iteration_gauss'     #trial21  174,0.75
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y5e8_fri0.5_p0.3_n1000_rM01_Denser/iteration_gauss' #trial22 258 0.75
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg432_x3_y3_Y4e8_fri0.5_p0.3_n1000_rM01_Denser_accum/iteration_gauss'  #trial23 224 0.75
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg2000_x5_y5_Y4e8_fri0.5_p0.3_n1000_rM01_Denser_accum/iteration_gauss'   #trial24 248 0.75
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y6e8_fri0.3_p0.2_n600_rM01_denser_accum_norotation/iteration_gauss'   #trial25 194 0.75  trial25 split_D
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y6e8_fri0.2_p0.5_n600_rM01_denser_accum_norotation_St12/iteration_gauss' #trial26 271 0.75
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y6e8_fri0.5_p0.3_n600_rM01_denser_accum_rotation/iteration_gauss'  #trial27 258 0.75
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg2000_x5_y5_Y7e8_fri0.5_p0.3_n600_rM01_denser_accum_rotation/iteration_gauss'  #trial28 289 0.75
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg2000_x5_y5_Y7e8_fri0.5_p0.3_n600_rM01_denser_accum_rotation_modified/iteration_gauss'  #trial28 309 0.7
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y6e8_fri0.3_p0.2_n1000_rM01_denserhalf_accum_norotation/iteration_gauss' #trial29 289 0.75
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg2000_x5_y5_Y3e8_fri0.5_p0.3_rM01_denser_accum_rotation/iteration_gauss'   #trial30  276 0.75
        # '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y6e8_fri0.5_p0.3_n1000_rM007_denser_accum_rotation/iteration_gauss'  trial31 320 0.7 split D
        '../../simu/Triaxial_Reld/biax_3d_rough_implicit_dem3d_intorder1_numg1024_x4_y4_Y3e8_fri0.5_p0.3_rM01_denser_accum_rotation_mono/iteration_gauss'   #trial32

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

    returned_dict = get_data_3d(root_path_list=data_paths, maxTime=int(1e4), explicit_flag=True, add_flag=True)
    strain, stress, stress_v, stress_r, H_3d, H_2d, tangent, D_voigt, D_rest= \
        returned_dict['eps'], \
        returned_dict['sig'], \
        returned_dict['sigv'], \
        returned_dict['sigr'], \
        returned_dict['H_3D'], \
        returned_dict['H_2D'], \
        returned_dict['tangent'] if 'tangent' in returned_dict.keys() else None, \
        returned_dict['D_voigt'] if 'tangent' in returned_dict.keys() else None, \
        returned_dict['D_rest'] if 'tangent' in returned_dict.keys() else None,

    # ------------------------- sig ---------------------------
    # layer_list = ['ddd', 'dddd', 'dmdd', 'dmmd']
    layer_list = ['dddd']
    for layers_name in layer_list:
        for node_num in [20]:
            for fourier_features in [True]:
                input_features = 'epsAND3d'  # 'epsANDH' epsANDqH epsANDpqH
                output_features = 'sig'

                train_main_def(
                    datas=returned_dict,
                    numg = int(data_paths[0].split('numg')[1].split('_')[0]),
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
                    manual= False,
                    iteration= 276,        #174 trial21
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
