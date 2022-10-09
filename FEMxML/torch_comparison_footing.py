from FEMxML.torch_main_footing_tongming import train_main_def
from FEMxML.utils_ml import get_data, check_mkdir, echo

outer_directory = './footing_ml'
check_mkdir(outer_directory)
echo('\tReading data ...')
data_paths = ['../../simu/footing/footing_dem_footing552_2D_order1_numG480']
strain, strain_abs, stress, tangent = get_data(
    root_path_list=data_paths, maxTime=int(165))
numSamplesUsed = int(2e5)
epoch_max = int(1e5)
echo('numUsed/numTotal:  %d/%d' %(numSamplesUsed, len(strain)),
     'epoch max:         %d' % (epoch_max))

# ----------------------------Node_num----------------------------
# for node_num in [2, 5, 10, 20, 30, 40, 60, 100]:
for node_num in [20]:
    train_main_def(
        strain=strain, strain_abs=strain_abs, stress=stress, tangent=tangent,
        input_features='epsANDabsxy', output_features='sig',
        layerList='dd', fourier_features=False,
        node_num=node_num, epoch_max=epoch_max, numSamplesUsed=numSamplesUsed, outer_directory=outer_directory)

for node_num in [20]:
    train_main_def(
        strain=strain, strain_abs=strain_abs, stress=stress, tangent=tangent,
        input_features='epsANDabsxy', output_features='D',
        layerList='dd', fourier_features=False,
        node_num=node_num, epoch_max=epoch_max, numSamplesUsed=numSamplesUsed, outer_directory=outer_directory)

# ----------------------------layer_list----------------------------
# for layer_list in ['d', 'dmd', 'ddd', 'dddd', 'dmdd']:
#     trainMain(
#         strain=strain, strain_abs=strain_abs, stress=stress, tangent=tangent,
#         input_features='epsANDabsxy', output_features='sig',
#         layerList=layer_list, fourier_features=False,
#         node_num=10, epoch_max=epoch_max, numSamplesUsed=numSamplesUsed, outer_directory=outer_directory)
#
#
# for layer_list in ['d', 'dmd', 'ddd', 'dddd', 'dmdd']:
#     trainMain(
#         strain=strain, strain_abs=strain_abs, stress=stress, tangent=tangent,
#         input_features='epsANDabsxy', output_features='D',
#         layerList=layer_list, fourier_features=False,
#         node_num=10, epoch_max=epoch_max, numSamplesUsed=numSamplesUsed, outer_directory=outer_directory)