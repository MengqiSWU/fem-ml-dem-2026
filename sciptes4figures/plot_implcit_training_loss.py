import matplotlib.pyplot as plt
import os
import numpy as np
from sciptes4figures.utils_plot import plot_training_loss


if __name__ == '__main__':
    # ml_path = '../FEMxML/ptModels'
    ml_path = '../FEMxML/biax_ml_1e5'
    file_paths = os.listdir(ml_path)
    train_loss_D, train_loss_sig, epoch = {}, {}, {}
    for i, file in enumerate(file_paths):
        if 'csuh_biaxial_norm_0' not in file:
        # if 'X_epsANDabsxy_Y_sig_dd20_Fourier_noRotate_von_biax_test_0' not in file:
            continue
        temp_loss = []
        temp_loss_validation = []
        temp_epoch = []
        # label = file.split('_')[-3]+'_'+file.split('_')[-2]
        # label = file.split('H')[1]
        label =file.split('_')[4] + '_' + file.split('_')[5]
        file_name = os.path.join(file, 'history.dat')
        try:
            f = open(os.path.join(ml_path, file_name), 'r')
            datas = f.readlines()
            f.close()
        except:
            continue
        i, n = 10, len(datas)
        while i < n:
            if 'Epoch:' in datas[i]:
                try:
                    loss = float(datas[i].split(' ')[2])
                except:
                    print(file)
                temp_loss.append(loss)
                temp_loss_validation.append(float(datas[i].split(' ')[4]))
                temp_epoch.append(int(datas[i].split(' ')[0].split(':')[1].replace('\t', '')))
            i += 1
        train_loss_sig[label] = [np.array(temp_epoch), 0.07*np.array(temp_loss), 0.07*np.array(temp_loss_validation)]
    plot_training_loss(loss_dic=train_loss_sig, )