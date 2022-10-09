import matplotlib.pyplot as plt
import os
import numpy as np
from sciptes4figures.utils_plot import plot_training_loss


if __name__ == '__main__':
    ml_path = '../FEMxML/biax_ml_1e5'
    file_paths = os.listdir(ml_path)
    loss_dic = {}
    for file in file_paths:
        if 'X' not in file or 'csuh_all' not in file or 'm' not in file:
            continue
        epoch, temp_loss_train, temp_loss_valid = [],[], []
        output_features = file.split('_')[-4]
        # label = file.split('_')[-3]+'_'+file.split('_')[-2]
        label = file.split('_')[4]
        file_name = os.path.join(file, 'history.dat')
        try:
            f =open(os.path.join(ml_path, file_name), 'r')
            datas = f.readlines()
            f.close()
        except:
            continue
        i, n = 10, len(datas)
        while i < n:
            if 'Epoch:' in datas[i]:
                epoch.append(int(datas[i].split(' ')[0].split(':')[1][:-1]))
                temp_loss_train.append(float(datas[i].split(' ')[2]))
                temp_loss_valid.append(float(datas[i].split(' ')[4]))
            i += 1
        loss_dic[label] = [np.array(epoch), np.array(temp_loss_train), np.array(temp_loss_valid)]
    plot_training_loss(loss_dic=loss_dic, train_plot_flag=False)