import numpy as np
from utilSelf.general import echo
from sciptes4figures.utils_plot import configurations, plot_training_loss

font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()

f = open(file='../FEMxEPxML/classical_model_train/csuh_dem_train_m_constrained_5/history.dat')
raw_data = f.readlines()
f.close()

temp_lines = raw_data[6:]
echo(temp_lines[-1])
epoch, train_loss, validation_loss = [], [], []
for i in temp_lines:
    temp = i.split('\t')
    loss = float(temp[1].split(':')[1].replace(' ', ''))
    if loss > 4e8:
        pass
    epoch.append(int(temp[0].split(' ')[1].replace(' ', '')))
    train_loss.append(loss)
loss_dic = {
    'mlcons': [np.array(epoch), np.array(train_loss), np.array(validation_loss)]
}
plot_training_loss(loss_dic=loss_dic, validation_plot_flag=False, train_plot_flag=True)
