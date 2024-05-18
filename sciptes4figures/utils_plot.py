import os
import pickle
import numpy as np
import json
import matplotlib.pyplot as plt


def generate_configuration():
    font_1 = {'weight': 'normal', 'size': 23}
    font_2 = {'weight': 'normal', 'size': 20}
    font_3 = {'weight': 'normal', 'size': 18}
    font_4 = {'weight': 'normal', 'size': 16}
    font_5 = {'weight': 'normal', 'size': 14}

    tickParamsDic = {'which': 'major',
                     'direction': 'out',
                     'length': 6,
                     'width': 1.5,
                     'labelsize': 16}
    legendDic = {'prop': font_4,
                 'fancybox': True,
                 'framealpha': 0.1,
                 'facecolor': 'c'}
    fname = os.path.join(os.getcwd(), './plot_configuration.json')
    print('File saved as: \t\t%s' % fname)
    with open('plot_configuration.json', 'w', encoding='utf-8') as f:
        json.dump([font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic], f, ensure_ascii=False, indent=4)


def readTopForce_biaxial(path='/home/mengqi/simu/biaxial_0.08', split_keyword='dem'):
    flist = os.listdir(path)
    fname = None
    for i in flist:
        if '.dat' in i:
            fname = i
            break
    if fname is None:
        print('There is no dat file in %s' % path)
        raise
    fileName = os.path.join(path, fname)
    label = os.path.split(path)[-1].split(split_keyword)[-1][1:]
    file = open(fileName)
    datas = file.readlines()
    file.close()
    # head = datas[0]
    data = np.array([[float(j) for j in i.replace('\n', '').split(' ')] for i in datas[1:-1]])
    return data, label


def read_footing(path):
    fileName = os.path.join(path, 'bearing.dat')
    label = os.path.split(path)[-1].split('footing_')[-1]
    file = open(fileName)
    datas = file.readlines()
    file.close()
    file.close()
    # head = datas[0]
    data = np.array([[float(j) for j in i.replace('\n', '').split(' ')] for i in datas[1:-1]])
    return data


def get_color_list():
    color_list = [
        '#1f77b4',
        '#ff7f0e',
        '#2ca02c',
        '#d62728',
        '#9467bd',
        '#8c564b',
        '#e377c2',
        '#7f7f7f',
        '#bcbd22',
        '#17becf']
    return color_list


def configurations():
    # with open('plot_configuration.json', 'r', encoding='utf-8') as f:
    # font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = json.load(f)
    font_1 = {'weight': 'normal', 'size': 23}
    font_2 = {'weight': 'normal', 'size': 20}
    font_3 = {'weight': 'normal', 'size': 18}
    font_4 = {'weight': 'normal', 'size': 16}
    font_5 = {'weight': 'normal', 'size': 14}

    tickParamsDic = {'which': 'major',
                     'direction': 'out',
                     'length': 6,
                     'width': 1.5,
                     'labelsize': 16}
    legendDic = {'prop': font_4,
                 'fancybox': True,
                 'framealpha': 0.1,
                 'facecolor': 'c'}
    return font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic


def plot_sig_series(eps, sig, numg, prediction_save_path, sig_pre=None, eps_simu=None, sig_simu=None, legend_flag=False, scatter_flag=True):
    color_list = get_color_list()
    direction_list = ['x', 'xy', 'y']
    for i in range(3):
        if scatter_flag:
            n = len(eps)
            plot_index = np.arange(0, n, n//20)
            plt.scatter(-eps[plot_index, 2], -sig[plot_index, i] / 1e3,
                        label=r'exFEM-CSUH $\sigma_{%s}$' % direction_list[i], marker='o', s=20, color=color_list[i],
                        edgecolors='k')
        else:
            plt.plot(-eps[:, 2], -sig[:, i] / 1e3,
                     label=r'exFEM-CSUH $\sigma_{%s}$' % direction_list[i], linewidth=2, color=color_list[i])
    if sig_simu is not None:
        for i in range(3):
            plt.plot(-eps_simu[:, 2], -sig_simu[:, i] / 1e3,
                     label='exFEM-NN $\sigma_{%s}$' % direction_list[i], linewidth=2, color=color_list[i])
    if sig_pre is not None:
        for i in range(3):
            n = len(eps)
            plot_index = np.arange(0, n, n // 20)
            plt.scatter(-eps[plot_index, 2], -sig_pre[plot_index, i] / 1e3,
                        label='Directly NN $\sigma_{%s}$' % direction_list[i], marker='v', s=40, color=color_list[i],
                    edgecolors='k', alpha=0.5)
    font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
    plt.xlabel(r'$\epsilon_y$', font_2)
    plt.ylabel(r'Pressure (kPa)', font_2)
    plt.tick_params(**tickParamsDic)
    if legend_flag:
        plt.legend(**legendDic)
    plt.tight_layout()
    plt.savefig('%s/numg_%d.png' % (prediction_save_path, numg))
    plt.close()


def get_num_error_prediction(path_temp):
    step, num = [], []
    file_list = os.listdir(os.path.join(path_temp, 'added_points'))
    for i in file_list:
        temp = i.split('_')
        step.append(int(temp[1]))
        num.append(int(temp[2].replace('.dat', '')))
    index = np.argsort(step)
    step_sorted = np.array(step)[index]
    num_sorted = np.array(num)[index]
    return step_sorted, num_sorted


def plot_training_loss(loss_dic: dict, train_plot_flag=False, validation_plot_flag=True):
    '''
    loss_dic = {
    'label xxx': [epoch, train, validation],
    }
    '''
    font_1, font_2, font_3, font_4, font_5, tickParamsDic, legendDic = configurations()
    plt.figure()
    keys = loss_dic.keys()
    for key in keys:
        epoch, train, validation = loss_dic[key]
        if train_plot_flag:
            plt.semilogy(epoch / 1e3, train, label=key + '_train')
        if validation_plot_flag:
            plt.semilogy(epoch / 1e3, validation, label=key + '_validation')
    plt.legend(**legendDic)
    # plt.title(r'Training loss of $%s$' % (s if s!='sigma' else r'\sigma'))
    plt.xlabel('Epoch/1e3', fontdict=font_3)
    plt.ylabel('Loss', fontdict=font_3)
    plt.tick_params(axis='x', **tickParamsDic)
    plt.tick_params(axis='y', **tickParamsDic)
    plt.tight_layout()
    plt.show()
    plt.close()


if __name__ == '__main__':
    generate_configuration()
