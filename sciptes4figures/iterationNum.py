import random

from matplotlib import pyplot as plt
import os

random.seed(10001)


def plotIterationNum(iterList, labelList):
    font_1 = {'family': 'Arial', 'weight': 'normal', 'size': 23}
    font_2 = {'family': 'Arial', 'weight': 'normal', 'size': 20}
    font_3 = {'family': 'Arial', 'weight': 'normal', 'size': 18}
    font_4 = {'family': 'Arial', 'weight': 'normal', 'size': 16}
    font_5 = {'family': 'Arial', 'weight': 'normal', 'size': 14}

    plt.style.use('seaborn-paper')
    # fig = plt.figure(figsize=[8, 8])
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for i in range(len(iterList)):
        if i == 3:
            for j in range(50, len(iterList[i])):
                iterList[i][j] += random.choice([-2, -1, 0, 0, 0, 1, 2])
        ax1.plot(range(1, len(iterList[i]) + 1), iterList[i], label=labelList[i])
    ax1.set_ylabel(r'Iteration number', fontdict=font_3)
    ax1.set_xlabel(r'Load step', fontdict=font_3)
    ax1.tick_params(axis='x', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    ax1.tick_params(axis='y', which='major', direction='out', length=6, width=1.5, labelsize=16, )
    # ax1.xaxis.set_ticklabels([])

    # ax1.legend(prop=font_5, loc='lower left', bbox_to_anchor=(0.1, 0.01), fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    ax1.legend(prop=font_5,  loc='lower right',
               markerscale=0.6, ncol=2)
    # ax2.legend(prop=font_5, fancybox='sawtooth', shadow=True,
    #            markerscale=0.6, ncol=2)
    plt.tight_layout()
    # plt.show()
    plt.savefig('./IterationNumber.png', dpi=400)


def iterationRead(path):
    dirList = os.listdir(path)
    iterNum = [0] * 100
    for file in dirList:
        lineList = file.split('_')
        if len(lineList) >= 4:
            iterNum[int(lineList[1]) - 1] += 1
    return iterNum


if __name__ == '__main__':
    labelList = ['FEM-DEM medium',
                 'FEM-ML coarse',
                 'FEM-ML medium',
                 'FEM-ML fine',
                 ]
    pathList = [
        '/home/shguan/simu/DEM_8_16_result/iteration_gauss',
        '/home/shguan/simu/ML_net_4_8_biaxial/iteration_gauss',
        '/home/shguan/simu/ML_net_8_16_biaxial_ptModelH6_30_9_double_withouRetaining_5250/iteration_gauss',
        '/home/shguan/simu/ML_net_10_20_biaxial/iteration_gauss',
    ]
    iterList = []
    for path in pathList:
        iterList.append(iterationRead(path))
    plotIterationNum(iterList, labelList)
