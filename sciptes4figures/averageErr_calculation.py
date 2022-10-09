import numpy as np


def get_sample_variance(fname):
    f = open(fname)
    datas = f.readlines()
    f.close()
    i = 0
    while True:
        if 'Index of argsort' in datas[i]:
            i += 1
            temp = datas[i][:-1].split(' ')
            sampleIndex = np.array([int(temp[j]) for j in range(len(temp))])
        elif 'Variance of the prediction' in datas[i]:
            i += 1
            temp = datas[i].split(' ')
            sampleVariance = np.array([float(temp[j]) for j in range(len(temp))])
            break
        else:
            i += 1
    return sampleIndex, sampleVariance


f = open('./average_uncertainty_of_first30percents.txt', 'w')
f.writelines('time\taverageErr\n')
num = 5
time_list = [1] + list(range(5, 101, 5))
average_uncertainty = []
for time_ in time_list:
    fname = '../FEMxML/activeModels/samplingIndex_NNnum_NN%d_TIME%d.txt' % (num, time_)
    sampleIndex, sampleVariance = get_sample_variance(fname)
    aver_err = np.average(sampleVariance)
    line = '%d\t%.6f\n' % (time_, aver_err)
    print(line)
    f.writelines(line)
f.close()
