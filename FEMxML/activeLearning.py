import numpy as np
import os
from netTorchDD import getFilesPathList, getGaussianPointsIndex, getMeanStd, \
    pickle_dump, pickle_load, modelTrain, Net
import torch
from mlUtils import blockDataReader


def get_data4ActiveLearning(root_path_list, maxTime=100, scalarPath=None, mixflag=False, time=None):
    file_list = getFilesPathList(root_path_list, maxTime)
    n = 512
    if mixflag:
        indexList = []
        for rootfile in root_path_list:
            indexList.append(getGaussianPointsIndex(filePath=rootfile))

    stress, strain, fabric, strain_increment, tangent, strain_abs = [], [], [], [], [], []
    for file in file_list:
        if time is not None:
            time_current = int(os.path.split(file)[-1].split('_')[1])
            if time != time_current:
                continue
        f = open(file, 'r')
        datas = f.readlines()
        f.close()
        n = int(datas[0].split(' ')[-1])  # n is the number of the RVEs
        listTemp = range(1, n + 1)
        if mixflag:
            if n == 32:
                listTemp = np.concatenate((indexList[0]['lower'], indexList[0]['upper']), axis=0) + 1
            elif n == 512:
                listTemp = indexList[1]['shear'] + 1
        i, tol = 0, len(datas)
        while i < tol:
            '''
                NAME                # 
            strain_increment        0
            strain_toatal           1
            stress_increment        2
            stress_toatal           3 
            tangent                 4
            fabric                  5
            vR                      6
            strain_abs              7
            '''
            if 'strain_increment' in datas[i]:
                strain_increment.append(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'strain_toatal' in datas[i]:
                strain.append(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'stress_toatal' in datas[i]:
                stress.append(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'fabric' in datas[i]:
                fabric.append(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'tangent' in datas[i]:
                tangent.append(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            elif 'strain_abs' in datas[i]:
                strain_abs.append(blockDataReader(datas[i + 1:i + n + 1]))
                i += (n + 1)
            else:
                i += 1
    # delete stress subject to the symmetricity \sigma_xy = \sigma_yx
    stress = np.array(stress)
    stress = np.concatenate((stress[:, :, 0:1], stress[:, :, 2:4]), axis=2)
    strain, strain_increment = np.array(strain), np.array(strain_increment)
    # strain_total = strain + strain_increment
    fabric = np.array(fabric)
    tangent = np.array(tangent)
    strain_abs = np.array(strain_abs)

    # split the rigid rotation from the displacement gradients
    strain = np.concatenate((strain[:, :, 0:1], .5 * (strain[:, :, 1:2] + strain[:, :, 2:3]), strain[:, :, 3:4]),
                            axis=2)
    strain_increment = np.concatenate((strain_increment[:, :, 0:1],
                                       .5 * (strain_increment[:, :, 1:2] + strain_increment[:, :, 2:3]),
                                       strain_increment[:, :, 3:4]), axis=2)
    # strain_total = strain
    strain_abs = np.concatenate((strain_abs[:, :, 0:1],
                                 .5 * (strain_abs[:, :, 1:2] + strain_abs[:, :, 2:3]),
                                 strain_abs[:, :, 3:4]), axis=2)

    input_value = np.concatenate((strain, strain_abs[:, :, 2:3]), axis=2)
    output_value = np.concatenate((stress, tangent), axis=2)

    # find the mean and the std of the data
    s_mean, s_std = getMeanStd(stress.reshape(-1, stress.shape[-1]))
    total_e_mean, total_e_std = getMeanStd(data=strain.reshape(-1, strain.shape[-1]))
    tangent_mean, tangent_std = getMeanStd(data=tangent.reshape(-1, tangent.shape[-1]))
    strain_abs_mean, strain_abs_std = getMeanStd(data=strain_abs.reshape(-1, strain_abs.shape[-1]))
    input_mean, input_std = getMeanStd(data=input_value.reshape(-1, input_value.shape[-1]))
    output_mean, output_std = getMeanStd(data=output_value.reshape(-1, output_value.shape[-1]))

    cwd = os.getcwd()
    if scalarPath:
        pickle_dump(root_path=os.path.join(cwd, scalarPath) if scalarPath else cwd,
                    s_mean=s_mean, s_std=s_std,
                    total_e_mean=total_e_mean, total_e_std=total_e_std,
                    tangent_mean=tangent_mean, tangent_std=tangent_std,
                    strain_abs_mean=strain_abs_mean, strain_abs_std=strain_abs_std,
                    input_mean=input_mean, input_std=input_std,
                    output_mean=output_mean, output_std=output_std)
    input_value = np.einsum('ijk->jik', input_value)
    output_value = np.einsum('ijk->jik', output_value)
    return input_value, output_value, n


class preTrianedModel:
    def __init__(self, modelPath, numModel=5):
        self.models = [Net(layerList=[4, 30, 30, 9]) for _ in range(numModel)]
        self.modelPath = modelPath
        if not os.path.exists(self.modelPath):
            os.mkdir(self.modelPath)

    def preTrain(self, x, y):
        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])
        for i, model in enumerate(self.models):
            tempPath = os.path.join(self.modelPath, 'model_%d' % (i))
            if not os.path.exists(tempPath):
                os.mkdir(tempPath)
            temp = modelTrain(model=model, patienceNum=2.5e3, savePath=tempPath,
                              optimMode='adam', scalerPath=self.modelPath)
            index = np.random.permutation(range(len(x)))
            epoch = temp.train(inputs=x[index], outputs=y[index],
                               optimMode='adam',
                               epochMax=50001)  # pre-training 1.11e-3


class restoreAndSampling:
    def __init__(self, modelPath):
        self.modelPath = modelPath
        modelList = os.listdir(modelPath)
        modelList2 = []
        for i in modelList:
            if 'model' in i:
                modelList2.append(i)
        modelList = modelList2.copy()
        del modelList2
        modelList = [os.path.join(modelPath, modelList[i]) for i in range(len(modelList))]
        # datas
        self.savePath = modelList[0]
        self.input_mean, self.input_std, self.output_mean, self.output_std, = pickle_load(
            'input_mean', 'input_std', 'output_mean', 'output_std', root_path=self.modelPath)
        # models initialization
        self.losses = torch.nn.MSELoss()
        self.models = []
        for files in modelList:
            self.models.append(torch.load(os.path.join(files, 'entire_model.pt')))
        # resampling
        # self.sampleIndex = self.sampling()
        # # train model
        # for model in self.models:
        #     model

    def sampling(self, x, time):
        if time is None:
            nn_num_list = [3, 5, 6]
        else:
            nn_num_list = [5]
        for num in nn_num_list:
            predictionsVariance = []
            for nodeData in x:
                predictionTemp = []
                for model in self.models:
                    x_normed = torch.from_numpy(
                        modelTrain.normalize(nodeData, self.input_mean, self.input_std)).to(torch.device('cuda'))
                    predictionTemp.append(model(x_normed).cpu().detach().numpy())
                predictionTemp = np.array(predictionTemp)
                variance = calVariance(predictionTemp, num=num)
                predictionsVariance.append(variance)

            predictionsVariance = np.array(predictionsVariance)
            sampleIndex = np.argsort(predictionsVariance)[::-1]
            if time is not None:
                fname = 'samplingIndex_NNnum_NN%d_TIME%d.txt' % (num, time)
            else:
                fname = 'samplingIndex_NNnum_NN%d.txt' % num
            with open(os.path.join(self.modelPath, fname), 'w') as f:
                f.write('Index of argsort (From large to small)\n')
                message = ' '.join('%d' % i for i in sampleIndex)
                f.write(message)
                f.write('\n'+'='*80+'\n')
                f.write('Variance of the prediction\n')
                message = ' '.join('%.8e' % i for i in predictionsVariance)
                f.write(message)
        return

    def trainModel(self, datas):
        return


def calVariance(data, num=6):
    std = np.std(data[:num], axis=0)
    stdAverage = np.average(std)
    return stdAverage


def readSampleIndex(path='./activeModels/samplingIndex.txt'):
    f = open(path)
    datas = f.readlines()
    f.close()
    i = 0
    while True:
        if 'Index of argsort' in datas[i]:
            i += 1
            temp = datas[i][:-1].split(' ')
            sampleIndex = np.array([int(temp[j]) for j in range(len(temp))])
            break
        else:
            i += 1
    return np.array(sampleIndex)


if __name__ == "__main__":
    pretrainPath = 'activeModels'

    # -------------------------------------------------------------------------
    # pre train
    # print()
    # print('=' * 80)
    # print('Reading data (MESH 2 4) ...')
    # x_24, y_24, n_24 = get_data4ActiveLearning(
    #     root_path_list=['../../simu/DEM_implicitSmooth_dem_2_4_stiffnessDouble'],
    #     mixflag=False,
    #     scalarPath=pretrainPath,
    # )
    # n = len(x_24)
    # print()
    # print('-' * 80)
    # print('Read data finished!')
    #
    # pretrain = preTrianedModel(modelPath=pretrainPath)
    # pretrain.preTrain(x=x_24, y=y_24)

    # -------------------------------------------------------------------------
    # resampling
    # model restoring
    restoreAndSampling = restoreAndSampling(modelPath=pretrainPath)
    # # NN number sensitivity analysis
    # print()
    # print('=' * 80)
    # print('Reading data (MESH 8 16) ...')
    # x_816, y_816, n_816 = get_data4ActiveLearning(
    #     root_path_list=['../../simu/DEM_implicitSmooth_dem_8_16_stiffnessDouble'],
    #     mixflag=False, scalarPath=None)
    # print()
    # print('-' * 80)
    # print('Read data finished!')
    # print('Length of input:  %d' % len(x_816[0]))
    # print('Length of output: %d' % len(y_816[0]))
    # restoreAndSampling.sampling(x=x_816, time=None)

    # analysis: resampling along time
    for time_ in [1]+list(range(5, 101, 5)):
        print('\n  time: %d' % (time_))
        x_temp, y_temp, n_temp = get_data4ActiveLearning(
            root_path_list=['../../simu/DEM_implicitSmooth_dem_8_16_stiffnessDouble'],
            mixflag=False, scalarPath=None, time=time_)
        sampling = restoreAndSampling.sampling(x=x_temp, time=time_)

    # -------------------------------------------------------------------------
    # train the resampled model (retrain the model with regard to the first 35% of the samples with higher variance)
    # resampledModelPath = './activeLearningResampledModel'
    # if not os.path.exists(resampledModelPath):
    #     os.mkdir(resampledModelPath)
    # usingRatio = .35
    # sampleIndex = readSampleIndex(path=os.path.join(pretrainPath, 'samplingIndex.txt'))
    # print()
    # print('=' * 80)
    # print('Reading data (MESH 2 4 & 8 16) ...')
    # x_24, y_24, n_24 = get_data4ActiveLearning(
    #     root_path_list=['../../simu/DEM_implicitSmooth_dem_2_4_stiffnessDouble'],
    #     mixflag=False,
    #     scalarPath=resampledModelPath,
    # )
    # x_816, y_816, n_816 = get_data4ActiveLearning(
    #     root_path_list=['../../simu/DEM_implicitSmooth_dem_8_16_stiffnessDouble'],
    #     mixflag=False)
    # x_total = np.concatenate(
    #     (x_24.reshape(-1, x_24.shape[-1]),
    #      x_816[sampleIndex[:int(usingRatio*len(sampleIndex))]].reshape(-1, x_816.shape[-1])), axis=0)
    # y_total = np.concatenate(
    #     (y_24.reshape(-1, y_24.shape[-1]),
    #      y_816[sampleIndex[:int(usingRatio*len(sampleIndex))]].reshape(-1, y_816.shape[-1])), axis=0)
    #
    # model = Net(layerList=[4, 30, 30, 9])
    #
    # temp = modelTrain(model=model, patienceNum=2.5e3, savePath=resampledModelPath,
    #     optimMode='adam')
    # index = np.random.permutation(range(len(x_total)))
    # epoch = temp.train(inputs=x_total[index], outputs=y_total[index],
    #     optimMode='adam',
    #     epochMax=150001)  # pre-training 1.11e-3
