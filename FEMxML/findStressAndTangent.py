from FEMxML.train_model_strain import get_data
import numpy as np

class MatchInputAndOutput:
    def __init__(self, ):
        root_path_list = ['/home/shguan/simu/ABS_DEM_2_4_biaxial',
                          # '/home/shguan/simu/ABS_DEM_2_4_gaussianConfinedPressure0',
                          # '/home/shguan/simu/ABS_DEM_2_4_gaussianConfinedPressure1',
                          # '/home/shguan/simu/ABS_DEM_2_4_gaussianConfinedPressure2'
                          ]
        self.inputs, self.outputs, _ = get_data(root_path_list=root_path_list, maxTime=101)

    def calDistance(self, inputs, singleInputs):
        distance = np.linalg.norm(inputs-singleInputs, axis=1)
        return distance

    def get_stressAndStiffness(self, inputs):
        indexList = [0]*len(inputs)
        distanceList = [np.inf] * len(inputs)
        for i, singleInputs in enumerate(self.inputs):
            distance = self.calDistance(inputs, singleInputs)
            for j in range(len(distance)):
                if distance[j] < distanceList[j]:
                    distanceList[j] = distance[j]
                    indexList[j] = i
        outputs = self.outputs[indexList]
        return outputs[:, :3], outputs[:, 3:]

