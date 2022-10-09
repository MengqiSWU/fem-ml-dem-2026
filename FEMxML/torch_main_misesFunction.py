from FEMxML.torch_main import train_main_dy_mask
from FEMxML.mlUtils import get_data
from utilSelf.general import echo, check_mkdir
import numpy as np
from FEMxEPxML.utils_constitutive import getQ_2d, get_dq_dsig_2d


if __name__ == '__main__':
    x = np.random.random(size=[int(1e3), 3])
    y = getQ_2d(sig=x)
    dy = get_dq_dsig_2d(sig=x)
    train_main_dy_mask(
        x=x, y=y, dy=dy,
        outer_directory='sobolev_training',
        numSamplesUsed=int(1e4), epoch_max=int(1e4), layerList='dmdmd',
        fourier_features=True, dy_weight=0.5, node_num=30)