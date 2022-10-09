import numpy as np
from yadeimport import *


yade_gz_file_name = './0.yade.gz'
O.load(yade_gz_file_name)
print('='*80)
print(" "*5+"Successfully restore model from %s !\n" % yade_gz_file_name)

#%%
# strain
def strainOrRotaion(mode='strain'):
    if 'strain' in mode:
        param = np.array([0.1, 0, 0, 0, -0.1, 0, 0, 0, 0])
    if '&' in mode:
        param = np.array([0.1, 0, 0, 0, -0.1, 0, 0, 0, 0])+np.array([0, 0.1, 0, -0.1, 0, 0, 0, 0, 0])
    elif 'ro' in mode:
        param = np.array([0, 0.1, 0, -0.1, 0, 0, 0, 0, 0])
    ns = int(max(1e5 * np.max(np.abs(param)), 2))
    dstrain = utils.Matrix3(param)
    O.cell.velGrad = dstrain / (ns * O.dt)
    O.run(ns, True)
    O.cell.velGrad = utils.Matrix3.Zero
    O.wait()


def printMatrix(mm):
    print('%.4f %.4f' % (mm[0, 0], mm[0, 1]))
    print('%.4f %.4f' % (mm[1, 0], mm[1, 1]))

#%%
print('='*80)
# strain
O.load(yade_gz_file_name)
strainOrRotaion('strain')
print(" "*5+"Pure strain:")
printMatrix(O.cell.trsf-Matrix3.Identity)
print()

# rotation
O.load(yade_gz_file_name)
strainOrRotaion('rotation')
print(" "*5+"Pure rotation:")
printMatrix(O.cell.trsf-Matrix3.Identity)
print()

# strain -> rotation
O.load(yade_gz_file_name)
strainOrRotaion('strain')
strainOrRotaion('rotation')
print(" "*5+"strain -> rotation:")
printMatrix(O.cell.trsf-Matrix3.Identity)
print()

# rotation -> strain
O.load(yade_gz_file_name)
strainOrRotaion('rotation')
strainOrRotaion('strain')
print(" "*5+"rotation -> strain:")
printMatrix(O.cell.trsf-Matrix3.Identity)
print()

# rotation & strain
O.load(yade_gz_file_name)
strainOrRotaion('&')
print(" "*5+"rotation & strain:")
printMatrix(O.cell.trsf-Matrix3.Identity)
print()
