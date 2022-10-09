import time
from mpipool import MPIPool
import sys

def func(i):
    time.sleep(i*0.1)

pool = MPIPool()
pool.start()
print('*\t  %d' % pool.comm.rank)
# if pool.is_worker():
#     sys.exit()
num = 12
funcList, argsList = func, [None]*num
for i in range(num):
    argsList[i] = i
pool.map(function=funcList, iterable=argsList)

print('\n'+'='*80)
print('finished')
print('\t * %d' % pool.comm.rank)
# pool.close()
# print(pool.comm.size)
