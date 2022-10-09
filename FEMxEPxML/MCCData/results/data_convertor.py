# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:15:59 2020

@author: Tongming
"""
from pandas.core.frame import DataFrame
import numpy as np  # import module
import pandas as pd
import glob, os

# x = [0, 1]  # 通过指定列表，删除multiple多余的列

# path=r'F:\文件\DEM_data_20200909\DEM_data_20200909\cpcb'
file = glob.glob(os.path.join(r'*.dat'))  # the py script should be in the save directory of the data files or set the path of the data files

# print(file_cp)
dl = []
# df = []

# order = ["case", "xx_strain", "yy_strain", "zz_strain", "xx_stress", "yy_stress", "zz_stress", "void_ratio",
#          "shear_strain", "volumetric_strain", "q", "p"]
order = ["case", "sigma_xx", "sigma_yy", "sigma_xy", "epsilon_xx", "epsilon_yy", "epsilon_xy", "vonMises", \
         "epsPlastic", "hardening", "epsilonP__xx", "epsilonP__yy", "epsilonP__xy", "yieldValue", "iteration"]
i = 0

for f in file:
    # dfl1=pd.read_excel(f, engine='openpyxl', header=0, index_col=None, names=["case","step","xx_strain","yy_strain","zz_strain","xx_stress","yy_stress","zz_stress","void_ratio","shear_strain","volumetric_strain","q","p"])
    '''
    dfl1 = pd.read_csv(f, header=0)  # this is where the error arises since a key " delimiter=',' " should be added to the function
    header = dfl1.columns
    dfl1['case'] = i   # add a new column called 'case' to the Dataframe: dfl1
    i = i + 1
    dfl1 = dfl1.drop(dfl1.columns[x], axis=1)  # this is used to detele the column [0, 1], which is not necessary in this issue
    dl.append(dfl1)
    '''
    dfl1 = pd.read_csv(f, header=0, delimiter=',')
    dfl1['case'] = i  # add a new column called 'case' to the Dataframe: dfl1
    i = i + 1
    dl.append(dfl1)
    header = dfl1.columns

df = pd.concat(dl)
# df.index = df['case'].apply(str).tolist()

# reset the names of the columns since there are space and # in the original names
order_new = []
for i, h in enumerate(header):
    order_new.append(h.split(' ')[-1])
df.columns = order_new

## 对data重新排序
data = df[order]
## 输出dataframe为csv文件
data.to_csv('DEM_vonMises2D.csv', sep=',', index_label="number", header=True, index=True)  # 输出CSV文件
data1 = pd.read_csv('DEM_vonMises2D.csv', sep=',', index_col=0)

print("Done!")
