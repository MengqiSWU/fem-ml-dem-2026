import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file_name = '/media/shguan/Elements SE/ubuntu_home/simu/mcc_examples/consolidation_cylic/biaxial_surf.dat'
dataFrame = pd.read_csv(file_name, delimiter=' ', skiprows=[])
data_array = dataFrame.values
l, force, epsv = data_array[:, 2], data_array[:, 1], data_array[:, 3]
p = np.log(-force/l/1e3)
v0 = 1.348
v = v0*epsv

plt.plot(p, v)
plt.xlabel('$\ln (p)$', fontsize=15)
plt.ylabel('$v$', fontsize=15)
# plt.axes(xscale="log")
plt.tight_layout()
plt.show()
