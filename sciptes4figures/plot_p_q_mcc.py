import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file_name = '/media/shguan/Elements SE/ubuntu_home/simu/mcc_examples/consolidation_cylic/biaxial_surf.dat'
dataFrame = pd.read_csv(file_name, delimiter=' ', skiprows=[])
data_array = dataFrame.values
l, force, epsv = data_array[:, 2], data_array[:, 1], data_array[:, 3]
p = -force/l/1e3
v0 = 1.348
M = 1.344
pc0 = 2.*100
q0 = 0.
p_array = np.linspace(0, pc0, 100)
q_array = np.sqrt((pc0-p_array)*M**2.*p_array)
v = v0*epsv
qq = p-100
pp = (p+100)/2.
p_critical = np.linspace(0, np.max(pp), 100)

plt.plot(pp, qq)
plt.plot(p_array, q_array)
plt.plot(p_critical, p_critical*M, 'r-.')
plt.xlabel('$p$(kPa)', fontsize=15)
plt.ylabel('$q$(kPa)', fontsize=15)
# plt.axes(xscale="log")
plt.axis('equal')
plt.tight_layout()
plt.show()
