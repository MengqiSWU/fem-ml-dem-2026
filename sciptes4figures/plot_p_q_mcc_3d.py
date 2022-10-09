import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


file_name = '/media/shguan/Elements SE/ubuntu_home/simu/mcc_examples/mcc_conventionalP_8_8_16_3D/biaxial_surf.dat'
dataFrame = pd.read_csv(file_name, delimiter=' ', skiprows=[92])
data_array = dataFrame.values
axialEps, force, area, epsv = data_array[:, 0], data_array[:, 1], data_array[:, 2], data_array[:, 3]
fz = -force/area/1e3
v0 = 1.348
M = 1.344
pc0 = 2.*100
q0 = 0.
p_array = np.linspace(0, pc0, 100)
q_array = np.sqrt((pc0-p_array)*M**2.*p_array)
v = v0*(epsv+1.)
qq = fz-100
pp = (fz+200)/3.
p_critical = np.linspace(0, np.max(pp), 100)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(-axialEps, qq, 'g-', label='$q - \epsilon_{axial}$')
plt.xlabel('$\epsilon_{axial}$', fontsize=15)
plt.ylabel('$q$ (kPa)', fontsize=15)
plt.legend(loc='upper left', fontsize=15)
ax.twinx()
plt.plot(-axialEps, v-1., 'y-', label='$e-\epsilon_{axial}$')
plt.legend(loc='upper right', fontsize=15)
plt.ylabel('$e$ (void ratio)', fontsize=15)
ax.twiny()
plt.plot(pp, qq, label='$q-p$')
plt.plot(p_array, q_array, label='Yield surface')
plt.plot(p_critical, p_critical*M, 'r-.', label='CSL')
plt.xlabel('$p$(kPa)', fontsize=15)
plt.ylabel('$q$(kPa)', fontsize=15)
# plt.axes(xscale="log")
# plt.axis('equal')
plt.legend(loc='lower right', fontsize=15)
plt.title('Conventional compression ($\sigma_2=\sigma_3=100$kPa)', fontsize=15)
plt.tight_layout()
plt.show()

# p q space
plt.plot(pp, qq)
plt.plot(p_array, q_array)
plt.plot(p_critical, p_critical*M, 'r-.')
plt.xlabel('$p$(kPa)', fontsize=15)
plt.ylabel('$q$(kPa)', fontsize=15)
# plt.axes(xscale="log")
plt.axis('equal')
plt.tight_layout()
plt.show()

# plot v-lnp
plt.plot(np.log(pp), v)
plt.xlabel('$\ln (p)$', fontsize=15)
plt.ylabel('$v$', fontsize=15)
plt.tight_layout()
plt.show()

