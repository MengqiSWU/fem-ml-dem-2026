import matplotlib.pyplot as plt
import numpy as np

epsilon1 = np.linspace(0., 0.2, 100)
E0, sigma_ult = 1e8, 1e6
sigma_d = epsilon1 / (1 / E0 + epsilon1 / sigma_ult)

fig = plt.figure()
axes = fig.gca()
plt.plot(epsilon1, sigma_d)
plt.text(0.025, 4e5, s='$E_0=%.2f$' % (E0), fontsize=15)
plt.text(0.025, 5e5, s='$\sigma_{ult}=%.2f$' % (sigma_ult), fontsize=15)
plt.xlabel('$Axial strain$', fontsize=15)
plt.ylabel('$\sigma_{1}-\sigma_{3}$', fontsize=15)
plt.show()
