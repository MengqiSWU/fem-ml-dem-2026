from train_model import get_data
from matplotlib import pyplot as plt
import numpy as np
import random

x, y, x_scalar, y_scalar, number_points = get_data()
x_origin = x_scalar.inverse_transform(x)
y_origin = y_scalar.inverse_transform(y)
strain = x_origin
epsilon_22 = x_origin[:, 3]
deviatoric_strain = np.zeros(shape=(len(strain), len(strain[0])))
deviatoric_strain[:, 0] = strain[:, 0] - (strain[:, 0]+strain[:, 3])*0.5
deviatoric_strain[:, 1] = strain[:, 1] - (strain[:, 0]+strain[:, 3])*0.5
shear_strain = np.sqrt(np.sum(deviatoric_strain*deviatoric_strain, axis=1)*1.5)
stress_low_bound = 500000
stress = np.power(10, y_origin[:, :4])-stress_low_bound
deviatoric_stress = np.zeros(shape=(len(stress), len(stress[0])))
deviatoric_stress[:, 0] = stress[:, 0] - (stress[:, 0]+stress[:, 3])*0.5
deviatoric_stress[:, 1] = stress[:, 1] - (stress[:, 0]+stress[:, 3])*0.5
shear_stress = np.sqrt(np.sum(deviatoric_stress*deviatoric_stress, axis=1)*1.5)

index = random.randint(0, 100)
stress2 = [shear_stress[i] for i in range(index, len(shear_stress), number_points)]
strain2 = [shear_strain[i] for i in range(index, len(shear_strain), number_points)]
epsilon_sinle_22 = [epsilon_22[i] for i in range(index, len(epsilon_22), number_points)]

# shear_stress vs deviatoric_strain
fig = plt.figure(figsize=[6.4*2, 4.8*2])
axes = fig.gca()
plt.scatter(strain2, stress2)
for i in range(len(strain2)):
    plt.text(strain2[i], stress2[i], str(i))
plt.title('$\sigma_{deviatoric}$ vs $\epsilon_{deviatoric}$ Guassion point %d' % index)
plt.show()

# epsilon_22
fig = plt.figure(figsize=[6.4*4, 4.8*4])
axes = fig.gca()
plt.scatter(range(len(epsilon_sinle_22)), epsilon_sinle_22)
for i in range(len(epsilon_sinle_22)):
    plt.text(i, epsilon_sinle_22[i], str(i))
plt.title('$\epsilon_{22}$ vs $time step$ Guassion point %d' % index, fontsize=15)
plt.show()
