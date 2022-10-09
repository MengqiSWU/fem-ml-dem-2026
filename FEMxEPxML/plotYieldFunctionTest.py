import numpy as np
import matplotlib.pyplot as plt

M = 1.25  # ratio at critical state
lambdaa = 0.135
kappa = 0.04
poisson = 0.3
N = 1.973  # location of asymptotic line in the e-lnp plane
e0 = 0.8
e_c0 = 0.934
px0 = 100.
c_p = (lambdaa-kappa)/(1+e0)
epsvp = [0, 0.001, 0.01, 0.02, 0.03]

'''
    Plot the q-p plane
    Eq. (1)
'''
for i in epsvp:
    px = px0 * np.exp(i / c_p)
    p = np.linspace(0.0, px, 100)
    plt.plot(p, M*np.sqrt((p*px-p**2.)), label='yield $\epsilon_v^p=%.3f$' % i)
q_crit = p*M
plt.plot(p, q_crit, label='critical')
plt.axis('equal')
plt.xlabel('p')
plt.ylabel('q')
plt.tight_layout()
plt.legend()
plt.show()
plt.close()

# plot the e-lnp curves
'''
 e_c0 is the void ratio at \eta = 0  p = 0
 Z    is the void ratio at \eta = 0. p = 1kPa(in the paper)  while p = 1 Pa (in this code)
'''
pmax = 1000e3
ps = np.exp((N - e_c0) / lambdaa)
Z = N-lambdaa*np.log(np.exp((N-e_c0)/lambdaa)+1.)
lnp = np.linspace(0, np.log(pmax), 101)
p = np.exp(lnp)
e = N-lambdaa*lnp
plt.plot(lnp, e, label='e-lnp')
e_curved = Z-lambdaa*np.log((p+ps)/(1+ps))
plt.plot(lnp, e_curved, label='$e_{curved}-\mathrm{ln}(p)$')
lnps = np.log(ps)
e_ps = Z-lambdaa*np.log((ps+ps)/(1+ps))
plt.scatter([lnps], [e_ps])
plt.text(x=lnps, y=e_ps, s='(%.2f kPa, %.4f)' % (ps, e_ps))
plt.xlabel('ln$(p)$')
plt.xlim([0, np.log(pmax)])
plt.legend()
plt.tight_layout()

'''
Plot the e_eta while eta=M 

Eq. (30)
'''
chiList = [0., 0.4, 0.7]
for chi in chiList:
    e_eta = Z-lambdaa*np.log((p+ps)/(1.+ps))-(lambdaa-kappa)*np.log(((2/(1-chi))*p+ps)/(p+ps))
    plt.plot(lnp, e_eta, label='$\chi=$%.2f' % chi)
plt.xlabel('ln$(p)$')
plt.legend()
plt.title('NCL, CSL ($\chi_{1}, \chi_{2}, \chi_{3}$)')
plt.tight_layout()
plt.show()
plt.close()



