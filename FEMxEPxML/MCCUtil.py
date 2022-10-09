import os
import numpy as np
import sympy as sym
from matplotlib import pyplot as plt
import warnings


def getP(sigma):
    p = np.trace(sigma) / 3.
    # if p < 0:
    #     warnings.warn('Mean stress can not be negative while iteration, p=%3.e' % p)
    #     p = p0
    return p


def getS(sigma):
    return sigma-np.eye(3)*getP(sigma)


def getJ2(sigma):
    s = getS(sigma)
    return 0.5*np.sum(s*s)


def getQ(sigma):
    J2 = getJ2(sigma)
    return np.sqrt(3. * J2)


def getVolStrain(eps):
    return np.trace(eps)


def getEpsDevitoric(eps):
    return eps - np.eye(3)*getVolStrain(eps)/3.


def getJ2Eps(eps):
    e = getEpsDevitoric(eps)
    return 0.5*np.sum(e*e)


def getQEps(eps):
    J2eps = getJ2Eps(eps)
    return 2. / 3. * np.sqrt(3. * J2eps)


def getdEpsMagtitude(deps):
    return np.sqrt(2.*np.sum(deps*deps)/3.)


def getPrinciple(sigma):
    if sigma.shape == (3, 3):# tensor notion
        sigma_matrix = sigma
    else: # if stress in voigt notion
        sigma_matrix = np.array([[sigma[0], sigma[3], 0.],
                             [sigma[3], sigma[1], 0.],
                             [0., 0., sigma[2]]])
    return np.linalg.eigvals(sigma_matrix)


def get_dqeps_deps(eps):
    eps_s = getS(eps)
    j2eps = getJ2Eps(eps)
    dqeps_dj2 = 1./np.sqrt(3.*j2eps) if j2eps!=0. else 2./np.sqrt(3.)
    dj2deps_s = eps_s
    dqeps_deps = dqeps_dj2*np.einsum('ij, ijkl->kl', dj2deps_s, dsdsigma)
    return dqeps_deps


def get_dpdsig_dqdsigma(sigma):
    s = getS(sigma)
    p, q = getP(sigma), getQ(sigma)
    dpdsig = np.eye(3)/3.
    dqdJ2 = 1.5/q if q != 0. else 1.
    dJ2ds = s
    # dqdsigma = dqdJ2*dJ2ds*dsdsigma
    dqdsig = dqdJ2*np.einsum('ij, ijkl->kl', dJ2ds, dsdsigma)
    return dpdsig, dqdsig


dsdsigma = np.zeros([3, 3, 3, 3])
for i in range(3):
    for j in range(3):
        if i == j:
            dsdsigma[i, j, i, j] = 2./3.
        else:
            dsdsigma[i, j, i, j] = 1.
            dsdsigma[j, i, i, j] = 1.
            dsdsigma[j, i, j, i] = 1.
            dsdsigma[i, j, j, i] = 1.
dsdsigma[0, 0, 1, 1] = dsdsigma[1, 1, 2, 2] = dsdsigma[0, 0, 2, 2] = -1./3.
dsdsigma[1, 1, 0, 0] = dsdsigma[2, 2, 1, 1] = dsdsigma[2, 2, 0, 0] = -1./3.


def getInvariantsSigma(sigma):
    ''' https://en.wikipedia.org/wiki/Cauchy_stress_tensor '''
    I1 = np.trace(sigma)
    I2 = 0.5*(np.trace(sigma)**2.-np.trace(sigma**2.))
    I3 = np.linalg.det(sigma)
    return I1, I2, I3


def get_qc_smp(sigma):
    i1, i2, i3 = getInvariantsSigma(sigma)
    temp1 = i1*i2-i3
    temp2 = i1*i2-9.*i3
    if temp2 == 0.:
        return 0.
    temp = temp1/temp2
    if temp < 0.:
        return 0.
    qc = 2.*i1/(3.*np.sqrt(temp)-1.)
    return qc


def get_dqc_di(sig):
    x, y, z = getInvariantsSigma(sig)
    temp1 = y*x-z
    temp2 = y*x-9*z
    if temp2 <= 0.:
        print('I1=%.3f I2=%.3f I3=%.3f' % (x, y, z))
        print('I1*I2-9I3=%.3f' % temp2)
        return np.array([1., 1., 1.])
    temp3 = temp1/temp2
    if temp3 <= 0.:
        return np.array([1., 1., 1.])
    temp4 = np.sqrt(temp3)  # overflow
    dqc_di1 = 2./(3*temp4-1.)-3*x*y*(1-temp3)/temp2/(temp4*(3*temp4-1.)**2.)
    dqc_di2 = 24.*x**2*z/(temp2**2*temp4*(3*temp4-1.)**2.)
    dqc_di3 = -24.*x**2*y/((3*temp4-1.)**2.*temp4*temp2**2.)
    return np.array([dqc_di1, dqc_di2, dqc_di3])


def get_di_dsig(sig):
    '''
        https://en.wikipedia.org/wiki/Tensor_derivative_(continuum_mechanics)
    '''
    i1, i2, i3 = getInvariantsSigma(sig)
    di1_dsig = np.eye(3)
    di2_dsig = i1*np.eye(3)-sig.T
    di3_dsig = (sig*sig-i1*sig+i2*np.eye(3)).T
    # di3_dsig = i2 * np.eye(3) - sig.T @ (i1 * np.eye(3) - sig.T)
    return np.array([di1_dsig, di2_dsig, di3_dsig])


def getSigma_ts(sigma, p, q, qc):
    if q == 0.:
        return sigma
    sigma_ts = p*np.eye(3)+qc/q*(sigma-np.eye(3)*p)
    return sigma_ts


def get_b(sigma1, sigma2, sigma3):
    b = (sigma3-sigma2)/(sigma3-sigma1) if sigma3-sigma1 != 0. else 0.
    return b


def getLode(b):
    return np.arctan((1-2.*b)/np.sqrt(3.))


def getMaterialMatrix(lam, G):
    matrix = np.zeros(shape=[3, 3, 3, 3])
    for i in range(3):
        for j in range(3):
            matrix[i, i, j, j] += lam
    for i in range(3):
        matrix[i, i, i, i] += 2. * G
        matrix[i, (i + 1) % 3, i, (i + 1) % 3] = \
            matrix[i, (i + 1) % 3, (i + 1) % 3, i] = \
            matrix[(i + 1) % 3, i, (i + 1) % 3, i] = \
            matrix[(i + 1) % 3, i, i, (i + 1) % 3] = G
    return matrix


def voigt2tensor(vector, epsFlag):
    """
    vector: 00 11 22 01 12 20
    """
    scaler = 1.0
    if epsFlag:
        scaler = 0.5
    tensor = np.array([[vector[0], vector[3]*scaler, vector[5]*scaler],
                    [vector[3]*scaler, vector[1], vector[4]*scaler],
                    [vector[5]*scaler, vector[4]*scaler, vector[2]]])
    return tensor


def tensor2voigt(tensor, epsFlag):
    """
    vector: 00 11 22 01 12 20
    """
    scaler = 1.0
    if epsFlag:
        scaler = 2.
    vector = np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2],
                       scaler*tensor[0, 1], scaler*tensor[1, 2], scaler*tensor[2, 0]])
    return vector


def plotSubFigures(ax, x, y, label, xlabel, ylabel, num=None, color=None):
    if num and num >= 1:
        for i in range(num):
            if color:
                ax.plot(x[i], y[i], label=label[i], lw=3, alpha=0.5, color=color)
            else:
                ax.plot(x[i], y[i], label=label[i], lw=3, alpha=0.5)
    else:
        raise ValueError('Please give the num')
    plt.legend(fontsize=15)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()


def loadingPathReader(path='MCCData'):
    path = os.path.join(path, 'loadingPath')
    fileList = [os.path.join(path, i) for i in os.listdir(path) if '.dat' in i]
    loadPathList = []
    for i in fileList:
        pathTemp = np.loadtxt(fname=i, delimiter=',', skiprows=1)
        loadPathList.append(pathTemp)
    return loadPathList
