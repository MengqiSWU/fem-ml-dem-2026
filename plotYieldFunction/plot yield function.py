import numpy as np
import matplotlib.pyplot as plt
from FEMxEPxML.MCCUtil import getQ,getP,getInvariantsSigma,getLode,getPrinciple
"""

@file: GNST.py
@time: 2019/11/22 16:28
@author: Luke
@email: guanshaoheng@qq.com
@application：

               1.变换应力空间
               2.generalized non-linear strength theory 广义非线性强度理论

               【参考文献：
               1. Yaoyangping, 2004, General non-linear strength theory and transformed stress space
               
               2. Matsuoka H, Yao YP, Sun D (1999) The Cam-clay models revised by the SMP criterion. 
               Soils Found 39:81–95. https://doi.org/10.3208/sandf.39.81
               
               3. Da Fontoura SAB (2012) Lade and modified lade 3D rock strength criteria. 
               Rock Mech Rock Eng 45:1001–1006. https://doi.org/10.1007/s00603-012-0279-1

"""

from numpy import sin, pi, sqrt
import numpy as np
from matplotlib import pyplot as plt


def get_b(sigma):
    sigma_sorted = sorted(sigma, reverse=True)
    b = (sigma_sorted[1]-sigma_sorted[2])/(sigma_sorted[0]-sigma_sorted[2])
    return sigma_sorted, b


def get_c(sigma_sorted):
    c = max(sigma_sorted)-min(sigma_sorted)
    return c


# 求解主应力
def get_sigma(p, q, theta):
    # 偏应力第二不变量j2 = q**2/3.0
    sigma = np.empty([3])
    sigma[0] = p + 2.0 / 3.0 * q * sin(theta + 2.0 / 3.0 * pi)
    sigma[1] = p + 2.0 / 3.0 * q * sin(theta)
    sigma[2] = p + 2.0 / 3.0 * q * sin(theta - 2.0 / 3.0 * pi)
    return np.array(sigma)


# Mises 准则对应的最大偏应力
def get_qm(sigma):
    [sigma1, sigma2, sigma3] = [sigma[0], sigma[1], sigma[2]]
    i1 = sum([sigma1, sigma2, sigma3])
    i2 = sigma1 * sigma2 + sigma2 * sigma3 + sigma1 * sigma3
    qm = sqrt(i1 ** 2 - 3. * i2)
    return qm


# smp 准则对应的最大偏应力
def get_qs(sigma):
    '''
    Reference:

    1. Matsuoka H, Yao YP, Sun D (1999) The Cam-clay models revised by the SMP criterion.
    Soils Found 39:81–95. https://doi.org/10.3208/sandf.39.81
    '''
    [sigma1, sigma2, sigma3] = [sigma[0], sigma[1], sigma[2]]
    i1 = sum([sigma1, sigma2, sigma3])
    i2 = sigma1 * sigma2 + sigma2 * sigma3 + sigma1 * sigma3
    i3 = sigma1 * sigma2 * sigma3
    qs = 2. * i1 / (3. * sqrt((i1 * i2 - i3) / (i1 * i2 - 9. * i3)) - 1.)
    return qs


def get_q_alpha(sigma, alpha):
    q_alpha = get_qm(sigma) * alpha + (1 - alpha) * get_qs(sigma)
    return q_alpha


def get_qs_sigma(p=10, phi=30):
    '''
    Reference:

    1. Matsuoka H (1976) on the Significance of the ″Spatial Mobilized Plane″ .
    Soils Found 16:91–100. https://doi.org/10.3208/sandf1972.16.91
    '''
    phi_rad = phi/180*np.pi
    const_compression = (9.-np.sin(phi_rad)**2.)/(1.-np.sin(phi_rad)**2.)
    theta_array = np.linspace(-pi/6, pi/6., 180)
    # 求出 SMP 准则下的sigma
    qs_sigma = []
    for i, theta in enumerate(theta_array):
        q = 0.
        sigma = get_sigma(p, q, theta)
        while get_smp(sigma, const=const_compression) < 0.:
            q += 0.001*p
            sigma = get_sigma(p, q, theta)
        qs_sigma.append(sigma)
    qs_sigma = np.array(qs_sigma)
    return qs_sigma


def get_smp(sigma, const):
    i1 = np.sum(sigma)
    i2 = sigma[0]*sigma[1]+sigma[1]*sigma[2]+sigma[0]*sigma[2]
    i3 = sigma[0]*sigma[1]*sigma[2]
    return i1*i2/i3-const


def get_qa_sigma(c, p, alpha):
    theta_array = np.linspace(-pi/6, pi/6., 180)
    # 求出 SMP 准则下的sigma
    qa_sigma = []
    for i, theta in enumerate(theta_array):
        q_uesed = c
        sigma = get_sigma(p, c, theta)
        while get_q_alpha(sigma, alpha) > c:
            q_uesed -= 0.001*c
            sigma = get_sigma(p, q_uesed, theta)
        qa_sigma.append(sigma)
    qa_sigma = np.array(qa_sigma)
    return qa_sigma


def get_lade_sigma(p=10, phi=30):
    phi_rad = phi*np.pi/180
    eta = 4.*np.tan(phi_rad)**2.*(9.-7.*np.sin(phi_rad))/(1-np.sin(phi_rad))
    theta_array = np.linspace(-pi/6, pi/6., 180)
    lade_sigma = []
    # qm_sigma = np.array([get_sigma(p, q, i) for i in theta_array])
    for theta in theta_array:
        q_used = 0.
        sigma = get_sigma(p, q_used, theta)
        while get_lade(sigma, eta) <= 0:
            q_used += 0.001 * p
            sigma = get_sigma(p, q_used, theta)
        lade_sigma.append(sigma)
    lade_sigma = np.array(lade_sigma)
    return lade_sigma


def get_lade(sigma, eta):
    '''
        According to the modified Lade criterion

        https://www.youtube.com/watch?v=bf8DtRAFfEE&ab_channel=PGE334ReservoirGeomechanics
    '''
    [sigma1, sigma2, sigma3] = [sigma[0], sigma[1], sigma[2]]
    i1 = sum([sigma1, sigma2, sigma3])
    # i2 = sigma1 * sigma2 + sigma2 * sigma3 + sigma1 * sigma3
    i3 = sigma1 * sigma2 * sigma3
    lade = i1**3./i3-27-eta
    return lade


def get_mohr_sigma(p, phi_rad):
    theta_array = np.linspace(-pi/6, pi/6., 180)
    sigma_list = []
    for theta in theta_array:
        q = 0
        sigma = get_sigma(p, q, theta)
        while get_mohr(sigma, phi_rad=phi_rad) < 0:
            q += 0.001*p
            sigma = get_sigma(p, q, theta)
        sigma_list.append(sigma)
    return sigma_list


def get_mohr(sigma, phi_rad):
    maxii, minii = np.max(sigma), np.min(sigma)
    v = (maxii-minii) - (minii+maxii)*np.sin(phi_rad)
    return v


def get_mises_sigma(p, phi_rad):
    M_compression = 6. * np.sin(phi_rad) / (3 - np.sin(phi_rad))
    theta_array = np.linspace(-pi/6, pi/6., 180)
    sigma_list = []
    for theta in theta_array:
        q = 0
        sigma = get_sigma(p, q, theta)
        while get_mises(sigma, M=M_compression) < 0:
            q += 0.001 * p
            sigma = get_sigma(p, q, theta)
        sigma_list.append(sigma)
    return sigma_list


def get_mises(sigma, M):
    p = np.average(sigma)
    s = sigma-p
    q_0 = np.sqrt(3.*0.5*np.sum(s*s))
    # q = get_qc(sigma)
    # print('q_0: %.3f q_c: %.3f' % (q_0, q))
    v = q_0-M*p
    return v


def get_qc(sigma):
    '''
        Transformed stress space:

        Reference:
        1. Matsuoka H, Yao YP, Sun D (1999) The Cam-clay models revised by the SMP criterion.
        Soils Found 39:81–95. https://doi.org/10.3208/sandf.39.81

    '''
    i1, i2, i3 = getInvariants(sigma=sigma)
    if (i1*i2-9.*i3) == 0:
        return 0.
    qc = 2.*i1/(3.*np.sqrt((i1*i2-i3)/(i1*i2-9.*i3))-1.)
    return qc


def getInvariants(sigma):
    i1 = np.sum(sigma)
    i2 = sigma[0]*sigma[1]+sigma[2]*sigma[1]+sigma[0]*sigma[2]
    i3 = sigma[0]*sigma[1]*sigma[2]
    return i1, i2, i3


def hybrid_failure_surface():
    phi = 30
    p = 10.
    phi_rad = phi / 180 * np.pi
    M_compression = 6. * np.sin(phi_rad) / (3 - np.sin(phi_rad))
    q = M_compression*p
    theta_array = np.linspace(-pi/6, pi/6., 180)
    label_list = []
    sigma = []

    # mises
    qm_sigma = np.array(get_mises_sigma(p=p, phi_rad=phi_rad))
    label_list += ['Mises']
    sigma += [qm_sigma]

    # mohr-coulomb surface
    qmohr_sigma = np.array(get_mohr_sigma(p=p, phi_rad=phi_rad))
    label_list += ['Mohr-Coulomb']
    sigma += [qmohr_sigma]

    # 求出 SMP 准则下的sigma
    qs_sigma = get_qs_sigma(p=10, phi=phi)
    label_list += ['SMP']
    sigma += [qs_sigma]
    #
    # Lade criterion
    lade_sigma = get_lade_sigma(p=10, phi=30)
    label_list += ['Lade']
    sigma += [lade_sigma]

    # 求出 混合 准则下的sigma
    # alpha_1, alpha_2 = 0.5, 0.8
    # qa_sigma_1 = get_qa_sigma(c=10, p=10, alpha=alpha_1)
    # qa_sigma_2 = get_qa_sigma(c=10, p=10, alpha=alpha_2)
    # label_list += ['Hybrid alpha=%.2f' % alpha_1, 'Hybrid alpha=%.2f' % alpha_2]
    # sigma += [qa_sigma_1, qa_sigma_2]

    # 将图作在pi平面上
    r = 13.
    plt.figure()
    for i, sigma_ in enumerate(sigma):
        p = np.sum(sigma_, axis=1).reshape(-1, 1) / 3.
        sigma_deviatoric = sigma_ - p
        x = (sigma_deviatoric[:, 1] - sigma_deviatoric[:, 2]) * 3. ** 0.5 / 2.
        y = sigma_deviatoric[:, 0] - (sigma_deviatoric[:, 1] + sigma_deviatoric[:, 2]) / 2.
        plt.plot(x, y, label=label_list[i], alpha=0.8)
    line1 = plt.plot([0, 0.], [0, r], 'k-')[0]  # axis sigma_1
    # line2 = plt.plot([0, r*np.sin(np.pi/3)], [0, r*np.cos(np.pi/3)], 'k-')[0]
    plt.annotate("", xy=(0*1.02, r*1.02), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->"))
    plt.annotate("", xy=(r*np.sin(np.pi/3)*1.02, r*np.cos(np.pi/3)*1.02), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->"))
    plt.text(x=r*np.sin(np.pi/3), y=r*np.cos(np.pi/3),
             s='$%d^{\circ}$' % int(np.pi/3/np.pi*180), fontsize=12)
    plt.text(x=-0.8, y=13-0.5, s='$\sigma_1$', fontsize=12)
    # plt.text(x=11+0.5, y=11./np.sqrt(3.), s='$60^{\circ}$', fontsize=12)
    for theta in np.linspace(np.pi/12, np.pi/3-np.pi/12, 3):
        plt.plot([0, r*np.sin(theta)], [0, r*np.cos(theta)], 'k-.')
        plt.text(x=r*np.sin(theta), y=r*np.cos(theta),
                 s='$%d^{\circ}$' % int(theta/np.pi*180), fontsize=12)

    plt.legend(fontsize=12)
    plt.axis('equal')
    plt.title('Failure surface on $\pi$ plane')
    plt.tight_layout()
    plt.show()


def meridian_plane():
    mf = 0.8
    # n = 0.8
    pr = 10.
    c = 0.
    p = np.linspace(-c, 20, 1000)
    plt.figure()
    for n in [0., 0.2, 0.5, 0.8, 1.]:
        q = mf*((p+c)/pr)**n*pr
        plt.plot(p, q, label='mf=%.2f n=%.2f pr=%.2f c=%.2f' % (mf, n, pr, c), alpha=0.5)
    plt.plot([-c, pr-c], [0, mf*pr])
    plt.scatter(pr-c, mf*pr, linewidths=2)
    plt.xlim([-c-1, max(p)])
    plt.xlabel('p')
    plt.ylabel('q')
    plt.legend()
    plt.title('Failure surface on meridian q-p plane')
    plt.show()


def lade_surface():
    lade_sigma = get_lade_sigma()
    p = np.sum(lade_sigma, axis=1).reshape(-1, 1) / 3.
    lade_sigma_deviatoric  = lade_sigma-p
    # 将图作在pi平面上
    plt.figure()
    x = (lade_sigma_deviatoric[:, 1] - lade_sigma_deviatoric[:, 2]) * 3. ** 0.5 / 2.
    y = lade_sigma_deviatoric[:, 0] - (lade_sigma_deviatoric[:, 1] + lade_sigma_deviatoric[:, 2]) / 2.
    plt.plot(x, y, label='Lade', alpha=0.8)
    plt.legend()
    plt.axis('equal')
    plt.title('Lade Failure surface on pi plane')
    plt.show()


if __name__ == '__main__':
    print(__doc__)
    hybrid_failure_surface()
    # meridian_plane()
    # lade_surface()
