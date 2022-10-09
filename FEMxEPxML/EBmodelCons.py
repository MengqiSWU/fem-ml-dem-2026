import matplotlib.pyplot as plt
import numpy as np

from FEMxEPxML.constitutive import ConstitutiveMask
from FEMxEPxML.nonlinearCons import nonlinearCons
from FEMxEPxML.utils_constitutive import get_principle_stress, get_elasticMatrix


class EBmodelConstitutive(ConstitutiveMask):
    def __init__(self, ndim, explicit_flag, numg, save_flag, **kwarg):
        cons = [EBmodelConSingle(p0=kwarg['p0'], fai0=kwarg['fric'],
                                 ndim=ndim, explicit_flag=explicit_flag) for _ in range(numg)]
        ConstitutiveMask.__init__(
            self, p0=kwarg['p0'], ndim=ndim, explicitFlag=explicit_flag,
            numg=numg, save_path=kwarg['save_path'], cons=cons, name='eb', save_flag=save_flag, rho=kwarg['rho'])


class EBmodelConSingle(nonlinearCons):
    def __init__(self, p0, ndim, explicit_flag, K=704, n=0.38, Rf=0.9, C=110,
                 fai0=30., d_fai=0., Kb=303, m=0.18, Kur=844.8, Pa=1e5):
        nonlinearCons.__init__(self, p0=p0, ndim=ndim, explicit_flag=explicit_flag)

        # initialise the material constants
        self.K = K  # torch.Variable(1e6, )
        self.n = n  # torch.Variable(1e6, )
        self.Rf = Rf  # torch.Variable(1e6, )
        self.C = C  # torch.Variable(1e6, )
        self.fai0 = fai0  # torch.Variable(1e6, )
        self.d_fai = d_fai  # torch.Variable(1e6, )
        self.Kb = Kb  # torch.Variable(1e6, )
        self.m = m  # torch.Variable(1e6, )
        self.Kur = Kur  # torch.Variable(1e6, )
        self.Pa = Pa
        self.S_max = 0
        self.s1_s3_max = 0  # check if unloading
        self.p, self.q = p0, 0.
        _, self.D = self.cal_E_mu(p=p0, q=0, fai=self.fai0)
        self.D_load = self.D.copy()

    def solver(self, deps):
        sig_trial = np.einsum("ijkl, kl->ij", self.D_load, deps) + self.sig
        ps_trial = np.sort(get_principle_stress(sig_trial[:self.ndim, :self.ndim]))
        # p, q = getP(self.sig), getQ(self.sig)
        p_trial, q_trial = ps_trial[0], ps_trial[1] - ps_trial[0]
        # fai = self.fai0 - self.d_fai * np.log10(ps[2] / self.Pa)  # note: use sigma_3 or p
        if self.p < 0:
            fai = self.fai0
        else:
            fai = self.fai0 - self.d_fai * np.log10(self.p / self.Pa)  # note: use sigma_3 or p
        S, D = self.cal_E_mu(p_trial, q_trial, fai)
        sig = np.einsum("ijkl, kl->ij", D, deps) + self.sig

        ps = np.sort(get_principle_stress(sig[:self.ndim, :self.ndim]))
        p = ps[0]
        s1_s3_max = q = ps[1] - ps[0]
        scene = [sig, self.eps + deps, self.eps_abs + np.abs(deps),
                 [S, s1_s3_max, p, q, D]]
        if self.explcit_flag:
            return sig, scene
        else:
            return sig, D, scene

    def cal_E_mu(self, p, q, fai):
        fai_rad = fai / 180. * np.pi
        s1_s3_f = (2. * self.C * np.cos(fai_rad) + 2. * p * np.sin(fai_rad)) \
                  / (1 - np.sin(fai_rad))
        if s1_s3_f <= 0:
            S = 1
        else:
            S = q / s1_s3_f
        if p <= 0. or self.p <= 0.:
            S = 1.0
            return S, np.zeros([3, 3, 3, 3])

        if S < self.S_max and q < self.s1_s3_max:
            E = self.Kur * self.Pa * (self.p / self.Pa) ** self.n
        else:
            E = (1 - self.Rf * S) ** 2 * self.K * self.Pa * (self.p / self.Pa) ** self.n

        Bt = self.Kb * self.Pa * (self.p / self.Pa) ** self.m
        # nu = 0.5 - E / 6 / Bt
        # nu = min(max(nu, 0.), 0.5)
        # print(Bt)
        nu = min(max((3. * Bt - E) / 6. / Bt, 0.), 0.48)
        # if (3.*Bt-E) <= 0:
        #     echo('Caution: E=%.3e Bt=%.3e, and 3.*Bt-E=%.3e' % (E, Bt, (3.*Bt-E)))
        #     Bt=E/3.
        G = E / 2. / (1 + nu)
        lam = E * nu / (1 + nu) / (1 - 2 * nu)
        # G = 3.*Bt*E/(9.*Bt-E)
        # lam = 3.*Bt*(3.*Bt-E)/(9.*Bt-E)
        D = get_elasticMatrix(lam=lam, G=G)
        return S, D

    def update_internal(self, S, s1_s3_max, p, q, D):
        if S > self.S_max:
            self.S_max = S
        if s1_s3_max > self.s1_s3_max:
            self.s1_s3_max = s1_s3_max
        self.p = p
        self.q = q
        if self.D[0, 0, 0, 0] != 0:
            self.D_load = self.D
        self.D = D


if __name__ == '__main__':
    eb_model = EBmodelConSingle(p0=1e5, ndim=3, explicit_flag=False)
    step_num = 1001
    eps_array = np.linspace(0, 0.1, step_num)
    sig_list, eps_list = [], []
    for i in range(1, step_num):
        deps_axial = eps_array[i] - eps_array[i - 1]
        deps = np.diag([-deps_axial * 0.5, -deps_axial * 0.5, deps_axial])
        sig, D, scene = eb_model.solver(deps)
        sig_list.append(sig)
        eps_list.append(scene[1])
        eb_model.update(*scene)
    print()
    sig_array = np.array(sig_list)
    plt.plot(eps_array[1:], sig_array[:, 2, 2])
    plt.show()
