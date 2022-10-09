import multiprocessing
import numpy as np
from FEMxEPxML.constitutive import ConstitutiveMask
from FEMxEPxML.CSUHmodel import CSUH
from FEMxEPxML.utils_constitutive import tensor2_tensor3, returnedDatasDecode, \
    get_elasticMatrix, getVolStrain, getP, getQ, get_theta, getQEps
from utilSelf.general import echo, mapMask


class NorSandConstitutive(ConstitutiveMask):
    def __init__(self, explicitFlag, numg, pool: multiprocessing.Pool,save_path, rho,
                 p0=1e5, ndim=3, e0=0.5):
        self.pool = pool
        self.cons = [Norsand_single(p0=p0, e0=e0, explicit_flag=explicitFlag) for _ in range(numg)]
        ConstitutiveMask.__init__(self, save_path=save_path, p0=p0, ndim=ndim, explicitFlag=explicitFlag, numg=numg, rho=rho, cons=self.cons)
        self.D = np.array([self.cons[i].D for i in range(self.numg)])
        self.sig = np.array([self.cons[i].sig for i in range(self.numg)])
        self.solvers = [self.cons[i].solver for i in range(self.numg)]

    def solver(self, deps):
        if len(deps[0]) == 2:
            deps = tensor2_tensor3(t2=deps)
        param = list(zip(self.solvers, deps))
        if self.pool:
            datas = self.pool.map(mapMask, param)
        else:
            datas = []
            for i in range(self.numg):
                datas.append(mapMask(param[i]))
        if self.explicitFlag:
            sig_geo, scenes = returnedDatasDecode(explicitFlag=self.explicitFlag, datas=datas, numg=self.numg)
            self.update(scenes=scenes)
            return sig_geo
        else:
            sig_geo, D, scenes = returnedDatasDecode(explicitFlag=self.explicitFlag, datas=datas, numg=self.numg)
            return sig_geo, D, scenes

    def update(self, scenes):
        for i in range(self.numg):
            self.cons[i].update(*scenes[i])


class Norsand_single:
    '''
        Imagine state is the vertex of the cap!!!
    '''

    def __init__(self, p0=1e5, e0=0.70, lambdaa=0.05, Gamma=1.0,
                 ca=0.9, cb=0.02, cc=0., M_tc=1.2, chi_tc=3.5, N=0.3, H_0=300,
                 H_y=0., G_ref=3e4, m=1, mu=0.3, p_ref=100, explicit_flag=False):  # p's unit is kPa
        self.ca, self.cb, self.cc = ca, cb, cc
        self.lambdaa, self.Gamma = lambdaa, Gamma
        self.M_tc = M_tc
        self.chi_tc = chi_tc
        self.N = N
        self.H_0, self.H_y = H_0, H_y
        self.G_ref = G_ref  # kPa
        self.m = m
        self.mu = mu
        self.p_ref = p_ref  # kPa
        self.S = 0.5
        self.chi_tc = 3.5
        self.chi_i = self.get_chi_i()
        # ---------------------------------------------
        # state [yieldValue, p, q, sig, lam, G, D, e,theta, phi, phi_i, M_i, p_i, M, M_i_tc, eps_p]
        self.p = p0  # Pa
        self.sig = np.eye(3)*self.p
        self.q = getQ(sigma=self.sig)
        self.e = e0
        self.p_i = self.p/np.e
        self.phi = self.e-self.get_e_csl(self.p)
        self.phi_i = self.get_phi_i(phi=self.phi, p_i=self.p_i, p=self.p)
        self.theta = get_theta(sig=self.sig)
        self.M = self.get_M(theta=self.theta)
        self.M_i = self.get_M_i(M=self.M, chi_tc=self.chi_tc, phi_i=self.phi_i)
        self.M_i_tc = self.get_M_i(M=self.M_tc, chi_tc=self.chi_tc, phi_i=self.phi_i)
        self.lam, self.G = self.get_lam_G(p=self.p)
        self.D = get_elasticMatrix(lam=self.lam, G=self.G)
        self.yield_px0 = 1.
        # self.yield_px0 = self.yield_function(p=self.p, q=self.q, M_i=self.M_i, p_i=self.p_i, yield_px0=0.)
        self.yieldValue = self.yield_function(p=self.p, q=self.q, M_i=self.M_i, p_i=self.p_i, yield_px0=self.yield_px0)
        self.eps_p = np.zeros([3, 3])
        # -----------------Calculation parameters-----------------
        self.yieldTolerance = 0.05
        self.explicit_flag = explicit_flag

    def get_lam_G(self, p):
        G = self.G_ref*(p/1e3/self.p_ref)**self.m*1e3
        lam = 2.*G*self.mu/(1.-2*self.mu)
        return lam, G

    def get_e_csl(self, p):
        if self.lambdaa:
            e_csl = self.Gamma-self.lambdaa*np.log(p/1e3)
        else:
            e_csl = self.ca - self.cb * (p / 1e3 / self.p_ref) ** self.cc
        return e_csl

    def get_phi_i(self, phi, p_i, p):
        phi_i = phi + self.lambdaa*np.log(p_i/p)
        return phi_i

    def yield_function(self, p, q, M_i, p_i, yield_px0):
        f = np.log(p/p_i)+q/p/M_i-yield_px0
        return f

    def get_M_i(self, M, chi_tc, phi_i):
        M_i = M*(1.-self.N*chi_tc*np.abs(phi_i)/self.M_tc)
        return M_i

    def get_M(self, theta): # Lode's angle theta in rad
        '''
            theta =  pi/6 TC (tri-axial compression)
                  = -pi/6 TE (tri-axial extension)
        '''
        M = self.M_tc*(1.-self.M_tc/(3.+self.M_tc)*np.cos(1.5*theta+np.pi/4.))
        return M

    def get_dp_i(self, phi, p, p_i, chi_i, phi_i, M_i,M_i_tc, deps_p_q):
        p_i_max = p*np.exp(-chi_i*phi_i/M_i_tc)
        dp_i = p_i* self.get_H(phi=phi)*M_i/M_i_tc*(p/p_i)**2.*(p_i_max/p-p_i/p)*deps_p_q
        return dp_i

    def get_dp_i_inner(self, H_i, H, deps_p_v, p_i):
        dp_i = -p_i*H_i*H*np.abs(deps_p_v)
        return dp_i

    def get_H(self, phi):
        return self.H_0-self.H_y*phi

    def get_M_i_tc(self, chi_i, phi):
        M_i_tc = self.M_tc-self.N*chi_i*np.abs(phi)
        return M_i_tc

    def get_chi_i(self):
        lambdaa = self.lambdaa #!!!!!!!!!!!!!!!
        chi_i = self.M_tc*self.chi_tc/(self.M_tc-lambdaa*self.chi_tc)
        return chi_i

    def solver(self, deps):
        sig = self.sig + np.einsum('ijkl, kl->ij', self.D, deps)
        q, p = getQ(sig), getP(sig)
        if p < 0:
            print()
        yieldValue = self.yield_function(p, q, M_i=self.M_i, p_i=self.p_i, yield_px0=self.yield_px0)
        if yieldValue < 0:
            e = self.e - (1+self.e)*getVolStrain(deps)
            lam, G = self.get_lam_G(p=p)
            D = get_elasticMatrix(lam=lam, G=G)
            theta = get_theta(sig=sig)
            phi = e-self.get_e_csl(p=p)
            phi_i = self.get_phi_i(phi=phi, p_i=self.p_i, p=p)
            M = self.get_M(theta=theta)
            p_i = self.p_i
            M_i = self.get_M_i(M=M, chi_tc=self.chi_tc, phi_i=phi_i)
            M_i_tc = self.get_M_i_tc(chi_i=self.chi_i, phi=phi)
            eps_p = self.eps_p
            scence = [yieldValue, p, q, sig, lam, G, D, e, theta, phi, phi_i, M_i, p_i, M, M_i_tc, eps_p]
            if self.explicit_flag:
                return sig, scence
            else:
                return sig, D, scence
        else:
            return self.return_mapping(deps)

    def return_mapping(self, deps):
        e = self.e-(self.e+1.)*getVolStrain(deps)
        df_dsig = self.get_dfdsig(sig=self.sig, p=self.p, q=self.q, M_i=self.M_i)
        p_i_max = self.p * np.exp(-self.chi_i * self.phi_i / self.M_i_tc)
        temp1 = self.yieldValue + np.einsum('ij, ijkl, kl->', df_dsig, self.D, deps)
        temp2 = np.einsum('ij, ijkl, kl->', df_dsig, self.D, df_dsig)
        temp3 = self.get_H(phi=self.phi) / self.M_i_tc / self.p_i ** 2. * (p_i_max  - self.p_i )
        dlam = temp1/(temp2+temp3)
        deps_p = dlam*df_dsig
        sig = self.sig+np.einsum('ijkl, kl->ij', self.D, deps-deps_p)
        q, p = getQ(sigma=sig), getP(sigma=sig)
        theta = get_theta(sig=sig)
        dp_i = self.get_dp_i(
            phi=self.phi, p=self.p, p_i=self.p_i,
            chi_i=self.chi_i, phi_i=self.phi_i, M_i=self.M_i, M_i_tc=self.M_i_tc, deps_p_q=getQEps(deps_p))
        p_i = dp_i+self.p_i
        e_csl = self.get_e_csl(p)
        phi = e-e_csl
        phi_i = self.get_phi_i(phi=phi,p_i=p_i, p=p)
        M =self.get_M(theta=theta)
        M_i = self.get_M_i(M, chi_tc=self.chi_tc, phi_i=phi_i)
        lam, G = self.get_lam_G(p=p)
        D = get_elasticMatrix(lam=lam, G=G)
        yieldValue = self.yield_function(p=p, q=q, M_i=M_i, p_i=p_i, yield_px0=self.yield_px0)
        M_i_tc = self.get_M_i_tc(chi_i=self.chi_i, phi=phi)
        eps_p = self.eps_p+np.trace(deps_p)
        scence = [yieldValue, p, q, sig, lam, G, D, e, theta, phi, phi_i, M_i, p_i, M, M_i_tc, eps_p]
        Dep = self.D - np.einsum('ijmn, mn, st, stkl->ijkl', self.D, df_dsig, df_dsig, self.D) / \
                   (temp2+temp3)
        if self.explicit_flag:
            return sig, scence
        else:
            return sig, Dep, scence

    def get_dfdsig(self, sig, p, q, M_i):
        dfdp = (1.-q/p/M_i)/p
        dfdq = 1./p/M_i
        dpdsig = np.eye(3)/3.
        dqdsig = 1.5*(sig-p*np.eye(3))/q if q != 0 else 0.5*np.sqrt(3)*np.eye(3)
        dfdsig = dfdp*dpdsig+dfdq*dqdsig
        return dfdsig

    def update(self, yieldValue, p, q, sig, lam, G, D, e,theta, phi, phi_i, M_i, p_i, M, M_i_tc, eps_p):
        self.yieldValue = yieldValue
        self.p, self.q, self.sig, self.lam, self.G, \
        self.D, self.e,self.theta, self.phi, self.phi_i, self.M_i= p, q, sig, lam, G, D, e,theta,  phi, phi_i, M_i
        self.p_i = p_i
        self.M = M
        self.M_i_tc = M_i_tc
        self.eps_p = eps_p


if __name__ == '__main__':
    object_axial_strain = 0.15
    load_step = 200
    stress_total, eps_p_total = [], []
    e0_list = [0.7, 0.75, 0.77, 0.78, 0.8, 0.83]
    for e0 in e0_list:
        stress = []
        xi = []
        # deps_axial = object_axial_strain / load_step
        axialStrainArray = np.linspace(0., object_axial_strain, 1000)
        uh_single_object = Norsand_single(e0=e0)
        for i in range(1, load_step):
            deps_axial = axialStrainArray[i] - axialStrainArray[i - 1]
            deps = np.diag([-0.5 * deps_axial, -0.5 * deps_axial, deps_axial])
            sig_trial, D, scence = uh_single_object.solver(deps=deps)
            uh_single_object.update(*scence)
            print('%d yieldValue %.3e p: %.3e q: %.3e phi: %.3e phi_i: %.3e M_i:%.3e p_i: %.3e M: %.3e M_i_tc: %.3e' %
                  (i + 1, scence[0], getP(sig_trial), getQ(sig_trial),scence[-7],scence[-6],scence[-5],
                   scence[-4],scence[-3], scence[-2]))
            stress.append(sig_trial)
            xi.append(scence[-1])
        stress_total.append(np.array(stress))
        eps_p_total.append(xi)

    import matplotlib.pyplot as plt

    for i in range(len(stress_total)):
        p = np.array([getP(i) for i in stress_total[i]])
        q = np.array([getQ(i) for i in stress_total[i]])
        plt.plot(p / 1e3, q / 1e3, label=r'$e_0 = %.2f$' % e0_list[i])
    plt.axis('equal')
    plt.title('Undrained compression q-p (kPa)')
    plt.tight_layout()
    plt.legend()
    plt.show()
    # q
    for i in range(len(stress_total)):
        p = np.array([getP(i) for i in stress_total[i]])
        q = np.array([getQ(i) for i in stress_total[i]])
        plt.plot(q / 1e3, label=r'$e_0 = %.2f$' % e0_list[i])
    plt.title('Undrained compression q (kPa)')
    plt.tight_layout()
    plt.legend()
    plt.show()

    # eps_p_v
    for i in range(len(stress_total)):
        eps_v_p = np.array([getVolStrain(i) for i in eps_p_total[i]])
        plt.plot(-eps_v_p, label=r'$e_0 = %.2f$' % e0_list[i])
    plt.title(r'Undrained compression $\epsilon_{v}^{p}$')
    plt.tight_layout()
    plt.legend()
    plt.show()