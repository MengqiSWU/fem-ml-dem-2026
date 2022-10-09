import multiprocessing
import numpy as np
from utilSelf.general import echo, mapMask
from FEMxEPxML.utils_constitutive import tensor2_tensor3, get_elasticMatrix, returnedDatasDecode
from utilSelf.saveGauss import save_loading


class ConstitutiveMask:
    def __init__(self, p0, ndim, explicitFlag, numg, save_path: str,
                 cons=None, pool=None, name='general', save_flag=False, rho=2650.):
        '''
            Caution: in geo-mechanics, compression is positive (opposite to the general mechanics)
        '''
        self.p0 = p0
        self.pool = pool
        self.ndim = ndim
        self.explicitFlag = explicitFlag
        self.save_flag = save_flag
        self.save_path = save_path
        self.rho = rho
        self.t = 0
        if p0 < 0:
            echo('Caution: in geo-mechanics, '
                 '         compression is positive '
                 '         (opposite to the general mechanics)')
            raise ValueError
        self.numg = numg
        self.eps = np.zeros(shape=(numg, 3, 3))
        self.internal = None
        self.name = name
        self.cons = cons
        if self.cons is not None:
            self.solvers = [self.cons[i].solver for i in range(self.numg)]
            self.sig = np.array([self.cons[i].sig for i in range(self.numg)])
            if not self.explicitFlag:
                self.D = np.array([self.cons[i].D for i in range(self.numg)])
        else:
            pass
        if self.name == 'csuh':
            self.save_loading_mask(sig_geo=self.sig, eps_geo=self.eps[:, :self.ndim, :self.ndim])
        elif self.name == 'vonmises':
            self.save_loading_mask(
                sig_geo=self.sig, eps_geo=self.eps[:, :self.ndim, :self.ndim],
                H_1=[self.cons[i].H for i in range(numg)])
        else:
            pass
        self.t += 1

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
            sig_geo, scenes = returnedDatasDecode(
                explicitFlag=self.explicitFlag, datas=datas, numg=self.numg, name=self.name)
            if self.save_flag and self.t % 10 == 0:
                H_1 = None
                eps_geo = np.array([scenes[i][1][:self.ndim, :self.ndim] for i in range(self.numg)])
                if 'vonmises' in self.name:
                    H_1= np.array([scenes[i][3][3] for i in range(self.numg)])
                self.save_loading_mask(sig_geo=sig_geo, H_1=H_1, eps_geo=eps_geo)
            self.update(scenes=scenes)
            self.t += 1
            return sig_geo
        else:
            sig_geo, D, scenes = returnedDatasDecode(
                explicitFlag=self.explicitFlag, datas=datas, numg=self.numg)
            return sig_geo, D, scenes

    def update(self, scenes):
        for i in range(self.numg):
            self.cons[i].update(*scenes[i])

    def return2initial(self):
        if type(self.cons) is list:
            for i in range(len(self.cons)):
                self.cons[i].return2initial()

    def save_loading_mask(self, sig_geo, eps_geo, H_1=None):
        kwargs = {
            'sig_last': -np.array([self.cons[i].sig for i in range(self.numg)])[:, :self.ndim, :self.ndim],
            'sig': -sig_geo[:, :self.ndim, :self.ndim],
            'eps': -eps_geo[:, :self.ndim, :self.ndim],
            'eps_abs': np.array([self.cons[i].eps_abs for i in range(self.numg)])[:, :self.ndim, :self.ndim],
        }
        if 'vonmises' in self.name:
            '''
                scene = [sig_trial, self.eps + deps, yieldValue, self.eps_p, self.eps_s_p, self.H]
            '''
            kwargs['H_0'] = np.array([self.cons[i].H for i in range(self.numg)])
            kwargs['H_1'] = H_1
        elif 'csuh' in self.name or 'eb' in self.name:
            pass
        else:
            echo('Waiting to add the %s model\'s saving manipulation' % self.name)
            raise
        save_loading(save_path=self.save_path, t=self.t, **kwargs)
        return

class constitutiveSingle:
    def __init__(self, p0: float, ndim: int):
        self.p0 = p0
        self.sig = np.eye(3)*self.p0
        self.eps = np.zeros(shape=[3, 3])
        self.eps_p = np.zeros(shape=[3, 3])
        self.ndim = ndim
        self.eps_abs = np.zeros(shape=[3, 3])

    def update(self, sig, eps, eps_abs, internals):
        self.sig = sig
        self.eps = eps
        self.eps_abs = eps_abs
        self.update_internal(*internals)

    def dsigCal(self, D, deps):
        dsig = np.einsum('ijkl, kl->ij', D, deps)
        return dsig

    def yieldFunction(self):
        pass

    def hardeningFunction(self):
        pass

    def dfdsig(self):
        pass

    def dgdsig(self):
        pass

    def dfdH(self):
        pass

    def transformSplit(self, deps):
        pass

    def plasticReturnMapping(self, deps):
        pass

    def update_internal(self, internals):
        pass