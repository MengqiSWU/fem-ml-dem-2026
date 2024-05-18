from esys.escript import Vector, Solution, grad, trace, transpose, kronecker, \
    length, interpolate, FunctionOnBoundary, matrix_mult, \
    whereZero, whereNegative, integrate, sup, inf, sqrt, inner, L2, \
    Tensor, Tensor4, Function, Data, Scalar, symmetric, sign
from esys.escript.pdetools import Locator, Projector
from esys.escript.linearPDEs import LinearPDE, SolverOptions
import numpy as np
from FEMxDEM.escriptSolver import escriptSolver
from utilSelf.saveGauss import save_loading
from utilSelf.general import echo


class ImplicitSolver(escriptSolver):
    def __init__(self, domain, cons, loadInfor, tolerance=1e-2, save_loading_flag=False):
        escriptSolver.__init__(self, domain=domain, cons=cons,
                               loadInfor=loadInfor, explicitFlag=False)
        self.tolerance = tolerance
        self.save_loading_flag = save_loading_flag


    def solve(self, t, iter_max):
        x_safe = self.domain.getX()   # 只有顶部变形已知，其他区域变形量未知
        u = self.pde.getSolution()   # 根据已知边界位移边界力和初始刚度矩阵(弹性矩阵)计算内部和两侧节点位移
        self.pde.setValue(r=Data())  # set the displacement to 0, after the 1st solution4；
        deps = symmetric(grad(u))  # 计算内部节点应变
        # d_deps = symmetric(grad(u)) - self.eps_last
        sig_data, D_data, scenes = self.stressSolver_3d(deps=deps)  # 更新高斯点应力和对应的stiffness矩阵
        iterate = 0
        if self.save_loading_flag:
            self.save_loading_mask(t=t, sig_data=sig_data, D_data=D_data, u_grad=deps, scenes=scenes, iterate=iterate)
        end_flag = False
        while True:
            self.domain.setX(x_safe+u) # 更新内部和两侧节点位移(顶部位移未变化)
            self.pde.setValue(A=D_data, X=-sig_data) # 更新高斯点应力和stiffness矩阵
            du = self.pde.getSolution()  # 再次求解内部和两侧节点位移

            try:
                d, l = L2(du), L2(u)
                if d > 1e2:
                    echo('*'*60,
                         'The strain increments is too big, d=%.2e' % d,
                         'fatal error risen!',
                         )
                    end_flag = True
                err = d/l
                converge = True if err < self.tolerance else False
            except:
                converge = True
                err = 1e-2
            if end_flag:
                raise
            if converge:
                print('Converged at iter %d with err=%.3e' % (iterate, err))
                break
            else:
                print('\t Iter %d d=%.3e l=%.3e err=%.3e' % (iterate, d, l, err))
                if iterate > iter_max:
                    print('Ended at max Iter ')
                    break
            self.domain.setX(x_safe)
            u += du
            deps = symmetric(grad(u))
            sig_data, D_data, scenes = self.stressSolver_3d(deps=deps)
            if self.save_loading_flag:
                self.save_loading_mask(t=t, sig_data=sig_data, D_data=D_data, u_grad=deps, scenes=scenes, iterate=iterate)
            iterate += 1

        self.u += u
        self.sig, self.D = sig_data, D_data
        self.eps += deps
        self.eps_abs += deps*sign(deps)
        self.volume = self.volume*(1+trace(deps))
        self.updateCons(scenes=scenes)

    def save_loading_mask(self, t, sig_data, D_data, u_grad, scenes, iterate=None):   # du_grad,
        save_dict = {
            'sig': sig_data,
            'D': D_data,
            'eps': self.eps + symmetric(u_grad),
            'eps_abs': self.eps_abs + u_grad*sign(u_grad),
            'H_3D': self.setHistVector_3D(scenes[2]),
            # 'H_2D': self.setHistVector_3D(scenes[2]),
        }
        save_loading(save_path=self.savePath, t=t, iter=iterate,  **save_dict)




