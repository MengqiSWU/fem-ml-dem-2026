from __future__ import print_function

import warnings
from builtins import input
from builtins import zip
from builtins import range
from builtins import object

__author__ = "Ning Guo, ceguo@connect.ust.hk"
__supervisor__ = "Jidong Zhao, jzhao@ust.hk"
__institution__ = "The Hong Kong University of Science and Technology"

import numpy
import numpy as np

""" 3D model for multiscale simulation
which implements a Newton-Raphson scheme
into FEM framework to solve the nonlinear
problem where the tangent operator is obtained
from DEM simulation by calling simDEM modules"""

""" import Escript modules """
import esys.escript as escript
from esys.escript import util, trace, Vector, Tensor
from esys.escript.linearPDEs import LinearPDE, SolverOptions
from simDEM import *
from itertools import repeat

""" function to return pool for parallelization
    supporting both MPI (experimental) on distributed
    memory and multiprocessing on shared memory.
"""


def get_pool(mpi=False, threads=1):
    if mpi:  # using MPI
        from mpipool import MPIPool
        pool = MPIPool()
        pool.start()
        if not pool.is_master():
            sys.exit(0)
    elif threads > 1:  # using multiprocessing
        from multiprocessing import Pool
        pool = Pool(processes=threads)
    else:
        raise RuntimeError("Wrong arguments: either mpi=True or threads>1.")
    return pool


class MultiScale(object):
    """
    problem description:
    -(A_{ijkl} u_{k,l})_{,j} = -X_{ij,j} + Y_i
    Neumann boundary: n_j A_{ijkl} u_{k,l} = n_j X_{ij} + y_i
    Dirichlet boundary: u_i = r_i where q_i > 0
    :var u: unknown vector, displacement
    :var A: elastic tensor / tangent operator
    :var X: old/current stress tensor
    :var Y: vector, body force
    :var y: vector, Neumann bc traction
    :var q: vector, Dirichlet bc mask
    :var r: vector, Dirichlet bc value
    """

    def __init__(self, domain, ng=1, useMPI=False, np=6, random=False, rtol=1.e-2, verbose=True, mpi_pool=None,
                 mode='mcc', frictionalAngle=30):
        """
        initialization of the problem, i.e. model constructor
        :param domain: type Domain, domain of the problem
        :param ng: type integer, number of Gauss points
        :param useMPI: type boolean, use MPI or not
        :param np: type integer, number of processors
        :param random: type boolean, if or not use random density field
        :param rtol: type float, relevant tolerance for global convergence
        :param verbose: type boolean, if or not print messages during calculation
        """
        self.mode = mode

        self.domain = domain
        self.pde = LinearPDE(domain, numEquations=self.domain.getDim(), numSolutions=self.domain.getDim())
        try:
            self.pde.getSolverOptions().setSolverMethod(SolverOptions.DIRECT)
        except:
            # import time
            print("=======================================================================")
            print("For better performance compile python-escript with direct solver method")
            print("=======================================================================")
            input("Press Enter to continue...")
            # time.sleep(5)
        self.pde.setSymmetryOn()
        # self.pde.getSolverOptions().setTolerance(rtol**2)
        # self.pde.getSolverOptions().setPackage(SolverOptions.UMFPACK)
        self.numGaussPoints = ng
        self.rtol= rtol
        self.verbose = verbose
        self.pool = mpi_pool if useMPI else get_pool(mpi=useMPI, threads=np)
        self.strain = escript.Tensor(0, escript.Function(self.domain))
        self.stress = escript.Tensor(0, escript.Function(self.domain))
        self.S = escript.Tensor4(0, escript.Function(self.domain))
        self.volume = trace(Tensor(1, escript.Function(self.domain)))/3.

        if 'dem' in self.mode:
            self.scenes = self.pool.map(initLoad3D, list(range(ng)))
            st = self.pool.map(getStressAndTangent3D, self.scenes)
            for i in range(ng):
                self.stress.setValueOfDataPoint(i, st[i][0])
                self.S.setValueOfDataPoint(i, st[i][1])
        if 'mcc' in self.mode:
            from FEMxEPxML.MCCmodel import MCCmodel
            self.mathSolver = [MCCmodel(mode='math', verboseFlag=False) for _ in range(self.numGaussPoints)]
            # sigma_index = [0, 1, 4]
            stress = numpy.array([i.sig for i in self.mathSolver])
            S = numpy.array([i.De for i in self.mathSolver])
            self.pc0 = numpy.array([i.pc0 for i in self.mathSolver])
            self.stress, self.S = self.setStressAndMatrix3D(stress, S)
        if 'csuh' in self.mode:
            from FEMxEPxML.CSUHmodel import CSUH
            self.mathSolver = [CSUH(verboseFlag=False, p0=1e5, e0=0.7, chi=0.0) for _ in range(self.numGaussPoints)]
            # sigma_index = [0, 1, 4]
            stress = numpy.array([-i.sigma for i in self.mathSolver])
            S = numpy.array([i.D for i in self.mathSolver])
            self.pc0 = numpy.array([i.px0 for i in self.mathSolver])
            self.stress, self.S = self.setStressAndMatrix3Dtensor(stress, S)
        if 'mises' in self.mode or 'lade' in self.mode:
            from FEMxEPxML.MisesModelTensor import MisesAssociateFlowIsoHarden
            self.mathSolver = [MisesAssociateFlowIsoHarden(verboseFlag=False, p0=1e5,
                ladeFlag=True if 'lade' in self.mode else False, frictionalAngle=frictionalAngle) for _ in range(self.numGaussPoints)]
            # sigma_index = [0, 1, 4]
            stress_geo = numpy.array([i.sig for i in self.mathSolver])
            S = numpy.array([i.D for i in self.mathSolver])
            self.stress, self.S = self.setStressAndMatrix3Dtensor(-stress_geo, S)

    def initialize(self, b=escript.Data(), f=escript.Data(), specified_u_mask=escript.Data(),
                   specified_u_val=escript.Data()):
        """
        initialize the model for each time step, e.g. assign parameters
        :param b: type vector, body force on FunctionSpace, e.g. gravity
        :param f: type vector, boundary traction on FunctionSpace (FunctionOnBoundary)
        :param specified_u_mask: type vector, mask of location for Dirichlet boundary
        :param specified_u_val: type vector, specified displacement for Dirichlet boundary
        """
        self.pde.setValue(Y=b, y=f, q=specified_u_mask, r=specified_u_val)

    def setStressAndMatrix3D(self, sig, stiffness):
        if stiffness.shape[1] != 6 or stiffness.shape[2] != 6 or sig.shape[1] != 6:
            raise ValueError('The shape of stiffness is (%d, %d) and the length of stress is (%d,)' %
                             (stiffness.shape[1], stiffness.shape[2], sig.shape[1]))
        # initialization stress and stiffness in format of escript data
        stress = escript.Tensor(0, escript.Function(self.domain))
        S = escript.Tensor4(0, escript.Function(self.domain))

        stiffness_list = []
        for i in range(self.numGaussPoints):
            t = np.zeros(shape=(3, 3, 3, 3))
            '''
                stiffness in Voigt Notion 
                    [[0000 0011 0022 0001 0012 0020],
                     [1100 1111 1122 1101 1112 1120],
                     [2200 2211 2222 2201 2212 2220],
                     [0100 0111 0122 0101 0112 0120],
                     [1200 1211 1222 1201 1212 1220],
                     [2000 2011 2022 2001 2012 2020]]
            '''
            # 0 0
            t[0, 0, 0, 0] = stiffness[i, 0, 0]
            t[0, 0, 1, 1] = t[1, 1, 0, 0] = stiffness[i, 0, 1]
            t[0, 0, 2, 2] = t[2, 2, 0, 0] = stiffness[i, 0, 2]
            t[0, 0, 0, 1] = t[0, 1, 0, 0] = t[1, 0, 0, 0] = t[0, 0, 1, 0] = stiffness[i, 0, 3]
            t[0, 0, 1, 2] = t[1, 2, 0, 0] = t[2, 1, 0, 0] = t[0, 0, 2, 1] = stiffness[i, 0, 4]
            t[0, 0, 0, 2] = t[0, 2, 0, 0] = t[2, 0, 0, 0] = t[0, 0, 2, 0] = stiffness[i, 0, 5]
            # 1 1
            t[1, 1, 1, 1] = stiffness[i, 1, 1]
            t[1, 1, 2, 2] = t[2, 2, 1, 1] = stiffness[i, 1, 2]
            t[1, 1, 0, 1] = t[0, 1, 1, 1] = t[1, 0, 1, 1] = t[1, 1, 1, 0] = stiffness[i, 1, 3]
            t[1, 1, 1, 2] = t[1, 2, 1, 1] = t[2, 1, 1, 1] = t[1, 1, 2, 1] = stiffness[i, 1, 4]
            t[1, 1, 0, 2] = t[0, 2, 1, 1] = t[2, 0, 1, 1] = t[1, 1, 2, 0] = stiffness[i, 1, 5]
            # 2 2
            t[2, 2, 2, 2] = stiffness[i, 2, 2]
            t[2, 2, 0, 1] = t[0, 1, 2, 2] = t[1, 0, 2, 2] = t[2, 2, 1, 0] = stiffness[i, 2, 3]
            t[2, 2, 1, 2] = t[1, 2, 2, 2] = t[2, 1, 2, 2] = t[2, 2, 2, 1] = stiffness[i, 2, 4]
            t[2, 2, 0, 2] = t[0, 2, 2, 2] = t[2, 0, 2, 2] = t[2, 2, 2, 0] = stiffness[i, 2, 5]
            # 0 1
            t[0, 1, 0, 1] = t[0, 1, 1, 0] = t[1, 0, 0, 1] = t[1, 0, 1, 0] = stiffness[i, 3, 3]
            t[0, 1, 1, 2] = t[1, 2, 0, 1] = t[2, 1, 0, 1] = t[0, 1, 2, 1] = stiffness[i, 3, 4]
            t[0, 1, 0, 2] = t[0, 2, 0, 1] = t[2, 0, 0, 1] = t[0, 1, 2, 0] = stiffness[i, 3, 5]
            # 1 2
            t[1, 2, 1, 2] = t[1, 2, 2, 1] = t[2, 1, 1, 2] = t[2, 1, 2, 1] = stiffness[i, 4, 4]
            t[1, 2, 0, 2] = t[1, 2, 2, 0] = t[2, 0, 1, 2] = t[0, 2, 1, 2] = stiffness[i, 4, 5]
            # 2 0
            t[2, 0, 0, 2] = t[2, 0, 2, 0] = t[0, 2, 2, 0] = t[0, 2, 0, 2] = stiffness[i, 5, 5]
            stiffness_list.append(t)
        # transform the stress and the stiffness into the form of escript.data
        for i in range(self.numGaussPoints):
            ''' sig: in Voigh notion [00 11 22 01 12 20] '''
            temp = [[sig[i, 0], sig[i, 3], sig[i, 5]],
                    [sig[i, 3], sig[i, 1], sig[i, 4]],
                    [sig[i, 5], sig[i, 4], sig[i, 2]]]
            S.setValueOfDataPoint(i, stiffness_list[i])
            stress.setValueOfDataPoint(i, temp)
        # # debug
        # eps = np.random.random(6)
        # eps33 = np.array([[eps[0], 0.5 * eps[3], 0.5 * eps[5]],
        #                   [0.5 * eps[3], eps[1], 0.5 * eps[4]],
        #                   [0.5 * eps[5], 0.5 * eps[4], eps[2]]])
        # s1 = stiffness[0] @ eps
        # s2 = np.einsum('ijkl, kl->ij', stiffness_list[0], eps33)
        return stress, S

    def setStressAndMatrix3Dtensor(self, sig, stiffness):
        # if stiffness.shape[1] != (3, 3, 3, 3) or stiffness.shape[2] != 6 or sig.shape[1] != 6:
        #     raise ValueError('The shape of stiffness is (%d, %d) and the length of stress is (%d,)' %
        #                      (stiffness.shape[1], stiffness.shape[2], sig.shape[1]))
        # initialization stress and stiffness in format of escript data
        stress = escript.Tensor(0, escript.Function(self.domain))
        S = escript.Tensor4(0, escript.Function(self.domain))
        for i in range(self.numGaussPoints):
            S.setValueOfDataPoint(i, stiffness[i])
            stress.setValueOfDataPoint(i, sig[i])
        return stress, S

    def getDomain(self):
        """
        return model domain
        """
        return self.domain

    def getRelTolerance(self):
        """
        return relative tolerance for convergence
        type float
        """
        return self.rtol

    def getCurrentPacking(self, pos=(), time=0, prefix=''):
        if len(pos) == 0:
            # output all Gauss points packings
            self.pool.map(outputPack, list(zip(self.scenes, repeat(time), repeat(prefix))))
        else:
            # output selected Gauss points packings
            scene = [self.scenes[i] for i in pos]
            self.pool.map(outputPack, list(zip(scene, repeat(time), repeat(prefix))))

    def getLocalVoidRatio(self):
        void = escript.Scalar(0, escript.Function(self.domain))
        e = self.pool.map(getVoidRatio3D, self.scenes)
        for i in range(self.numGaussPoints):
            void.setValueOfDataPoint(i, e[i])
        return void

    def getLocalAvgRotation(self):
        rot = escript.Vector(0, escript.Function(self.domain))
        r = self.pool.map(avgRotation3D, self.scenes)
        for i in range(self.numGaussPoints):
            rot.setValueOfDataPoint(i, r[i])
        return rot

    def getLocalFabric(self):
        fabric = escript.Tensor(0, escript.Function(self.domain))
        f = self.pool.map(getFabric3D, self.scenes)
        for i in range(self.numGaussPoints):
            fabric.setValueOfDataPoint(i, f[i])
        return fabric

    def getCurrentTangent(self):
        """
        return current tangent operator
        type Tensor4 on FunctionSpace
        """
        return self.S

    def getCurrentStress(self):
        """
        return current stress
        type: Tensor on FunctionSpace
        """
        return self.stress

    def getCurrentStrain(self):
        """
        return current strain
        type: Tensor on FunctionSpace
        """
        return self.strain

    def exitSimulation(self):
        """finish the whole simulation, exit"""
        self.pool.close()

    def solve(self, iter_max=10):
        converge = True
        if 'dem' in self.mode:
            u = self.solveDEM(iter_max=iter_max)
        elif 'mcc' in self.mode:
            u = self.solveMCC(iter_max=iter_max)
        elif 'csuh' in self.mode:
            u, converge = self.solveCSUH(iter_max=iter_max)
        elif 'mises' in self.mode or 'lade' in self.mode:
            u, converge = self.solveMises(iter_max=iter_max)
        else:
            raise ValueError('No solve mode: "%s"' % self.mode)
        return u, converge

    def solveDEM(self, iter_max=100):
        """
        solve the equation using Newton-Ralphson scheme
        """
        iterate = 0
        rtol = self.getRelTolerance()
        stress = self.getCurrentStress()
        s = self.getCurrentTangent()
        x_safe = self.domain.getX()
        self.pde.setValue(A=s, X=-stress)
        # residual0=util.L2(self.pde.getRightHandSide()) # using force error
        u = self.pde.getSolution()  # trial solution, displacement
        D = util.grad(u)  # trial strain tensor
        # !!!!!! obtain stress and tangent operator from DEM part
        update_stress, update_s, update_scenes = self.applyStrain_getStressTangentDEM(st=D)
        err = 1.0  # initial error before iteration
        converged = (err < rtol)
        while (not converged) and (iterate < iter_max):
            if self.verbose:
                print("\tNot converged after %d iteration(s)! Relative error: %e" % (iterate, err))
            iterate += 1
            self.domain.setX(x_safe + u)
            self.pde.setValue(A=update_s, X=-update_stress, r=escript.Data())
            # residual=util.L2(self.pde.getRightHandSide())
            du = self.pde.getSolution()
            u += du
            l, d = util.L2(u), util.L2(du)
            err = d / l  # displacement error, alternatively using force error 'residual'
            converged = (err < rtol)
            if err > rtol * 0.001:  # only update DEM parts when error is large enough
                self.domain.setX(x_safe)
                D = util.grad(u)
                update_stress, update_s, update_scenes = self.applyStrain_getStressTangentDEM(st=D)

            # if err>err_safe: # to ensure consistent convergence, however this may not be achieved due to fluctuation!
            #   raise RuntimeError, "No improvement of convergence with iterations! Relative error: %e"%err
        """
        update 'domain geometry', 'stress', 'tangent operator',
        'accumulated strain' and 'simulation scenes'.
        """
        self.domain.setX(x_safe + u)
        self.stress = update_stress
        self.S = update_s
        self.strain += D
        self.scenes = update_scenes
        if self.verbose:
            print("\tConvergence reached after %d iteration(s)! Relative error: %e" % (iterate, err))
        return u

    def solveMCC(self, iter_max=100):
        """
        solve the equation using Newton-Ralphson scheme
        """
        iterate = 0
        rtol = self.getRelTolerance()
        x_safe = self.domain.getX()
        self.pde.setValue(A=self.S, X=-self.stress)
        # residual0=util.L2(self.pde.getRightHandSide()) # using force error
        u = self.pde.getSolution()  # trial solution, displacement
        D = util.grad(u)  # trial strain tensor
        iteration, stress, S, update_scenes = self.getMCCStressAndTangent(D=D)

        # used for debugging
        # deps = numpy.array(D.toListOfTuples())
        # deps = 0.5 * (deps + deps.transpose([0, 2, 1]))
        # matrix = numpy.array(self.S.toListOfTuples())
        # dsig = numpy.einsum('nijkl,nkl->nij', matrix, deps)
        # dsig_mcc = numpy.array((stress - self.stress).toListOfTuples())
        # sig_new = numpy.array((self.stress).toListOfTuples()) + dsig
        # residual = dsig_mcc - dsig


        err = 1.0  # initial error before iteration
        converged = (err < rtol)
        while (not converged) and (iterate < iter_max):
            if self.verbose:
                print("\tNot converged after %d iteration(s)! Relative error: %e" % (iterate, err))
            iterate += 1
            self.domain.setX(x_safe + u)
            self.pde.setValue(A=S, X=-stress, r=escript.Data())
            # residual=util.L2(self.pde.getRightHandSide())
            du = self.pde.getSolution()
            u += du
            l, d = np.average(numpy.linalg.norm(np.array(u.toListOfTuples()), axis=1)), \
                   np.average(numpy.linalg.norm(np.array(du.toListOfTuples()), axis=1))
            err = d / l  # displacement error, alternatively using force error 'residual'
            converged = (err < rtol)
            if err > rtol * 0.001:  # only update DEM parts when error is large enough
                self.domain.setX(x_safe)
                D = util.grad(u)
                iteration, stress, S, update_scenes = self.getMCCStressAndTangent(D=D)

            # if err>err_safe: # to ensure consistent convergence, however this may not be achieved due to fluctuation!
            #   raise RuntimeError, "No improvement of convergence with iterations! Relative error: %e"%err
        """
        update 'domain geometry', 'stress', 'tangent operator',
        'accumulated strain' and 'simulation scenes'.
        """
        self.domain.setX(x_safe + u)
        self.stress = stress
        self.S = S
        self.strain += D
        self.scenes = update_scenes
        self.pc0 = update_scenes[5]
        # update the state
        itemNum = len(update_scenes)
        updata_scencesT = [[update_scenes[i][j] for i in range(itemNum)] for j in range(self.numGaussPoints)]
        self.pool.map(updateStateMask,
            list(zip(self.mathSolver, updata_scencesT)))
        self.Strain_increment = D
        # self.strain_abs += current_strain_abs
        # self.frobeniusNorm += frobeniusNorm
        self.volume = self.volume * (1 + trace(D))
        if self.verbose:
            print("Convergence reached after %d iteration(s)! Relative error: %e" % (iterate, err))
        return u

    def solveCSUH(self, iter_max=10):
        """
        solve the equation using Newton-Ralphson scheme
        """
        iterate = 0
        rtol = self.getRelTolerance()
        x_safe = self.domain.getX()
        self.pde.setValue(A=self.S, X=-self.stress)
        # used for debugging
        # p = numpy.sort(numpy.einsum('ijj->i', numpy.array(self.stress.toListOfTuples())) / 3.)
        # D0000 = numpy.sort(numpy.array(self.S.toListOfTuples())[:, 0, 0, 0, 0])
        # D1111 = numpy.sort(numpy.array(self.S.toListOfTuples())[:, 1, 1, 1, 1])
        # D2222 = numpy.sort(numpy.array(self.S.toListOfTuples())[:, 2, 2, 2, 2])

        # TODO
        '''
        Add a function to distinguish the loading or unloading
        
        if loading, use the D_ep, else use the D_e to calculate the trial strain 
        '''
        # residual0=util.L2(self.pde.getRightHandSide()) # using force error
        u = self.pde.getSolution()  # trial solution, displacement
        D = util.grad(u)  # trial strain tensor
        stress, S, update_scenes = self.getCSUHStressAndTangent(D=D)
        if update_scenes == False:
            return u, False
        # p = numpy.sort(numpy.einsum('ijj->i', numpy.array(stress.toListOfTuples())) / 3.)
        # D0000 = numpy.sort(numpy.array(S.toListOfTuples())[:, 0, 0, 0, 0])
        # D1111 = numpy.sort(numpy.array(S.toListOfTuples())[:, 1, 1, 1, 1])
        # D2222 = numpy.sort(numpy.array(S.toListOfTuples())[:, 2, 2, 2, 2])

        # used for debugging
        # deps = numpy.array(D.toListOfTuples())
        # deps = 0.5 * (deps + deps.transpose([0, 2, 1]))
        # matrix = numpy.array(self.S.toListOfTuples())
        # # matrix = numpy.array(S.toListOfTuples())
        # dsig = numpy.einsum('nijkl,nkl->nij', matrix, deps)
        # dsig_mcc = numpy.array((stress - self.stress).toListOfTuples())
        # sig_new = numpy.array((self.stress).toListOfTuples()) + dsig
        # residual = dsig_mcc - dsig

        err = 1.0  # initial error before iteration
        converged = (err < rtol)
        while (not converged) and (iterate < iter_max):
            if self.verbose:
                print("\tNot converged after %d iteration(s)! Relative error: %e" % (iterate, err))
            iterate += 1
            if iterate >= iter_max:
                return u, False
            self.domain.setX(x_safe + u)
            self.pde.setValue(A=S, X=-stress, r=escript.Data())
            # residual=util.L2(self.pde.getRightHandSide())
            du = self.pde.getSolution()
            u += du
            l, d = np.average(numpy.linalg.norm(np.array(u.toListOfTuples()), axis=1)), \
                   np.average(numpy.linalg.norm(np.array(du.toListOfTuples()), axis=1))
            err = d / l  # displacement error, alternatively using force error 'residual'
            converged = (err < rtol)
            if err > rtol * 0.001:  # only update DEM parts when error is large enough
                self.domain.setX(x_safe)
                D = util.grad(u)
                stress, S, update_scenes = self.getCSUHStressAndTangent(D=D)
                if update_scenes == False:
                    return u, False
                # p = numpy.sort(numpy.einsum('ijj->i', numpy.array(stress.toListOfTuples())) / 3.)
                # D0000 = numpy.sort(numpy.array(S.toListOfTuples())[:, 0, 0, 0, 0])
                # D1111 = numpy.sort(numpy.array(S.toListOfTuples())[:, 1, 1, 1, 1])
                # D2222 = numpy.sort(numpy.array(S.toListOfTuples())[:, 2, 2, 2, 2])
                # print()
        """
        update 'domain geometry', 'stress', 'tangent operator',
        'accumulated strain' and 'simulation scenes'.
        """
        self.domain.setX(x_safe + u)
        self.stress = stress
        self.S = S
        self.strain += D
        self.scenes = update_scenes
        # update the state
        itemNum = len(update_scenes)
        updata_scencesT = [[update_scenes[i][j] for i in range(itemNum)] for j in range(self.numGaussPoints)]
        '''
        sig, e, p, q, q_ts, xi_ts, yieldValue, H, epsvp
        '''
        # self.pool.map(updateStateMask,
        #     list(zip(self.mathSolver, updata_scencesT)))
        aa = list(zip(self.mathSolver, updata_scencesT))
        for i in range(self.numGaussPoints):
            updateStateMask(param=aa[i])
        self.Strain_increment = D
        # self.strain_abs += current_strain_abs
        # self.frobeniusNorm += frobeniusNorm
        self.volume = self.volume * (1 + trace(D))
        if self.verbose:
            print("Convergence reached after %d iteration(s)! Relative error: %e" % (iterate, err))
        return u, True

    def solveMises(self, iter_max=10):
        """
                solve the equation using Newton-Ralphson scheme
                """
        iterate = 0
        rtol = self.getRelTolerance()
        x_safe = self.domain.getX()
        self.pde.setValue(A=self.S, X=-self.stress)
        u = self.pde.getSolution()  # trial solution, displacement
        D = util.grad(u)  # trial strain tensor
        stress, S, update_scenes = self.getMisesStressAndTangent(D=D)
        if update_scenes == False:
            return u, False

        deps = np.array(D.toListOfTuples())
        deps = 0.5*(deps+deps.transpose(0, 2, 1))
        eps = np.array(self.strain.toListOfTuples())
        D_material = np.array(self.S.toListOfTuples())
        sig = np.array(self.stress.toListOfTuples())
        sig_Cal = np.einsum('nijkl, nkl->nij', D_material, eps)+np.eye(3)*(-1e5)
        sig_residua = sig-sig_Cal

        sig_new = np.array(stress.toListOfTuples())
        D_material_new = np.array(S.toListOfTuples())
        D_residual = D_material_new-D_material
        sig_cal = sig + np.einsum('nijkl, nkl->nij', D_material, deps)
        sig_residua_new = sig_new-sig_cal

        err = 1.0  # initial error before iteration
        converged = (err < rtol)
        while (not converged) and (iterate < iter_max):
            if self.verbose:
                print("\tNot converged after %d iteration(s)! Relative error: %e" % (iterate, err))
            iterate += 1
            if iterate >= iter_max:
                return u, False
            self.domain.setX(x_safe + u)
            self.pde.setValue(A=S, X=-stress, r=escript.Data())
            # residual=util.L2(self.pde.getRightHandSide())
            du = self.pde.getSolution()
            u += du
            l, d = np.average(numpy.linalg.norm(np.array(u.toListOfTuples()), axis=1)), \
                   np.average(numpy.linalg.norm(np.array(du.toListOfTuples()), axis=1))
            err = d / l  # displacement error, alternatively using force error 'residual'
            converged = (err < rtol)
            if err > rtol * 0.001:  # only update DEM parts when error is large enough
                self.domain.setX(x_safe)
                D = util.grad(u)
                stress, S, update_scenes = self.getMisesStressAndTangent(D=D)
                if update_scenes == False:
                    return u, False
        """
        update 'domain geometry', 'stress', 'tangent operator',
        'accumulated strain' and 'simulation scenes'.
        """
        self.domain.setX(x_safe + u)
        self.stress = stress
        self.S = S
        self.strain += D
        self.scenes = update_scenes
        # update the state
        itemNum = len(update_scenes)
        updata_scencesT = [[update_scenes[i][j] for i in range(itemNum)] for j in range(self.numGaussPoints)]
        '''
        sig, deps, yieldValue, epsPlastic, eps_plasticVector
        '''
        aa = list(zip(self.mathSolver, updata_scencesT))
        for i in range(self.numGaussPoints):
            updateStateMask(param=aa[i])
        self.Strain_increment = D
        # self.strain_abs += current_strain_abs
        # self.frobeniusNorm += frobeniusNorm
        self.volume = self.volume * (1 + trace(D))
        if self.verbose:
            print("Convergence reached after %d iteration(s)! Relative error: %e" % (iterate, err))
        return u, True

    def getMCCStressAndTangent(self, D):
        st = numpy.array(D.toListOfTuples())
        st = 0.5 * (st + st.transpose(0, 2, 1)).reshape(self.numGaussPoints, 9)  # tensor notion
        # make sure in a order of [00 11 01]
        st = numpy.concatenate((st[:, 0:1], st[:, 4:5], st[:, 8:9],
                                2.*st[:, 1:2], 2.*st[:, 5:6], 2.*st[:, 2:3]), axis=1)  # Voigt notion
        # temp = self.pool.map(SolverMask, list(zip(self.mathSolver, st)))
        # temp = []
        # for i in range(self.numGaussPoints):
            # temp.append(SolverMask([self.mathSolver[i], st[i]]))
        temp = self.pool.map(SolverMask, list(zip(self.mathSolver, st)))
        iteration, sig, materialMatrix, dEps_plastic_p, dEps_plastic_q, yieldValue, pc0 = \
            numpy.array([temp[i][0] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][1] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][2] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][3] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][4] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][5] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][6] for i in range(self.numGaussPoints)])
        # materialMatrix = numpy.array([[materialMatrix[i][0, 0], materialMatrix[i][0, 1],
        #                                materialMatrix[i][1, 0], materialMatrix[i][1, 1],
        #                                materialMatrix[i][3, 3]] for i in range(self.numGaussPoints)])
        update_scenes = [sig, st, yieldValue, dEps_plastic_p, dEps_plastic_q, pc0]
        stress, S = self.setStressAndMatrix3D(sig, materialMatrix)
        return iteration, stress, S, update_scenes

    def getCSUHStressAndTangent(self, D):
        st = numpy.array(D.toListOfTuples())
        '''
        Reset the strain Cuz compression is positive and the extension is negative in geomaterial computations
        '''
        st_symmetric = -0.5 * (st + st.transpose(0, 2, 1))

        param = list(zip(self.mathSolver, st_symmetric))
        # temp = []
        # for i in range(self.numGaussPoints):
        #     temp.append(SolverMask(param=param[i]))
        temp = self.pool.map(SolverMask, param)
        if False in temp:
            return False, False, False
        geo_sig, e, p, q, xi_ts, yieldValue, H, epsvp, D_ep = \
            numpy.array([temp[i][0] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][1] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][2] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][3] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][4] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][5] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][6] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][7] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][8] for i in range(self.numGaussPoints)])
        # materialMatrix = numpy.array([[materialMatrix[i][0, 0], materialMatrix[i][0, 1],
        #                                materialMatrix[i][1, 0], materialMatrix[i][1, 1],
        #                                materialMatrix[i][3, 3]] for i in range(self.numGaussPoints)])
        update_scenes = [geo_sig, e, p, q, xi_ts, yieldValue, H, epsvp]
        stress, S = self.setStressAndMatrix3Dtensor(-geo_sig, D_ep)
        p = np.array([numpy.trace(i)/3. for i in geo_sig])
        if any(p < 0.):
            s = 'Mean stress is negtive p=%.3e ' % np.sort(p)[0]
            warnings.warn(s)
        return stress, S, update_scenes

    def getMisesStressAndTangent(self, D):
        st = numpy.array(D.toListOfTuples())
        '''
        Reset the strain Cuz compression is positive and the extension is negative in geomaterial computations
        '''
        st_symmetric = -0.5 * (st + st.transpose(0, 2, 1))

        param = list(zip(self.mathSolver, st_symmetric))
        # temp = []
        # for i in range(self.numGaussPoints):
        #     temp.append(SolverMask(param=param[i]))
        temp = self.pool.map(SolverMask, param)
        if False in temp:
            return False, False, False
        iteration, geo_sig, D_ep, epsPlastic_vector, epsPlastic, yieldValue = \
            numpy.array([temp[i][0] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][1] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][2] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][3] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][4] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][5] for i in range(self.numGaussPoints)])
        update_scenes = [geo_sig, st_symmetric, yieldValue, epsPlastic, epsPlastic_vector]
        stress, S = self.setStressAndMatrix3Dtensor(-geo_sig, D_ep)
        return stress, S, update_scenes

    def applyStrain_getStressTangentDEM(self, st=escript.Data()):
        """
            apply strain to DEM packing,
            get stress and tangent operator (including two methods)
            """
        st = st.toListOfTuples()
        st = numpy.array(st).reshape(-1, 9)
        stress = escript.Tensor(0, escript.Function(self.domain))
        S = escript.Tensor4(0, escript.Function(self.domain))
        scenes = self.pool.map(shear3D, list(zip(self.scenes, st)))
        st = self.pool.map(getStressAndTangent3D, scenes)
        for i in range(self.numGaussPoints):
            stress.setValueOfDataPoint(i, st[i][0])
            S.setValueOfDataPoint(i, st[i][1])
        return stress, S, scenes


def SolverMask(param):
    '''iteration, sig, materialMatrix, eps, epsPlastic, epsPlasticVector'''
    misesClass, st = param[0], param[1]
    return misesClass.solver(st)


def updateStateMask(param):
    mathSolver, update_scenes = param[0], param[1]
    mathSolver.updateState(*update_scenes)
