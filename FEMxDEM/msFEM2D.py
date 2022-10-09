from __future__ import print_function

import os
from builtins import input
from builtins import zip
from builtins import range
from builtins import object

__author__ = "Ning Guo, ceguo@connect.ust.hk"
__supervisor__ = "Jidong Zhao, jzhao@ust.hk"
__institution__ = "The Hong Kong University of Science and Technology"

import numpy

""" 2D model for multiscale simulation
which implements a Newton-Raphson scheme
into FEM framework to solve the nonlinear
problem where the tangent operator is obtained
from DEM simulation by calling simDEM modules"""

# import Escript modules
# import tensorflow.compat.v1 as tf
import esys.escript as escript
from esys.escript import util, Vector, Solution, trace, Function, transpose, Tensor, kronecker
from esys.escript.linearPDEs import LinearPDE, SolverOptions
from esys.weipa import saveVTK
from utilSelf.saveGauss import saveGauss2D
# import YADE modules
from FEMxDEM.simDEM import *
# from simDEM import shear2D, getFabric2D, getStressAndTangent2D, getVoidRatio2D, getEquivalentPorosity, avgRotation2D
# other python modules
from itertools import repeat
import numpy as np
import time


# from train_model import restore # , get_stress_tengent !!!! check why can not import get_stress_tengent
# from FEMxML.network_stiffness import RestoreNet
# from network_stiffness_complex import RestoreNet
# import network
# initialization of the ml model


def get_pool(mpi=False, threads=1):
    """ function to return pool for parallelization
        supporting both MPI (experimental) on distributed
        memory and multiprocessing on shared memory.
    """
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

    def __init__(self, domain, ng=1, useMPI=False, np=1, random=False, rtol=1e-2, usePert=False, pert=-2.e-6,
                 verbose=True, loadInfor=None, mode='ml', ml_model_path=None, saved_model=None,
                 mpi_pool=None):
        """
        initialization of the problem, i.e. model constructor
        :param domain: type Domain, domain of the problem
        :param ng: type integer, number of Gauss points
        :param useMPI: type boolean, use MPI or not
        :param np: type integer, number of processors
        :param random: type boolean, if or not use random density field
        :param rtol: type float, relevative tolerance for global convergence
        :param usePert: type boolean, if or not use perturbation method
        :param pert: type float, perturbated strain applied to DEM to obtain tangent operator
        :param verbose: type boolean, if or not print messages during calculation
        """
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
            time.sleep(5)
        self.pde.setSymmetryOn()
        # self.pde.getSolverOptions().setTolerance(rtol**2)
        # self.pde.getSolverOptions().setPackage(SolverOptions.UMFPACK)
        self.numGaussPoints = ng
        self.rtol = rtol
        self.usepert = usePert
        self.mode = mode
        self.loadInfor = loadInfor
        self.pert = pert
        self.verbose = verbose
        self.pool = mpi_pool if useMPI else get_pool(mpi=False, threads=np)
        # a = Vector(0., Solution(self.domain)).toListOfTuples()
        # b = escript.Tensor(0, escript.Function(self.domain)).toListOfTuples()
        self.disp = Vector(0., Solution(self.domain))  # length=37 number of the nodes
        self.volume = escript.Tensor(1, escript.Function(self.domain))  # length=32 number of the gaussian points
        self.Strain = escript.Tensor(0, escript.Function(self.domain))
        self.strain_abs = escript.Tensor(0, escript.Function(self.domain))
        self.stress = escript.Tensor(0, escript.Function(self.domain))
        self.S = escript.Tensor4(0, escript.Function(self.domain))
        self.Strain_increment = escript.Tensor(0, escript.Function(self.domain))
        self.stress_increment = escript.Tensor(0, escript.Function(self.domain))
        self.stressLast = escript.Tensor(0, escript.Function(self.domain))
        self.StrainLast = escript.Tensor(0, escript.Function(self.domain))
        self.frobeniusNorm = numpy.zeros(shape=ng)

        # st_check = st.toListOfTuples()

        if 'elastic' in self.mode:
            youngsModulus = 6e7
            poison = 0.2
            self.lam = youngsModulus * poison / (1 + poison) / (1 - 2 * poison)
            self.mu = youngsModulus / 2 / (1 + poison)
            self.rho = 2650
            print()
            print('='*80)
            print('\t\t ELASTIC MODEL USED!')
            print('\tYoungs modulus: %.5e' % youngsModulus)
            print('\tPoisson:        %.5e' % poison)
            print('\trho:            %.5e' % self.rho)
            print()
            print()
            stiffness_list = []
            for i in range(self.numGaussPoints):
                t = [[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]

                # 6 components of the stiffness
                t[0][0][0][0] = self.lam + 2. * self.mu  # xx xx
                t[0][1][0][0] = t[0][0][0][1] = t[1][0][0][0] = t[0][0][1][0] = 0.  # xx xy
                t[1][1][0][0] = t[0][0][1][1] = self.lam  # xx yy
                t[0][1][0][1] = t[0][1][1][0] = t[1][0][0][1] = t[1][0][1][0] = self.mu  # xy xy
                t[1][1][0][1] = t[0][1][1][1] = t[1][1][1][0] = t[1][0][1][1] = 0.  # xy yy
                t[1][1][1][1] = self.lam + 2. * self.mu  # yy yy
                stiffness_list.append(t)
            S = escript.Tensor4(0, escript.Function(self.domain))  # initialization tangent in format of escript data
            for i in range(self.numGaussPoints):
                S.setValueOfDataPoint(i, stiffness_list[i])
            self.S = S
        elif 'vonmises' in self.mode:
            from FEMxEPxML.MisesAssociateFlowIsoHarden import MisesAssociateFlowIsoHarden
            if 'ml' in self.mode:
                self.mathSolver = [
                    MisesAssociateFlowIsoHarden(loadMode='random', mode='net') for _ in range(self.numGaussPoints)]
            elif 'semi' in self.mode:
                self.mathSolver = [
                    MisesAssociateFlowIsoHarden(loadMode='random', mode='semi') for _ in range(self.numGaussPoints)]
            else:
                self.mathSolver = [
                    MisesAssociateFlowIsoHarden(loadMode='random', mode='math') for _ in range(self.numGaussPoints)]
            stress = numpy.array([i.sig for i in self.mathSolver])
            S = numpy.array([[i.D[0, 0], i.D[0, 1], i.D[1, 0], i.D[1, 1], i.D[2, 2]] for i in self.mathSolver])
            self.stress, self.S = self.setStressAndMatrix(stress, S)
        elif 'mcc' in self.mode:
            from FEMxEPxML.MCCmodel import MCCmodel
            self.mathSolver = [MCCmodel(mode='math', verboseFlag=False) for _ in range(self.numGaussPoints)]
            sigma_index = [0, 1, 4]
            stress = numpy.array([i.sig[sigma_index] for i in self.mathSolver])
            S = numpy.array([[i.De[0, 0], i.De[0, 1], i.De[1, 0], i.De[1, 1], i.De[3, 3]] for i in self.mathSolver])
            self.stress, self.S = self.setStressAndMatrix(stress, S)
        elif 'dem' in self.mode or 'ml' in self.mode:
            # get tangent matrix and stress from the DEM RVE
            self.scenes = self.pool.map(initLoad, list(range(ng)))
            st = self.pool.map(getStressAndTangent2D, self.scenes)
            for i in range(ng):# NOTE: first step:
                self.stress.setValueOfDataPoint(i, st[i][0])
                self.S.setValueOfDataPoint(i, st[i][1])
            if 'ml' in self.mode:
                ml_model_path = './FEMxML'
                # -------------------tensorflow------------------------------------
                # from FEMxML.network_strain_abs import RestoreNet
                # saved_model = 'saved_model/epoch_100000'  # 'saved_model/epoch_22500'
                # self.net = RestoreNet(root_path=ml_model_path, saved_model=saved_model)
                # -------------------------Pytorch---------------------------------
                from FEMxML.netTorchDDFrobenius import modelRestore
                # from FEMxML.netTorchLastDouble import modelRestore
                # self.net = modelRestore(
                #     savedPath=os.path.join(ml_model_path, 'ptModel_324_1816_welltrained_10000ok/epoch_100000'),
                #     trainFlag=False)
                self.net = modelRestore(
                    savedPath=os.path.join(ml_model_path, 'ptModelDataDriven24_newGaussian_dddd'),
                    # savedPath=os.path.join(ml_model_path, 'ptModelDataDriven_4_30_30_9_'),
                    trainFlag=False)
                # -------------------------Pytorch 2 net---------------------------------
                # from FEMxML.netTorchStress import modelRestoreStress
                # from FEMxML.netTorchTangent import modelRestoreTanget
                # self.netStress = modelRestoreStress(
                #     savedPath=os.path.join(ml_model_path, 'ptModelStressOnly816onlyStrain'),
                #     trainFlag=False)
                # self.netTangent = modelRestoreTanget(
                #     savedPath=os.path.join(ml_model_path, 'ptModelTangentOnly816onlyStrain'),
                #     trainFlag=False)
                # -------------------------torch rotate--------------------------------
                # from FEMxML.netTorchRotate import modelRestore
                # self.net = modelRestore(savedPath=os.path.join(ml_model_path, 'ptModelOnly24onlyStrainRotate'),
                #                         trainFlag=False)
                # elif 'find' in mode:
                #     from FEMxML.findStressAndTangent import MatchInputAndOutput
                #     self.net = MatchInputAndOutput()  # MatchInputAndOutput
        else:
            raise ValueError('No this mode %s, please check you mode input' % mode)
            # self.net_stiffness = network_stiffness.RestoreNet(root_path=root_path, saved_model=saved_model)
            # self.net_stress = network.RestoreNet(root_path=root_path, saved_model='saved_model(delta strain)')

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
        e = self.pool.map(getVoidRatio2D, self.scenes)
        for i in range(self.numGaussPoints):
            void.setValueOfDataPoint(i, e[i])
        return void

    def getLocalAvgRotation(self):
        rot = escript.Scalar(0, escript.Function(self.domain))
        r = self.pool.map(avgRotation2D, self.scenes)
        for i in range(self.numGaussPoints):
            rot.setValueOfDataPoint(i, r[i])
        return rot

    def getLocalFabric(self):
        fabric = escript.Tensor(0, escript.Function(self.domain))
        f = self.pool.map(getFabric2D, self.scenes)
        for i in range(self.numGaussPoints):
            fabric.setValueOfDataPoint(i, f[i])
        return fabric

    def getCurrentFabric(self, scenes):
        fabric = escript.Tensor(0, escript.Function(self.domain))
        f = self.pool.map(getFabric2D, scenes)
        for i in range(self.numGaussPoints):
            fabric.setValueOfDataPoint(i, f[i])
        return fabric

    def getCurrentVoidRatio(self, scenes):
        void = escript.Scalar(0, escript.Function(self.domain))
        e = self.pool.map(getVoidRatio2D, scenes)
        for i in range(self.numGaussPoints):
            void.setValueOfDataPoint(i, e[i])
        return void

    """ used for clumped particle model only
    def getLocalParOriFab(self):
       fabric=escript.Tensor(0,escript.Function(self.domain))
       f = self.pool.map(getParOriFabric,self.scenes)
       for i in xrange(self.numGaussPoints):
          fabric.setValueOfDataPoint(i,f[i])
       return fabric
    """

    """ used for cohesive particle model only
    def getLocalBondBreakage(self,oriIntr=[]):
       debond = escript.Scalar(0,escript.Function(self.domain))
       num = self.pool.map(getDebondingNumber,zip(self.scenes,repeat(oriIntr)))
       for i in xrange(self.numGaussPoints):
          debond.setValueOfDataPoint(i,num[i])
       return debond
    """

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
        return self.Strain

    def getStrainIncrement(self):
        """
        return strain increment
        type: Tensor on FunctionSpace
        """
        return self.Strain_increment

    def getStressIncrement(self):
        """
        return stress increment
        type: Tensor on FunctionSpace
        """
        return self.stress_increment

    def getVolume(self):
        """
        return stress increment
        type: Tensor on FunctionSpace
        """
        return np.average(np.array(self.volume.toListOfTuples())[:, 0, 0])

    def getStrainAbs(self):
        """
        return absolute accumulation of the strain
        type: Tensor on FunctionSpace
        """
        return self.strain_abs

    def exitSimulation(self):
        """finish the whole simulation, exit"""
        self.pool.close()

    def solve(self, iter_max=100, t=0):
        """
        solve the equation using Newton-Ralphson scheme
        """
        if 'ml' in self.mode:
            u = self.solveDEM(iter_max, t)
        elif 'dem' in self.mode:
            u = self.solveDEM(iter_max, t)
        elif 'elastic' in self.mode:
            u = self.solveElastic(iter_max, t)
        elif 'vonmises' in self.mode:
            u = self.solvePlastic(iter_max, t)
        elif 'mcc' in self.mode:
            u = self.solvePlastic(iter_max, t)
        else:
            raise ValueError('No this mode %s, please check you mode input' % self.mode)
        return u

    def solveML(self, iter_max, t):
        """
        us ml method to solve the problem
        NOTE: increment of the displacement and stress are used in the Newton-Ralphson iteration.
        """

        #
        # renew the strain and stress of the last step
        self.stressLast = self.getCurrentStress()
        self.StrainLast = self.getCurrentStrain()

        iterate = 0
        rtol = self.getRelTolerance()
        x_safe = self.domain.getX()
        self.pde.setValue(A=self.S, X=-self.stress)
        # residual0=util.L2(self.pde.getRightHandSide()) # using force error
        u = self.pde.getSolution()  # trial solution, displacement
        D = util.grad(u)  # trial strain tensor

        current_strain_abs = self.getAbsStrain(D)
        frobeniusNorm = self.getFrobeniusNorm(D)
        update_stress, update_s = self.ml_get_stress_tangent(st=self.Strain + D,
                                                                                strain_abs = self.frobeniusNorm)
        # update_stress, update_s, stress_prediction = self.ml_get_stress_tangent(st=self.Strain + D,
        #                                                                     strain_abs=self.strain_abs+current_strain_abs)
        saveGauss2D(
            name=os.path.join(self.loadInfor, 'iteration_gauss/time_%s_iter_%s.dat' % (str(t), str(iterate))),
            strain_increment=D,
            strain_toatal=self.getCurrentStrain() + D,  # renewed total strain
            stress_increment=update_stress - self.stress,
            stress_toatal=update_stress,
            tangent=update_s,
            strain_abs=self.strain_abs + current_strain_abs,
            frobeniusNorm=self.frobeniusNorm
        )

        err = 1.0  # initial error before iteration
        converged = (err < rtol)
        l, d = 1, 1
        while True:  # only renew the u during the Iteration
            if self.verbose:
                print(
                    "Not converged after %d iteration(s)! d: %e l: %e Relative error=d/l=%e," %
                    (iterate, d, l, d / l))

            iterate += 1
            self.domain.setX(x_safe + u)
            # !! reset u=t=0 means that the displace increment constrained points in this iteration is 0
            self.pde.setValue(A=update_s, X=-update_stress, r=escript.Data())

            du = self.pde.getSolution()  # pde solution for $\left (A_{ijkl}u_{k,l} \right )_{,j}=-X_{ij,j}+Y_{i}$
            try:
                l, d = util.L2(u), util.L2(du)
                err = d / l  # displacement error, alternatively using force error 'residual'
                converged = (err < rtol)
            except:  # because the error is too small
                converged = True
                err = 0.
                pass

            # not renew the coordinate of the node during the Newton-Ralphson Iteration
            self.domain.setX(x_safe)
            u += du
            D = util.grad(u)
            current_strain_abs = self.getAbsStrain(D)
            frobeniusNorm = self.getFrobeniusNorm(D)

            if converged or iterate > iter_max:
                break

            # !!!!!! obtain stress and tangent operator from DEM part or ML
            update_stress, update_s = self.ml_get_stress_tangent(st=self.Strain + D,
                                                                                strain_abs=self.frobeniusNorm)
            # update_stress, update_s, stress_prediction = self.ml_get_stress_tangent(st=self.Strain + D,
            #                                                                     strain_abs=self.strain_abs+current_strain_abs)
            saveGauss2D(
                name=os.path.join(self.loadInfor, 'iteration_gauss/time_%s_iter_%s.dat' % (str(t), str(iterate))),
                strain_increment=D,
                strain_toatal=self.getCurrentStrain() + D,  # renewed total strain
                stress_increment=update_stress - self.stress,
                stress_toatal=update_stress,
                tangent=update_s,
                strain_abs=self.strain_abs + current_strain_abs,
                frobeniusNorm=self.frobeniusNorm
            )
            if t >= 101:
                saveVTK(os.path.join(self.loadInfor, "iteration_vtk/biaxialSmooth_%d.vtu" % t),
                        disp=self.disp + u, dispIncrs=du, strain=self.Strain + D, stress=update_stress)
        """
          update 'domain geometry', 'stress', 'tangent operator',
          'accumulated strain' and 'simulation scenes'.
        """
        self.domain.setX(x_safe + u)
        self.stress = update_stress
        self.S = update_s
        self.Strain += D
        self.disp += u
        self.strain_abs += current_strain_abs
        self.frobeniusNorm += frobeniusNorm
        # a = (self.volume*(1+trace(D))).toListOfTuples()
        self.volume = self.volume * (1 + trace(D))
        self.Strain_increment = D
        # self.scenes = update_scenes
        if self.verbose:
            print("Convergence reached after %d iteration(s)! Relative error: %e" % (iterate, err))
        return u

    def solveElastic(self, iter_max, t):
        """
            Elastic model used!
        """

        #
        # renew the strain and stress of the last step
        self.stressLast = self.getCurrentStress()
        self.StrainLast = self.getCurrentStrain()

        iterate = 0
        rtol = self.getRelTolerance()
        stress = self.getCurrentStress()
        s = self.getCurrentTangent()
        x_safe = self.domain.getX()
        disp_safe = self.disp
        self.pde.setValue(A=s, X=-stress)
        # residual0=util.L2(self.pde.getRightHandSide()) # using force error
        u = self.pde.getSolution()  # trial solution, displacement
        D = util.grad(u)  # trial strain tensor

        # u.toListOfTuples()
        # D.toListOfTuples()
        # current_strain_abs.toListOfTuples()

        # !!!!!! obtain stress and tangent operator from DEM part or ML
        current_strain_abs = self.getAbsStrain(D)
        update_stress = self.elastic_get_stress_tangent(st=self.Strain + D)
        # update_stress, update_s, stress_prediction = self.ml_get_stress_tangent(st=self.Strain + D,
        #                                                                         stLast=self.StrainLast,
        #                                                                         stressLast=self.stressLast)

        err = 1.0  # initial error before iteration
        converged = (err < rtol)
        l, d = 1, 1
        while True:  # only renew the u during the Iteration
            if self.verbose:
                print(
                    "Not converged after %d iteration(s)! d: %e l: %e Relative error=d/l=%e," %
                    (iterate, d, l, d / l))

            iterate += 1
            self.domain.setX(x_safe + u)
            # !! reset u=t=0 means that the displace increment constrained points in this iteration is 0
            self.pde.setValue(X=-update_stress, r=escript.Data())

            du = self.pde.getSolution()  # pde solution for $\left (A_{ijkl}u_{k,l} \right )_{,j}=-X_{ij,j}+Y_{i}$
            try:
                l, d = util.L2(u), util.L2(du)
                err = d / l  # displacement error, alternatively using force error 'residual'
                converged = (err < rtol)
            except:  # because the error is too small
                converged = True
                err = 0.
                pass

            # not renew the coordinate of the node during the Newton-Ralphson Iteration
            self.domain.setX(x_safe)
            u += du
            D = util.grad(u)
            current_strain_abs = self.getAbsStrain(D)

            if converged or iterate > iter_max:
                break

            # !!!!!! obtain stress and tangent operator from DEM part or ML
            update_stress = self.elastic_get_stress_tangent(st=self.Strain + D)
            saveGauss2D(
                name=os.path.join(self.loadInfor, 'iteration_gauss/time_%s_iter_%s.dat' % (str(t), str(iterate))),
                strain_increment=D,
                strain_toatal=self.getCurrentStrain() + D,  # renewed total strain
                stress_increment=update_stress - stress,
                stress_toatal=update_stress,
                tangent=self.S,
                strain_abs=self.strain_abs + current_strain_abs
            )
        """
          update 'domain geometry', 'stress', 'tangent operator',
          'accumulated strain' and 'simulation scenes'.
        """
        self.domain.setX(x_safe + u)
        self.stress = update_stress
        self.Strain += D
        self.disp += u
        self.strain_abs += current_strain_abs
        # a = (self.volume*(1+trace(D))).toListOfTuples()
        self.volume = self.volume * (1 + trace(D))
        self.Strain_increment = D
        # self.scenes = update_scenes
        if self.verbose:
            print("Convergence reached after %d iteration(s)! Relative error: %e" % (iterate, err))
        return u

    def getAbsStrain(self, D):
        current_strain_abs = D.copy()
        D_list = D.toListOfTuples()
        for i, strain_point in enumerate(D_list):
            current_strain_abs.setValueOfDataPoint(i, [[abs(strain_point[0][0]),
                                                        0.5 * abs(strain_point[0][1] + strain_point[1][0])],
                                                       [0.5 * abs(strain_point[0][1] + strain_point[1][0]),
                                                        abs(strain_point[1][1])]])
        return current_strain_abs

    def getFrobeniusNorm(self, D):
        # eye = np.eye(2)
        D = np.array(D.toListOfTuples())
        # frobeniusNormIncrement1 = np.sqrt(np.einsum("ijk, ijl, kl->i", D, D, eye))
        frobeniusNormIncrement = np.sqrt(np.array([np.sum(i*i) for i in D]))
        return frobeniusNormIncrement

    def getStressAndTangent(self, D, t, current_strain_abs, iterate):
        update_scenes = 0
        if self.mode == 'ml':
            update_stress, update_s = self.ml_get_stress_tangent(
                st=self.Strain + D,
                strain_abs = self.frobeniusNorm)
            saveGauss2D(
                name=os.path.join(self.loadInfor, 'iteration_gauss/time_%s_iter_%s.dat' % (str(t), str(iterate))),
                strain_increment=D,
                strain_toatal=self.getCurrentStrain() + D,  # renewed total strain
                stress_increment=update_stress - self.stress,
                stress_toatal=update_stress,
                tangent=update_s,
                strain_abs=self.strain_abs + current_strain_abs,
                frobeniusNorm=self.frobeniusNorm
            )
        elif 'dem' in self.mode:
            update_stress, update_s, update_scenes = self.applyStrain_getStressTangentDEM(st=D)
            saveGauss2D(
                name=os.path.join(self.loadInfor, 'iteration_gauss/time_%s_iter_%s.dat' % (str(t), str(iterate))),
                strain_increment=D,
                strain_toatal=self.getCurrentStrain() + D,  # renewed total strain
                stress_increment=update_stress - self.stress,
                stress_toatal=update_stress,
                tangent=update_s,
                fabric=self.getCurrentFabric(update_scenes),
                vR=self.getCurrentVoidRatio(update_scenes),
                strain_abs=self.strain_abs + current_strain_abs,
                frobeniusNorm=self.frobeniusNorm,
            )
        elif 'vonmises' in self.mode:
            iteration, update_stress, update_s, update_scenes = \
                self.vonmises_get_stress_tangent(st=D)
            saveGauss2D(
                name=os.path.join(self.loadInfor, 'iteration_gauss/time_%s_iter_%s.dat' % (str(t), str(iterate))),
                strain_increment=D,
                strain_toatal=self.getCurrentStrain() + D,  # renewed total strain
                stress_increment=update_stress - self.stress,
                stress_toatal=update_stress,
                tangent=update_s,
                strain_abs=self.strain_abs + current_strain_abs,
                frobeniusNorm=self.frobeniusNorm,
                iteration=iteration,
                epsPlastic=update_scenes[3]
            )
        else:
            raise
        return update_stress, update_s, update_scenes

    def solveDEM(self, iter_max=100, t=0):
        """
        us DEM subroutine to solve the problem
        NOTE: Total quantity of the displacement and stress are used in the iteration.
        """
        iterate = 0
        rtol = self.getRelTolerance()
        stress = self.getCurrentStress()
        s = self.getCurrentTangent()
        x_safe = self.domain.getX()
        self.pde.setValue(A=self.S, X=-self.stress)
        # residual0=util.L2(self.pde.getRightHandSide()) # using force error
        u = self.pde.getSolution()  # trial solution, displacement
        D = util.grad(u)  # trial strain tensor [5.71428571e-08, -2.91167576e-22, -1.00000000e-06]

        # used for debugging
        # eps = numpy.array(D.toListOfTuples()[0])
        # eps0 = 0.5*(eps+eps.T)
        # matrix0 = numpy.array(self.S.toListOfTuples()[0])
        # sig = numpy.einsum('ijkl,kl->ij', matrix0, eps0)
        # if t == 23:
        #     print()

        current_strain_abs = self.getAbsStrain(D)
        frobeniusNorm = self.getFrobeniusNorm(D)

        # !!!!!! obtain stress and tangent operator from DEM part or ML
        update_stress, update_s, update_scenes0 = self.getStressAndTangent(D, t, current_strain_abs, iterate)
        # update_stress0 = update_stress
        err = 1.0  # initial error before iteration
        converged = (err < rtol)
        l, d = 1, 1
        while True:  # only renew the u during the Iteration
            if self.verbose:
                print(
                    "Not converged after %d iteration(s)! d: %e l: %e Relative error=d/l=%e," %
                    (iterate, d, l, d / l))

            iterate += 1
            self.domain.setX(x_safe + u)
            # !! reset u=t=0 means that the displace increment constrained points in this iteration is 0
            self.pde.setValue(A=update_s, X=-update_stress, r=escript.Data())

            du = self.pde.getSolution()  # pde solution for $\left (A_{ijkl}u_{k,l} \right )_{,j}=-X_{ij,j}+Y_{i}$
            try:
                d, l = util.L2(du), util.L2(u)
                err = d / l  # displacement error, alternatively using force error 'residual'
                converged = (err < rtol)
            except:  # because the error is too small
                converged = True
                err = 0.
                pass

            # not renew the coordinate of the node during the Newton-Ralphson Iteration
            self.domain.setX(x_safe)
            u += du
            D = util.grad(u)
            current_strain_abs = self.getAbsStrain(D)
            frobeniusNorm = self.getFrobeniusNorm(D)
            # if 'vonmises' in self.mode:
            #     D = util.grad(du)

            update_stress, update_s, update_scenes = self.getStressAndTangent(D, t, current_strain_abs, iterate)

            if converged or iterate > iter_max:
                break
        """
          update 'domain geometry', 'stress', 'tangent operator',
          'accumulated strain' and 'simulation scenes'.
        """
        self.domain.setX(x_safe + u)
        self.stress = update_stress
        self.S = update_s
        self.Strain += D
        if 'dem' in self.mode:
            self.scenes = update_scenes
        elif 'vonmises' in self.mode:
            '''update_scenes = [sig, st, yieldValue, epsPlastic, deps_plasticVector]'''
            for i in range(self.numGaussPoints):
                self.mathSolver[i].updateState(
                    sig=update_scenes[0][i],
                    deps=update_scenes[1][i],
                    yieldValue=update_scenes[2][i],
                    epsPlastic=update_scenes[3][i],
                    deps_plasticVector=update_scenes[4][i])
        self.Strain_increment = D
        self.strain_abs += current_strain_abs
        self.frobeniusNorm += frobeniusNorm
        self.volume = self.volume * (1 + trace(D))
        if self.verbose:
            print("Convergence reached after %d iteration(s)! Relative error: %e" % (iterate, err))
        return u

    def solvePlastic(self, iter_max=100, t=0):
        """
        us DEM subroutine to solve the problem
        NOTE: Total quantity of the displacement and stress are used in the iteration.
        """
        iterate = 0
        rtol = self.getRelTolerance()
        x_safe = self.domain.getX()
        update_s, update_stress = self.S, self.stress

        self.pde.setValue(A=update_s, X=-update_stress)
        # residual0=util.L2(self.pde.getRightHandSide()) # using force error
        u = self.pde.getSolution()  # trial solution, displacement
        D = util.grad(u)  # trial strain tensor [5.71428571e-08, -2.91167576e-22, -1.00000000e-06]

        # used for debugging
        deps = numpy.array(D.toListOfTuples())
        deps = 0.5*(deps+deps.transpose([0, 2, 1]))
        matrix = numpy.array(self.S.toListOfTuples())
        dsig = numpy.einsum('nijkl,nkl->nij', matrix, deps)


        current_strain_abs = self.getAbsStrain(D)
        frobeniusNorm = self.getFrobeniusNorm(D)

        iteration, update_stress, update_s, update_scenes = \
            self.vonmises_get_stress_tangent(D)

        dsig_mcc = numpy.array((update_stress-self.stress).toListOfTuples())

        sig_new = numpy.array((self.stress).toListOfTuples())+dsig
        residual = dsig_mcc-dsig

        # debug
        # matrix1 = numpy.array(update_s.toListOfTuples()[0])
        # sigma1 = numpy.array(update_stress.toListOfTuples()[0])

        # update_stress0 = update_stress
        err = 1.0  # initial error before iteration
        # converged = (err < rtol)
        l, d = 1, 1
        while True:  # only renew the u during the Iteration
            if self.verbose:
                print(
                    "Not converged after %d iteration(s)! d: %e l: %e Relative error=d/l=%e," %
                    (iterate, d, l, d / l))

            iterate += 1
            self.domain.setX(x_safe)
            self.domain.setX(x_safe + u)
            # !! reset u=t=0 means that the displace increment constrained points in this iteration is 0
            self.pde.setValue(A=update_s, X=-update_stress, r=escript.Data())

            du = self.pde.getSolution()  # pde solution for $\left (A_{ijkl}u_{k,l} \right )_{,j}=-X_{ij,j}+Y_{i}$
            try:
                d, l = util.L2(du), util.L2(u)
                err = d / l  # displacement error, alternatively using force error 'residual'
                converged = (err < rtol)
            except:  # because the error is too small
                converged = True
                err = 0.
                pass

            # not renew the coordinate of the node during the Newton-Ralphson Iteration
            self.domain.setX(x_safe)
            u += du
            D = util.grad(u)
            current_strain_abs = self.getAbsStrain(D)
            frobeniusNorm = self.getFrobeniusNorm(D)
            # if 'vonmises' in self.mode:
            #     D = util.grad(du)

            if converged or iterate > iter_max:
                break

            iteration, update_stress, update_s, update_scenes = \
                self.vonmises_get_stress_tangent(D)

        """
          update 'domain geometry', 'stress', 'tangent operator',
          'accumulated strain' and 'simulation scenes'.
        """
        self.domain.setX(x_safe + u)
        self.stress = update_stress
        self.S = update_s
        self.Strain += D
        if 'dem' in self.mode:
            self.scenes = update_scenes
        elif 'vonmises' in self.mode:
            '''update_scenes = [sig, st, yieldValue, epsPlastic, deps_plasticVector]'''
            for i in range(self.numGaussPoints):
                self.mathSolver[i].updateState(
                    sig=update_scenes[0][i],
                    deps=update_scenes[1][i],
                    yieldValue=update_scenes[2][i],
                    epsPlastic=update_scenes[3][i],
                    deps_plasticVector=update_scenes[4][i])
        elif 'mcc' in self.mode:
            '''sig, deps, yieldValue, dEps_plastic_p, dEps_plastic_q, pc0'''
            for i in range(self.numGaussPoints):
                self.mathSolver[i].updateState(
                    sig=update_scenes[0][i],
                    deps=update_scenes[1][i],
                    yieldValue=update_scenes[2][i],
                    dEps_plastic_p=update_scenes[3][i],
                    dEps_plastic_q=update_scenes[4][i],
                    pc0=update_scenes[5][i],
                )
        self.Strain_increment = D
        self.strain_abs += current_strain_abs
        self.frobeniusNorm += frobeniusNorm
        self.volume = self.volume * (1 + trace(D))
        if self.verbose:
            print("Convergence reached after %d iteration(s)! Relative error: %e" % (iterate, err))
        return u

    def applyStrain_getStressTangentDEM(self, st=escript.Data()):
        """
           apply strain to DEM packing,
           get stress and tangent operator (including two methods)
       """
        '''
        :param st: gradient of u   D = util.grad(u)
        :return: stress, S, scenes
        '''
        st = st.toListOfTuples()
        st = numpy.array(st).reshape(-1, 4)  # shape = (NQ, 4)
        stress = escript.Tensor(0, escript.Function(self.domain))  # initialization stress in format of escript data
        S = escript.Tensor4(0, escript.Function(self.domain))  # initialization tangent in format of escript data
        scenes = self.pool.map(shear2D, list(zip(self.scenes, st)))

        ST = self.pool.map(getStressAndTangent2D, scenes)  # obtain the stress & tangent
        for i in range(self.numGaussPoints):
            stress.setValueOfDataPoint(i, ST[i][0])
            S.setValueOfDataPoint(i, ST[i][1])
        return stress, S, scenes

    def ml_get_stress_tangent(self, st, strain_abs):
        st = st.toListOfTuples()
        # split the rigid rotation from the displacement gradients
        st = numpy.array(st).reshape(-1, 4)  # shape = (NQ, 4)
        st = np.concatenate((st[:, 0:1], 0.5*(st[:, 1:2]+st[:, 2:3]), st[:, 3:4]), axis=1)
        try:
            strain_abs = numpy.array(strain_abs.toListOfTuples()).reshape(-1, 4)  # shape = (NQ, 4)
            strain_abs = np.concatenate((strain_abs[:, 0:1], .5 * (strain_abs[:, 1:2] + strain_abs[:, 2:3]),
                                    strain_abs[:, 3:4]), axis=1)
        except:
            pass

        # -------------------------torch 2 net---------------------------------
        # strainAndStrainAbs = st
        # stress_fabric = self.netStress.get_stressAndStiffness(inputs=strainAndStrainAbs)  # torch & strain_abs
        # stiffness = self.netTangent.get_stressAndStiffness(inputs=strainAndStrainAbs)  # torch & strain_abs
        # -------------------------torch rotate--------------------------------
        # strainAndStrainAbs = st
        # strainRotate, eigenVectorList = rotateStrain(strainAndStrainAbs)
        # stress_fabric, stiffness = self.net.get_stressAndStiffness(inputs=strainRotate)  # torch & strain_abs
        # stressRotate = rotateStress(stress_fabric, eigenVectorList)
        # stress_fabric = np.concatenate((stressRotate[:, 0:2], stressRotate[:, 3:4]), axis=1)
        # ----------------------------torch------------------------------------
        # strainAndStrainAbs = np.concatenate((stLast, stressLast, st), axis=1)
        strainAndStrainAbs = np.concatenate((st, strain_abs.reshape(-1, 1)), axis=1)
        # strainAndStrainAbs = st
        stress_fabric, stiffness = self.net.get_stressAndStiffness(inputs=strainAndStrainAbs)  # torch & strain_abs

        stress, S = self.setStressAndMatrix(stress_fabric, stiffness)
        return stress, S

    def elastic_get_stress_tangent(self, st):
        g = st
        kronecker_ = kronecker(self.pde.getDim())
        stress = self.lam * trace(g) * kronecker_ + self.mu * (g + transpose(g)) + \
                 Tensor(kronecker(self.domain.getDim()) * (-1.e5), Function(self.domain))
        return stress

    def vonmises_get_stress_tangent(self, st):
        st = numpy.array(st.toListOfTuples())
        st = 0.5*(st+st.transpose(0, 2, 1)).reshape(self.numGaussPoints, 4)  # tensor notion
        # make sure in a order of [00 11 01]
        st = numpy.concatenate((st[:, 0:1], st[:, 3:4], 2.*st[:, 2:3]), axis=1)  # Voigt notion
        # temp = self.pool.map(vonmisesSolver, list(zip(self.mathSolver, st)))
        temp = []
        if 'mcc' in self.mode:
            st_6 = np.zeros(shape=[len(st), 6])
            st_6[..., :2] = st[..., :2]
            st_6[..., 3] = st[..., 2]
            st = st_6
        for i in range(self.numGaussPoints):
            temp.append(vonmisesSolver([self.mathSolver[i], st[i]]))
        if 'mcc' in self.mode:
            iteration, sig, materialMatrix, dEps_plastic_p, dEps_plastic_q, yieldValue, pc0 = \
                numpy.array([temp[i][0] for i in range(self.numGaussPoints)]), \
                numpy.array([temp[i][1] for i in range(self.numGaussPoints)]), \
                numpy.array([temp[i][2] for i in range(self.numGaussPoints)]), \
                numpy.array([temp[i][3] for i in range(self.numGaussPoints)]), \
                numpy.array([temp[i][4] for i in range(self.numGaussPoints)]), \
                numpy.array([temp[i][5] for i in range(self.numGaussPoints)]), \
                numpy.array([temp[i][6] for i in range(self.numGaussPoints)])
            sig_3 = np.concatenate((sig[:, :2], sig[:, 3:4]), axis=1)  # voigt notion
            materialMatrix = numpy.array([[materialMatrix[i][0, 0], materialMatrix[i][0, 1],
                                           materialMatrix[i][1, 0], materialMatrix[i][1, 1],
                                           materialMatrix[i][3, 3]] for i in range(self.numGaussPoints)])
            update_scenes = [sig, st, yieldValue, dEps_plastic_p, dEps_plastic_q, pc0]
            stress, S = self.setStressAndMatrix(sig_3, materialMatrix)
            return iteration, stress, S, update_scenes
        else:  # vonmises
            iteration, sig, materialMatrix, epsPlastic, deps_plasticVector, yieldValue = \
             numpy.array([temp[i][0] for i in range(self.numGaussPoints)]),\
            numpy.array([temp[i][1] for i in range(self.numGaussPoints)]),\
            numpy.array([temp[i][2] for i in range(self.numGaussPoints)]),\
            numpy.array([temp[i][3] for i in range(self.numGaussPoints)]),\
            numpy.array([temp[i][4] for i in range(self.numGaussPoints)]), \
            numpy.array([temp[i][5] for i in range(self.numGaussPoints)])
            materialMatrix = numpy.array([[materialMatrix[i][0, 0], materialMatrix[i][0, 1],
                                           materialMatrix[i][1, 0], materialMatrix[i][1, 1],
                                           materialMatrix[i][2, 2]] for i in range(self.numGaussPoints)])
            update_scenes = [sig, st, yieldValue, epsPlastic, deps_plasticVector]
            stress, S = self.setStressAndMatrix(sig, materialMatrix)
            return iteration, stress, S, update_scenes

    def setStressAndMatrix(self, stress_fabric, stiffness):
        stress = escript.Tensor(0, escript.Function(self.domain))  # initialization stress in format of escript data
        S = escript.Tensor4(0, escript.Function(self.domain))  # initialization tangent in format of escript data

        stiffness_list = []
        for i in range(self.numGaussPoints):
            t = [[[[0, 0], [0, 0]], [[0, 0], [0, 0]]], [[[0, 0], [0, 0]], [[0, 0], [0, 0]]]]

            # 6 components of the stiffness
            if len(stiffness[0]) == 6:
                t[0][0][0][0] = stiffness[i, 0]
                t[0][1][0][0] = t[0][0][0][1] = t[1][0][0][0] = t[0][0][1][0] = stiffness[i, 1]
                t[1][1][0][0] = t[0][0][1][1] = stiffness[i, 2]
                t[0][1][0][1] = t[0][1][1][0] = t[1][0][0][1] = t[1][0][1][0] = stiffness[i, 3]
                t[1][1][0][1] = t[0][1][1][1] = t[1][1][1][0] = t[1][0][1][1] = stiffness[i, 4]
                t[1][1][1][1] = stiffness[i, 5]
            elif len(stiffness[0]) == 9:
                '''
                  0    1    2    3    4   5    6     7   8   
                0000 0001 0011 0100 0101 0111 1100 1101 1111
                '''
                t[0][0][0][0] = stiffness[i, 0]
                t[1][1][0][0] = stiffness[i, 6]
                t[0][0][1][1] = stiffness[i, 2]
                t[0][1][0][0] = t[1][0][0][0] = stiffness[i, 3]
                t[0][0][0][1] = t[0][0][1][0] = stiffness[i, 1]
                t[1][1][1][1] = stiffness[i, 8]
                t[1][1][0][1] = t[1][1][1][0] = stiffness[i, 7]
                t[0][1][1][1] = t[1][0][1][1] = stiffness[i, 5]
                t[0][1][0][1] = t[0][1][1][0] = t[1][0][0][1] = t[1][0][1][0] = stiffness[i, 4]
            elif len(stiffness[0]) == 5:
                t[0][0][0][0] = stiffness[i, 0]  # A
                t[0][0][1][1] = t[1][1][0][0] = stiffness[i, 1]  # B
                t[1][1][1][1] = stiffness[i, 3]  # C
                t[0][1][0][1] = t[1][0][0][1] = stiffness[i, 4]  # D
                t[0][1][1][0] = t[1][0][1][0] = stiffness[i, 4] # E
            else:
                raise ValueError('stiffness length should be 6 or 9, but get %d' % (len(stiffness[0])))
            stiffness_list.append(t)
        # transform the stress and the stiffness into the form of escript.data
        for i in range(self.numGaussPoints):
            S.setValueOfDataPoint(i, stiffness_list[i])
            temp = [[stress_fabric[i, 0], stress_fabric[i, 2]],
                    [stress_fabric[i, 2], stress_fabric[i, 1]]]
            stress.setValueOfDataPoint(i, temp)
        return stress, S


def vonmisesSolver(param):
    '''iteration, sig, materialMatrix, eps, epsPlastic, epsPlasticVector'''
    misesClass, st = param[0], param[1]
    return misesClass.solver(st)


def sig3toSig6(sig):
    ''' Voigt notion '''
    sig_6 = np.zeros(6)
    sig_6[:2] = sig[:2]
    sig_6[3] = sig[2]
    return sig_6