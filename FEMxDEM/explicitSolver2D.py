from __future__ import print_function
from builtins import input
from builtins import zip
from builtins import range
from builtins import object

__author__ = "Ning Guo, ceguo@connect.ust.hk"
__supervisor__ = "Jidong Zhao, jzhao@ust.hk"
__institution__ = "The Hong Kong University of Science and Technology"

""" 2D model for multiscale simulation
which implements a Newton-Raphson scheme
into FEM framework to solve the nonlinear
problem where the tangent operator is obtained
from DEM simulation by calling simDEM modules"""

# import Escript modules
# import tensorflow.compat.v1 as tf
# import esys.escript as escript
from esys.escript import Vector, Solution, grad, trace, transpose, kronecker, \
    length, interpolate, FunctionOnBoundary, matrix_mult, \
    whereZero, whereNegative, integrate, sup, inf, sqrt, inner, L2, \
    Tensor, Tensor4, Function, Data, Scalar
from esys.escript.pdetools import Locator, Projector
from esys.escript.linearPDEs import LinearPDE, SolverOptions
from utilSelf.saveGauss import saveGauss2D
from simDEM import initLoad, shear2D, getFabric2D, getStressAndTangent2D, getVoidRatio2D, getEquivalentPorosity, \
    avgRotation2D
# other python modules
from itertools import repeat
import numpy as np
from esys.weipa import saveVTK
import os, sys, time
from FEMxDEM.Boudary import BoudaryExplicit


# from train_model import restore # , get_stress_tengent !!!! check why can not import get_stress_tengent
# from network_stiffness import RestoreNet
# from network_stiffness_complex import RestoreNet

def getPool(threads=1, mpi=False):
    if mpi:  # using MPI
        from mpipool import MPIPool
        pool = MPIPool()
        if not pool.is_master():
            sys.exit(0)
    elif threads > 1:  # using multiprocessing
        from multiprocessing import Pool
        pool = Pool(processes=threads)
    else:
        raise RuntimeError("Wrong arguments: either mpi=True or threads>1.")
    return pool


class emplicitSolver():
    def __init__(self, domain,
                 timeStep, maxTime, mode, numG, numP, rho, engergyTol, loadInfor, cons,
                 attenuationCoefficient=0.8, dampCoefficient=0., pool=None):

        self.numG, self.numP = numG, numP
        self.loadInfor = loadInfor
        self.domain = domain
        self.timeStep = timeStep
        self.maxTime = maxTime
        self.mode = mode  # "elastic"  # this mode represent how to calculate the stress on the Gaussian points
        self.rho = rho
        self.pde = LinearPDE(self.domain,
                               numEquations=self.domain.getDim(),
                               numSolutions=self.domain.getDim())
        self.pde.getSolverOptions().setSolverMethod(SolverOptions.HRZ_LUMPING)  # accelerating the solving method
        self.saved_path = loadInfor
        self.kronecker_ = kronecker(self.pde.getDim())
        self.startTime = time.time()

        # for first two time steps
        self.du = Vector(0., Solution(self.domain))
        self.u = Vector(0., Solution(self.domain))
        self.u_last = Vector(0., Solution(self.domain))
        self.v = Vector(0., Solution(self.domain))
        self.vHalf = Vector(0., Solution(self.domain))
        self.a = Vector(0., Solution(self.domain))
        self.strain = Tensor(0, Function(self.domain))
        self.stress = self.setStressTensor(sig=np.array([np.eye(2)*(-1e5) for i in range(self.numG)]))
        self.stressDamp = Tensor(0, Function(self.domain))
        self.volume = Tensor(1, Function(self.domain))
        self.aMax = 0.

        # set the topSurf & bottomSurf
        self.x = self.domain.getX()
        self.bx = FunctionOnBoundary(self.domain).getX()
        self.lx, self.ly = sup(self.x[0]) - inf(self.x[0]), sup(self.x[1]) - inf(self.x[1])
        self.topSurf = whereZero(self.bx[1] - sup(self.bx[1]))
        self.bottomSurf = whereZero(self.bx[1] - inf(self.bx[1]))
        self.topDomain = whereZero(self.x[1] - sup(self.x[1]))

        # initialize the DEM scenesS
        self.pool = pool
        if 'dem' in self.mode:
            self.scenes = self.pool.map(initLoad, list(range(self.numG)))
            # NOTE: first step: get tangent matrix and stress from the DEM RVE
            st = self.pool.map(getStressAndTangent2D, self.scenes)
            # st_check = st.toListOfTuples()
            for i in range(self.numG):
                self.stress.setValueOfDataPoint(i, st[i][0])
        elif 'ml' in self.mode:
            from FEMxML.netTorchStress import NetStress, modelRestore
            self.net = modelRestore(savedPath='./FEMxML/ptModel_e2s_816', trainFlag=False)
        self.echo()

    def echo(self):
        print()
        print('-'*80)
        print('\t Central differentiation engaged!')
        print('\t Damping coefficient:     %.3e' % self.dampCoefficient)
        print('\t Attenuation coefficient: %.3e' % self.attenuationCoefficient)
        print('-'*80)
        print()

    def initialize(self, Y=Data(), y=Data(),
                   q=Data(), r=Data()):
        """
        initialize the model for each time step, e.g. assign parameters
        :param Y: type vector, body force on FunctionSpace, e.g. gravity
        :param y: type vector, boundary traction on FunctionSpace (FunctionOnBoundary)
        :param q: type vector, mask of location for Dirichlet boundary
        :param r: type vector, specified displacement for Dirichlet boundary
        """
        # self.Nbc = y
        self.pde.setValue(D=self.kronecker_ * self.rho, Y=Y, y=y, q=q, r=r)

    def topForceCal(self, stress, n):
        proj = Projector(self.domain)
        sig = proj(stress)  # project Gauss point value to nodal value
        sig_bounda = interpolate(sig, FunctionOnBoundary(self.domain))  # interpolate
        traction = matrix_mult(sig_bounda, self.domain.getNormal())  # boundary traction
        # temp1_check = sup(bx[1]).toListOfTuples()
        tractTop = traction * self.topSurf  # traction at top surface
        forceTop = integrate(tractTop, where=FunctionOnBoundary(self.domain))  # resultant force at top
        lengthTop = integrate(self.topSurf, where=FunctionOnBoundary(self.domain))  # length of top surface
        # print('Force:\tx%e\ty%e\tlength:\t%e' % (forceTop[0], forceTop[1], lengthTop))

        fout = open(os.path.join(self.saved_path, 'biaxial_surf.dat'), 'a')
        fout.write(str(n) + ' ' + str(self.axialStrain()[1]) +
                   ' ' + str(forceTop[1]) + ' ' + str(lengthTop) + '\n')
        fout.close()

        return

    def setStressTensor(self, sig):
        stress = Tensor(0, Function(self.domain))
        for i in range(self.numG):
            stress.setValueOfDataPoint(i, sig[i])
        return stress

    def getDomain(self, ):
        return self.domain

    def getAcceleration(self, ):
        return self.a

    def getCurrentStress(self, ):
        return self.stress

    def getCurrentStrain(self, ):
        return grad(self.u)

    def getCurrentPacking(self, pos, time, prefix):
        return 0

    def getVolume(self, ):
        return np.average(np.array(self.volume.toListOfTuples())[:, 0, 0])

    def stressCalElastic(self, D):
        g = D
        dsig = self.lam * trace(g) * self.kronecker_ + self.mu * (g + transpose(g))
        return dsig

    def stressCalDem(self, strainIncrement):
        """
            call DEM as the constitutive model at the gaussian point
        """
        stress, _, self.scenes = self.applyStrain2RVE(st=strainIncrement)
        return stress

    def stressCalML(self, ):
        """
            call netStress to calculate the stress
        """
        g = self.strain.toListOfTuples()
        gArray = np.array(g).reshape(-1, 4)
        stress_fabric = self.net.get_stressAndStiffness(inputs=gArray)

        #
        stress = Tensor(0, Function(self.domain))  # initialization stress in format of escript data
        for i in range(self.numG):
            temp = [[stress_fabric[i, 0], stress_fabric[i, 1]],
                    [stress_fabric[i, 1], stress_fabric[i, 2]]]
            stress.setValueOfDataPoint(i, temp)
        return stress

    def getDampStress(self, ):
        """
            Calculating the bulk damp stress according to the bulk damp viscosity
            Reference: Abaqus Theory Guide
                        stressDamp = b1*rho*c_d*L*volumeStrainRate
        """
        epsilonRate = grad(self.vHalf)
        epsilonRate = .5*(epsilonRate+transpose(epsilonRate))
        volumeStrainRate = trace(epsilonRate)
        stressDampValue = -self.dampCoefficient * self.rho * \
                     self.waveSpeed * self.charactericLength * \
                     volumeStrainRate
        stressDampValue = stressDampValue.toListOfTuples()
        for i in range(self.numG):
            self.stressDamp.setValueOfDataPoint(i, self.kronecker_*stressDampValue[i])
            # stress.setValueOfDataPoint(i, ST[i][0])
        # stress = self.lam * trace(g) * self.kronecker_ + self.mu * (g + transpose(g)) + \
        #          Tensor(kronecker(self.domain.getDim()) * (-1.e5), Function(self.domain))
        return

    def axialStrain(self, ):
        '''
        return the strain and deltaY on y direction
        '''
        topU = self.topSurf * self.u
        deltaY = sup(length(topU[1]))
        axialStrainCurrent = deltaY / self.ly
        # emplicitSolver.check(topu=topU, topsurf=self.topSurf, disp=self.u)
        return axialStrainCurrent, deltaY

    def applyStrain2RVE(self, st=Data()):
        '''
        :param st: gradient of u   D = util.grad(u)
        :return: stress, S, scenes
        '''
        st = st.toListOfTuples()
        st = np.array(st).reshape(-1, 4)  # shape = (NQ, 4)
        stress = Tensor(0, Function(self.domain))  # initialization stress in format of escript data
        S = Tensor4(0, Function(self.domain))  # initialization tangent in format of escript data
        scenes = self.pool.map(shear2D, list(zip(self.scenes, st)))

        ST = self.pool.map(getStressAndTangent2D, scenes)  # obtain the stress & tangent
        for i in range(self.numG):
            stress.setValueOfDataPoint(i, ST[i][0])
            S.setValueOfDataPoint(i, ST[i][1])

        return stress, S, scenes

    def getLocalVoidRatio(self):
        void = Scalar(0, Function(self.domain))
        e = self.pool.map(getVoidRatio2D, self.scenes)
        for i in range(self.numG):
            void.setValueOfDataPoint(i, e[i])
        return void

    def getLocalAvgRotation(self):
        rot = Scalar(0, Function(self.domain))
        r = self.pool.map(avgRotation2D, self.scenes)
        for i in range(self.numG):
            rot.setValueOfDataPoint(i, r[i])
        return rot

    def getLocalFabric(self):
        fabric = Tensor(0, Function(self.domain))
        f = self.pool.map(getFabric2D, self.scenes)
        for i in range(self.numG):
            fabric.setValueOfDataPoint(i, f[i])
        return fabric

    @staticmethod
    def check(**kwargs):
        for key in kwargs:
            kwargs[key] = kwargs[key].toListOfTuples()
        return

    def solve(self, n, t, duBoundary, loadStepMax):
        """
        n: the solving step
        t: the clock time
        duBoundary: du boundary at current step
        """
        # ... set initial values ....
        self.x = self.domain.getX()
        iterNum = 0
        D_l = grad(duBoundary)
        self.domain.setX(self.x+duBoundary)
        self.u = self.u + duBoundary
        self.strain = self.strain + D_l
        self.volume = self.volume * (1. + trace(D_l))

        dsig_load = self.stressCalElastic(D_l)
        self.pde.setValue(X=-(self.stress+dsig_load))
        a = self.pde.getSolution()

        # central difference rule
        if n == 0:
            v_half = self.vHalf +a*self.timeStep/2.
        else:
            v_half = self.vHalf + a*self.timeStep
        du = v_half*self.timeStep  # displacement due to the unbalanced force

        D = grad(du)
        self.stress = self.stress+dsig_load + self.stressCalElastic(D)

        self.domain.setX(self.domain.getX() + du)
        self.u = self.u + du
        self.strain = self.strain +D
        self.volume = self.volume * (1. + trace(D))

        # renew the global variables
        t = t + self.timeStep
        self.vHalf = v_half
        n += 1
        return self.u, n, t, iterNum

    def stressCal(self, deps):
        dsig = self.cons(deps)
        return dsig

    def exitSimulation(self):
        """finish the whole simulation, exit"""
        if self.pool is not None:
            self.pool.close()


