from esys.escript import util
from esys.escript import *
import esys.escript as escript
from esys.finley import Rectangle
from esys.weipa import saveVTK
from esys.escript.pdetools import Projector
from esys.escript.linearPDEs import LinearPDE, SolverOptions
import time
import os
import errno
from matplotlib import pyplot as plt
import numpy as np

# from saveGauss import saveGauss2D
# from msFEM2D import MultiScale

"""
file used for generating isotropic elastic loading data

"""


class ElasticSolever:
    def __init__(self, domain, tol, ng=1, E=8e8, v=0.2, verbose=True):
        self.__domain = domain
        self.__pde = LinearPDE(domain, numEquations=self.__domain.getDim(), numSolutions=self.__domain.getDim())
        self.__pde.getSolverOptions().setSolverMethod(SolverOptions.DIRECT)
        self.__pde.setSymmetryOn()
        self.__rtol = tol
        self.__E = E
        self.__v = v
        self.__numberGuassion = ng
        self.__S = self.getCurrentTangent()
        self.__strain = escript.Tensor(0, escript.Function(self.__domain))
        self.__stress = escript.Tensor(0, escript.Function(self.__domain))
        self.__verbose = True

    def initialize(self, b=escript.Data(), f=escript.Data(), specified_u_mask=escript.Data(),
                   specified_u_val=escript.Data()):
        """
        initialize the model for each time step, e.g. assign parameters
        :param b: type vector, body force on FunctionSpace, e.g. gravity
        :param f: type vector, boundary traction on FunctionSpace (FunctionOnBoundary)
        :param specified_u_mask: type vector, mask of location for Dirichlet boundary
        :param specified_u_val: type vector, specified displacement for Dirichlet boundary
        """
        self.__pde.setValue(Y=b, y=f, q=specified_u_mask, r=specified_u_val)

    def getDomain(self):
        """
        return model domain
        """
        return self.__domain

    def getRelTolerance(self):
        """
        return relative tolerance for convergence
        type float
        """
        return self.__rtol

    def getCurrentTangent(self):
        """
        return current tangent operator
        type Tensor4 on FunctionSpace
        """
        E = self.__E
        v = self.__v
        stiffness = escript.Tensor4(0, escript.Function(self.__domain))
        stif_matrix = np.zeros([2, 2, 2, 2])
        stif_matrix[0, 0, 0, 0] = stif_matrix[1, 1, 1, 1] = 1-v
        stif_matrix[0, 0, 1, 1] = stif_matrix[1, 1, 0, 0] = v
        stif_matrix[0, 1, 0, 1] = stif_matrix[1, 0, 1, 0] = (1-2*v)/2.
        stif_matrix *= (E/((1+v)*(1-2*v)))
        for i in range(self.__numberGuassion):
            stiffness.setValueOfDataPoint(i, stif_matrix)
        return stiffness

    def getStressIncrement(self, strainIncrement=escript.Data()):
        """
        calculate the stress increment
        :return: stressIncrement -> escript.Tensor2
        """
        stif_matrix = np.array(self.__S.toListOfTuples())
        stressIncrement = escript.Tensor(0, escript.Function(self.__domain))
        strainIncrement = np.array(strainIncrement.toListOfTuples())
        stressIncrement_array = np.einsum('pijkl, pkl->pij', stif_matrix, strainIncrement)
        for i in range(self.__numberGuassion):
            stressIncrement.setValueOfDataPoint(i, stressIncrement_array[i])
        return stressIncrement

    def getCurrentStress(self):
        """
        return current stress
        type: Tensor on FunctionSpace
        """
        return self.__stress

    def getCurrentStrain(self):
        """
        return current strain
        type: Tensor on FunctionSpace
        """
        return self.__strain

    def solve(self, iter_max=100):
        """
        solve the equation based on Newton
        :param iter_max:
        :return:
        """
        iterate = 0
        rtol = self.getRelTolerance()
        stress = self.getCurrentStress()
        s = self.__S
        x_safe = self.__domain.getX()
        self.__pde.setValue(A=s, X=-stress)  # set the pde value
        u = self.__pde.getSolution()  # solve for the solution
        print(u.toListOfTuples())
        new_strain = util.grad(u)  # calculate for the gradients of the displacement u as the new strains
        stressIncrement = self.getStressIncrement(strainIncrement=new_strain)
        stress += stressIncrement
        converged = (1.0 < rtol)
        while not converged and iterate < iter_max:
            self.__domain.setX(x_safe+u)  # renew node coordinate
            self.__pde.setValue(A=s, X=-stress)
            du = self.__pde.getSolution()
            strain_increment = util.grad(du)
            stressIncrement = self.getStressIncrement(strainIncrement=strain_increment)
            stress += stressIncrement
            u += du
            l, d = util.L2(u), util.L2(du)
            err = d / l  # displacement error, alternatively using force error 'residual'
            converged = (err < rtol)
        self.__domain.setX(x_safe + u)
        self.__stress = stress
        self.__strain += util.grad(u)
        if self.__verbose:
            print("Convergence reached after %d iteration(s)! Relative error: %e" % (iterate, err))
        return u

try:
    os.mkdir('./result/')
    os.mkdir('./result/gauss')
    os.mkdir('./result/vtk')
    os.mkdir('./result/packing')
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

# initialization
vel = -0.00015  # loading velocity
confining = -1.e5  # confining pressure
lx = 1.
ly = 1.  # sample size, 50mm by 100mm
nx = 2
ny = 2  # sample discretization, 8 by 16 quadrilateral elements
# mydomain 相当于一个有限元模型或者拉格朗日插值空间
mydomain = Rectangle(l0=lx, l1=ly, n0=nx, n1=ny, order=2, integrationOrder=2)
dim = mydomain.getDim()
k = kronecker(mydomain)
numg = 4*nx*ny
err_rol = 0.01
solver = ElasticSolever(domain=mydomain, tol=err_rol, ng=numg, E=8e8, v=0.2)

# initialize the displace solution domain
disp = Vector(0., Solution(mydomain))
disp_check = disp.toListOfTuples()
t = 0

stress = solver.getCurrentStress()  # initial stress
stress_check = stress.toListOfTuples()
proj = Projector(mydomain)
sig = proj(stress)  # project Gauss point value to nodal value
sig_check = sig.toListOfTuples()
sig_bounda = interpolate(sig, FunctionOnBoundary(mydomain))  # interpolate
sig_bounda_check = sig_bounda.toListOfTuples()
traction = matrix_mult(sig_bounda, mydomain.getNormal())  # boundary traction
traction_check = traction.toListOfTuples()
x = mydomain.getX()  # nodal coordinate
x_check = x.toListOfTuples()
bx = FunctionOnBoundary(mydomain).getX()
bx_check = bx.toListOfTuples()
temp_check = bx[1].toListOfTuples()
aaa = bx[1] - sup(bx[1])
temp1_check = aaa.toListOfTuples()
# temp1_check = sup(bx[1]).toListOfTuples()
topSurf = whereZero(bx[1] - sup(bx[1]))  # equals 1 if the node is on the top
topSurf_check = topSurf.toListOfTuples()
tractTop = traction * topSurf  # traction at top surface
tractTop_check = tractTop.toListOfTuples()
forceTop = integrate(tractTop, where=FunctionOnBoundary(mydomain))  # resultant force at top
lengthTop = integrate(topSurf, where=FunctionOnBoundary(mydomain))  # length of top surface
# forceTop_check = forceTop.toListOfTuples()
# lengthTop_check = lengthTop.toListOfTuples()
fout = open('./result/biaxial_surf.dat', 'w')
fout.write('0 ' + str(forceTop[1]) + ' ' + str(lengthTop) + '\n')
# Dirichlet BC positions, smooth at bottom and top, fixed at the center of bottom
# bind the mind point in order not to slide in x direction
Dbc = whereZero(x[1]) * [0, 1] + whereZero(x[1] - ly) * [0, 1] + whereZero(x[1]) * whereZero(x[0] - .5 * lx) * [1, 1]
# Dirichlet BC values
Vbc = whereZero(x[1]) * [0, 0] + whereZero(x[1] - ly) * [0, vel] + whereZero(x[1]) * whereZero(x[0] - .5 * lx) * [0, 0] # bind the mind point in order not to slide in x direction
# Neumann BC, constant confining pressure
Nbc = whereZero(bx[0]) * [-confining, 0] + whereZero(bx[0] - lx) * [confining, 0]  # Neuman BC on the interplotion points

Dbc_check = Dbc.toListOfTuples()
Vbc_check = Vbc.toListOfTuples()
Nbc_check = Nbc.toListOfTuples()


# plot the node & boundary-node
fig = plt.figure()
axes = fig.gca()
xx = [x_check[i][0] for i in range(len(x_check))]
yy = [x_check[i][1] for i in range(len(x_check))]
bxx = [bx_check[i][0] for i in range(len(bx_check))]
byy = [bx_check[i][1] for i in range(len(bx_check))]
plt.scatter(xx, yy, label='node')
plt.scatter(bxx, byy, label='boundary')
for i in range(len(xx)):
    plt.text(xx[i], yy[i], str(i+1))
for i in range(len(bxx)):
    plt.text(bxx[i], byy[i], str(i+1))
plt.legend()
plt.axis('equal')
# plt.savefig('./node.svg')
# plt.show()

time_start = time.time()
while t < 100:  # apply 100 load steps
    print('\n\n'+'-'*80+'\n'+'Loading step # %d' % t)

    solver.initialize(f=Nbc, specified_u_mask=Dbc, specified_u_val=Vbc)  # initialize BC
    Dbc_check = Dbc.toListOfTuples()
    Vbc_check = Vbc.toListOfTuples()
    Nbc_check = Nbc.toListOfTuples()
    t += 1
    du = solver.solve(iter_max=100)  # get solution: nod\n\nal displacement

    disp += du
    du_ckeck = du.toListOfTuples()
    disp_check = disp.toListOfTuples()

    stress = solver.getCurrentStress()

    dom = solver.getDomain()  # domain is updated Lagrangian formulation
    proj = Projector(dom)
    sig = proj(stress)

    sig_bounda = interpolate(sig, FunctionOnBoundary(dom))
    traction = matrix_mult(sig_bounda, dom.getNormal())
    tractTop = traction * topSurf
    forceTop = integrate(tractTop, where=FunctionOnBoundary(dom))
    lengthTop = integrate(topSurf, where=FunctionOnBoundary(dom))
    fout.write(str(t * vel / ly) + ' ' + str(forceTop[1]) + ' ' + str(lengthTop) + '\n')

    # vR = solver.getLocalVoidRatio()
    # fabric = solver.getLocalFabric()
    # strain_current = solver.getCurrentStrain()
    strain_current = util.grad(du)
    strain = solver.getCurrentStrain()
    # saveGauss2D(name='./result/gauss/time_' + str(t) + '.dat', strain_current=strain_current, strain_toatal=strain, stress=stress, fabric=fabric, vR=vR)
    volume_strain = trace(strain)
    dev_strain = symmetric(strain) - volume_strain * k / dim
    shear = sqrt(2 * inner(dev_strain, dev_strain))
    # vR_check = vR.toListOfTuples()
    # fabric_check = fabric.toListOfTuples()
    strain_check = strain.toListOfTuples()
    volume_strain_check = volume_strain.toListOfTuples()
    dev_strain_check = dev_strain.toListOfTuples()
    shear_check = shear.toListOfTuples()
    saveVTK("./result/vtk/biaxialSmooth_%d.vtu" % t, disp=disp, shear=shear, strain=strain, stress=stress)

# solver.getCurrentPacking(pos=(), time=t, prefix='./result/packing/')
time_elapse = time.time() - time_start
fout.write("#Elapsed time in hours: " + str(time_elapse / 3600.) + '\n')
fout.close()
# solver.exitSimulation()