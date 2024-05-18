from __future__ import print_function
from yadeimport import *
from utilSelf.general import echo
import numpy as np


#
# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .8
# fric = 0.5
# rMean=.01
# num=600
# seed=1
# for p in O.bodies:
#     p.state.blockedDOFs = 'ZXY'
#     p.state.mass = 2650 * 4 / 3 * pi * p.shape.radius * p.shape.radius ** 2  # 0.1 = thickness of cylindrical particle
#     inertia = 0.4 * p.state.mass * p.shape.radius ** 2
#     p.state.inertia = (0.5 * inertia, 0.5 * inertia, 0.5 * inertia)    # for 3D_600_mass: 5*5*10 modeling


# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .8
# fric = 0.15
# rMean=.01
# num=600
# seed=1                  #for 3D_600_mass2



# confining = 50e3
# size = .24
# young = 6.e8
# poisson = .8
# fric = 0.15
# rMean=.01
# num=600
# seed=1                 #for 3D_600_mass2_5e4



# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .8
# fric = 0.5
# rMean=.01
# num=600
# seed=1                 #for 3D_600_mass3


#
# confining = 100e3
# size = .24
# young = 3.e8
# poisson = .8
# fric = 0.5
# rMean=.01
# num=600
# seed=1                 #for 3D_600_mass4



# confining = 100e3
# size = .24
# young = 3.e8
# poisson = .8
# fric = 0.2
# rMean=.01
# num=600
# seed=1                 #for 3D_600_mass5


# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .8
# fric = 0.15
# rMean=.007
# num=1000
# seed=1                 #for 3D_1000_mass5



# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .2
# fric = 0.5
# rMean=.007
# num=1000
# seed=1                 #for 3D_1000_mass7


# confining = 100e3
# size = .24
# young = 2.5e8
# poisson = .2
# fric = 0.5
# rMean=.007
# num=1000
# seed=1            #for 3D_1000_mass8



# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .7
# fric = 0.5
# rMean=.007
# num=1000
# seed=1            #for 3D_1000_mass9






# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .2
# fric = 0.5
# rMean=.007
# num=1000
# seed=1            #for 3D_1000_mass10



# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .8
# fric = 0.15
# rMean=.007
# num=1000
# seed=1            #for 3D


# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .8
# fric = 0.2
# rMean=.01
# num=600
# seed=1            #for 3D


# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .2
# fric = 0.35
# rMean=.01
# num=600
# seed=1            #for 3D


# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .2
# fric = 0.35
# rMean=.007
# num=1000
# seed=1            #for 3D



# confining = 100e3
# size = .12
# young = 6.e8
# poisson = .2
# fric = 0.4
# rMean=.004
# num=1000
# seed=555         #for 3D




# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .2
# fric = 0.3
# rMean=.007
# num=1000
# seed=1            #for 3D





# confining = 100e3
# size = .24
# young = 4.5e8
# poisson = .2
# fric = 0.35
# rMean=.01
# num=1000
# seed=17        #for 3D



# confining = 100e3
# size = .24
# young = 4.5e8
# poisson = .2
# fric = 0.4
# rMean=.01
# num=600
# seed=17       #for 3D



# confining = 100e3
# size = .24
# young = 4.e8
# poisson = .2
# fric = 0.4
# rMean=.01
# num=600
# seed=123       #for 3D


# confining = 100e3
# size = .24
# young = 4.e8
# poisson = .2
# fric = 0.35
# rMean=.01
# num=600
# seed=123       #for 3D


# confining = 100e3
# size = .24
# young = 5.e8
# poisson = .2
# fric = 0.35
# rMean=.01
# num=600
# seed=55555      #for 3D



# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .2
# fric = 0.3
# rMean=.01
# num=600
# seed=1            #for 3D


# confining = 100e3
# size = .24
# young = 4.e8
# poisson = .2
# fric = 0.4
# rMean=.01
# num=600
# seed=123       #for 3D


# confining = 100e3
# size = .24
# young = 5.e8
# poisson = .2
# fric = 0.3
# rMean=.01
# num=600
# seed=30       #for 3D


# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .2
# fric = 0.3
# rMean=.01
# num=1200
# rRelFuzz=.4
# seed=1       #for 3D



# confining = 100e3
# size = .12
# young = 6.e8
# poisson = .2
# fric = 0.5
# rMean=.004
# num=1000
# rRelFuzz=.5
# seed=1       #for 3D



# confining = 100e3
# size = .24
# young = 4.e8
# poisson = .2
# fric = 0.5
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=1       #for 3D



# confining = 100e3
# size = .24
# young = 6.e8
# poisson = .2
# fric = 0.5
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=1       #for 3D


# confining = 100e3
# size = .12
# young = 3.e8
# poisson = .3
# fric = 0.5
# rMean=.005
# num=1000
# rRelFuzz=.2
# seed=1       #for 3D



# confining = 100e3
# size = .12
# young = 3.e8
# poisson = .2
# fric = 0.6
# rMean=.005
# num=1000
# rRelFuzz=.2
# seed=1       #for 3D



# confining = 100e3
# size = .24
# young = 4.e8
# poisson = .3
# fric = 0.5
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=1       #for 3D




# confining = 100e3
# size = .24
# young = 3.5e8
# poisson = .2
# fric = 0.6
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=1       #for 3D


# confining = 100e3
# size = .24
# young = 3.e8
# poisson = .2
# fric = 0.6
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=15       #for 3D


# confining = 100e3
# size = .24
# young = 3.e8
# poisson = .2
# fric = 0.5
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=15       #for 3D


# confining = 100e3
# size = .12
# young = 3.e8
# poisson = .2
# fric = 0.6
# rMean=.005
# num=1000
# rRelFuzz=.2
# seed=1       #for 3D



# confining = 100e3
# size = .24
# young = 3.e8
# poisson = .3
# fric = 0.5
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=15       #for 3D


# confining = 100e3
# size = .12
# young = 1.e8
# poisson = .4
# fric = 0.2
# rMean=.005
# num=1000
# rRelFuzz=.2
# seed=1       #for 3D no rotation



# confining = 100e3
# size = .12
# young = 2.e8
# poisson = .4
# fric = 0.2
# rMean=.005
# num=1000
# rRelFuzz=.2
# seed=1       #for 3D no rotation



# confining = 100e3
# size = .12
# young = 6.e8
# poisson = .8
# fric = 0.5
# rMean=.004
# num=1000
# rRelFuzz=.5
# seed=1       #for 3D


# confining = 100e3
# size = .24
# young = 4.e8
# poisson = .2
# fric = 0.5
# rMean=.01
# rRelFuzz=.4
# num=600
# seed=123      #for 3D


# confining = 100e3
# size = .24
# young = 4.e8
# poisson = .2
# fric = 0.5
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=1       #for 3D




# confining = 100e3
# size = .24
# young = 5.e8
# poisson = .2
# fric = 0.5
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=12345       #for 3D



# confining = 100e3
# size = .24
# young = 5.e8
# poisson = .3
# fric = 0.5
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=12345      #for 3D





# confining = 100e3
# size = .24
# young = 4.e8
# poisson = .3
# fric = 0.2
# rMean=.01
# num=1000
# rRelFuzz=.5
# seed=1       #for 3D



confining = 100e3
size = .24
young = 6.e8
poisson = .2
fric = 0.5
rMean=.01
rRelFuzz=.4
num=600
seed=1    #for 3D with rotation # mass=2700




O.materials.append(CohFrictMat(young = young, poisson = poisson, frictionAngle=.0,  etaRoll=0.1, etaTwist=0.1))
# CohFrictPhys(cohesionDisablesFriction=True)
# O.interactions.append(CohFrictPhys(cohesionDisablesFriction=True))




sp = pack.SpherePack()
sp.makeCloud(minCorner=(0, 0, 0), maxCorner=(size, size, size), rMean = rMean, rRelFuzz=rRelFuzz, num = num,
             periodic=True,seed=seed)

sp.toSimulation()
O.cell.hSize = Matrix3(size, 0, 0, 0, size, 0, 0, 0, size)
print(len(O.bodies))

# p = O.bodies[0]
# p.state.mass
# p.state.inertia

for p in O.bodies:
    p.state.mass = 2650 * 1.333 * pi * p.shape.radius * p.shape.radius ** 2
    # p.state.blockedDOFs = 'XYZ'
    inertia = 0.4 * p.state.mass * p.shape.radius ** 2
    p.state.inertia = (inertia, inertia, inertia)



O.dt = utils.PWaveTimeStep()
print(O.dt)






O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb()]),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom6D()],
        [Ip2_CohFrictMat_CohFrictMat_CohFrictPhys(shearCohesion=1e5)],
        [Law2_ScGeom6D_CohFrictPhys_CohesionMoment(always_use_moment_law=True, useIncrementalForm=True)]
    ),

    PeriTriaxController(
        dynCell=True,
        goal=(-confining, -confining, -confining),
        stressMask=7,
        relStressTol=.001,
        maxUnbalanced=.001,
        maxStrainRate=(.5, .5, .5),
        doneHook='term()',
        label='triaxial'
    ),
    NewtonIntegrator(damping=.1)
]


def term():
    O.engines = O.engines[:3] + O.engines[4:]
    zSize = O.cell.hSize[2, 2]
    echo(
        'stress:          %s' % str(getStress()),
        'hSize:           %s' % str(O.cell.hSize),
        'zlen:            %.3e' % zSize,
        'void ratio       %.3e' % voxelPorosity(resolution=200, start=(0, 0, 0), end=(zSize, zSize, zSize)),
        'Contact Friction %.3e' % fric,
        'Confing          %.3e kPa' % (confining/1e3),
    )

    # zSize = O.cell.hSize[2, 2]
    # print(voidratio2D(zlen=.24))
    # print(voxelPorosity(resolution=200, start=(0, 0, 0), end=(size, size, size)))
    # print(getStress())
    # print(O.cell.hSize)

    setContactFriction(fric)
    O.cell.trsf = Matrix3.Identity
    O.cell.velGrad = Matrix3.Zero
    for p in O.bodies:
        p.state.vel = Vector3.Zero
        p.state.angVel = Vector3.Zero
        p.state.refPos = p.state.pos
        p.state.refOri = p.state.ori
    O.save('3D_cohfric.yade.gz')
    O.pause()


O.run()
O.wait()