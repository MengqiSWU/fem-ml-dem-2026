from __future__ import print_function
import os.path
from yadeimport import *
from utilSelf.general import echo
import numpy as np

# --------------- parameters in lwj's paper ---------------
#confining = 20e3 # 20 kPa
#O.materials.append(FrictMat(young=8e8, poisson=0.5, frictionAngle=0./180*np.pi))
#fric = np.tan(23 / 180 * np.pi)
#fname = os.path.join(os.getcwd(), 'p_2e4.yade.gz')

# --------------- parameters in Guoning's paper ---------------
confining = 100e3  # 100 kPa
O.materials.append(FrictMat(young=6.e8, poisson=0.8, frictionAngle=0./180*np.pi))
fric = 0.5
fname = os.path.join(os.getcwd(), 'p_1e5_600.yade.gz')

# --------------- parameters for the cohesive materials -------
# O.materials.append(CohFrictMat(young=6.e8, poisson=.8, frictionAngle=.0, ))

sp = pack.SpherePack()
size = .24
sp.makeCloud(minCorner=(0, 0, .05), maxCorner=(size, size, .05), rMean=.005, rRelFuzz=.4, num=600, periodic=True,
             seed=1)
sp.toSimulation()
O.cell.hSize = Matrix3(size, 0, 0, 0, size, 0, 0, 0, .1)
print(len(O.bodies))
for p in O.bodies:
    p.state.blockedDOFs = 'zXY'
    p.state.mass = 2650 * 0.1 * np.pi * p.shape.radius ** 2  # 0.1 = thickness of cylindrical particle
    inertia = 0.5 * p.state.mass * p.shape.radius ** 2
    p.state.inertia = (.5 * inertia, .5 * inertia, inertia)

O.dt = utils.PWaveTimeStep()
print(O.dt)

O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb()]),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_FrictPhys()],
        [Law2_ScGeom_FrictPhys_CundallStrack()]
    ),
    PeriTriaxController(
        dynCell=True,
        goal=(-confining, -confining, 0),
        stressMask=3,
        relStressTol=.001,
        maxUnbalanced=.001,
        maxStrainRate=(.5, .5, .0),
        doneHook='term()',
        label='biax'
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
        'void ratio       %.3e' % voidratio2D(zlen=zSize),
        'Contact Friction %.3e' % fric,
        'Confing          %.3e kPa' % (confining/1e3),
    )
    setContactFriction(fric)
    O.cell.trsf = Matrix3.Identity
    O.cell.velGrad = Matrix3.Zero
    for p in O.bodies:
        p.state.vel = Vector3.Zero
        p.state.angVel = Vector3.Zero
        p.state.refPos = p.state.pos
        p.state.refOri = p.state.ori
    echo('Yade packing 2D saved as\t\t %s' % fname)
    O.save(fname)
    O.pause()


O.run()
O.wait()
