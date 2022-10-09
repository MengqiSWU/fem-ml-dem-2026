# FEM-ML-DEM
- Author: shguanWHU
- phone: +86-13016427343
- Other contributors will be warmly welcomed.
- This framework is based on the [FEM](https://launchpad.net/escript-finley)/[DEM](https://yade-dem.org/doc/) coupling platform proposed by [Prof. Guo](https://person.zju.edu.cn/nguo). 
- This is a framework of the **data-driven multiscale** mechanical computation platform of the **granular materials**.
- I sorted the files and deleted the unnecessary file.

## Explicit mode implemented
- Reason: the tangent operator is necessary in the Newton iteration. While the tangent operator is not easily accessable especially in the ML model since the tangent operator can not be derived through the differentiation due to the stress-strain fluctuation.
- Up to now, the explicit mode is under debug and testing.

## Simulation

| Biaxil simulation                                                                                                                                         | Retaining wall simumation                                                                                                      |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| <img alt="Shear strain of the smooth biaxial compression" height="300" src="./sciptes4figures/paraview_figures/biaxialSmooth_simulation/ml816_u_50.png"/> | <img alt="Displacement of the retaining wall" height="300" src="./sciptes4figures/paraview_figures/retainingWall/ml2-60.png"/> |
| <img alt="Shear strain of the smooth biaxial compression" height="300" src="./sciptes4figures/paraview_figures/biaxialSmooth_simulation/ml816_u_75.png"/> | <img alt="Displacement of the retaining wall" height="300" src="./sciptes4figures/paraview_figures/retainingWall/ml2-80.png"/> |
| <img alt="Shear strain of the smooth biaxial compression" height="300" src="./sciptes4figures/paraview_figures/biaxialSmooth_simulation/ml816_u_99.png"/> | <img alt="Displacement of the retaining wall" height="300" src="./sciptes4figures/paraview_figures/retainingWall/ml2-99.png"/> |


# 
