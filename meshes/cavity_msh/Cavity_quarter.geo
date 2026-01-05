// Quarter-domain annulus mesh for cavity expansion
// Open in Gmsh and Mesh -> 2D to generate.

SetFactory("OpenCASCADE");

// Geometry parameters
inner_radius = 15.0;   // cavity radius a
outer_radius = 150.0;  // farfield radius R

// Mesh parameters
n_radial = 20;        // divisions from inner to outer radius
n_theta = 20;         // divisions along the 90° arc
radial_progression = 1.1; // >1 clusters toward inner radius

// Points
Point(1) = {inner_radius, 0, 0, 1.0};
Point(2) = {0, inner_radius, 0, 1.0};
Point(3) = {outer_radius, 0, 0, 1.0};
Point(4) = {0, outer_radius, 0, 1.0};
Point(5) = {0, 0, 0, 1.0};

// Curves
Circle(1) = {1, 5, 2}; // inner arc (0° -> 90°)
Circle(2) = {3, 5, 4}; // outer arc (0° -> 90°)
Line(3) = {1, 3};      // radial line on x-axis
Line(4) = {2, 4};      // radial line on y-axis

// Surface
Line Loop(1) = {3, 2, -4, -1};
Plane Surface(1) = {1};

// Structured mesh
Transfinite Line {1, 2} = n_theta + 1;
Transfinite Line {3, 4} = n_radial + 1 Using Progression radial_progression;
Transfinite Surface {1};
Recombine Surface {1};

// Physical groups (optional)
Physical Curve("inner_boundary") = {1};
Physical Curve("outer_boundary") = {2};
Physical Curve("symmetry_x") = {3};
Physical Curve("symmetry_y") = {4};
Physical Surface("domain") = {1};