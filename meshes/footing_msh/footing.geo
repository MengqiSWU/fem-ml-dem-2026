lc1 = 0.5;
lc2 = 0.1;
height=2.*2.;
width=4.0*2.;
//+
Point(1) = {0, 0, 0, lc1};
//+
Point(2) = {width, 0, 0, lc1};
//+
Point(3) = {width, height, 0, lc1};
//+
Point(4) = { 0, height, 0, lc2};
//+
Point(5) = {.5*width/4.0, height, 0, lc2};
//+
Point(6) = {.8*width/4.0, height, 0, lc2};
//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 6};
Line(4) = {6, 5};
//+
Line(5) = {5, 4};
//+
Line(6) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4, 5, 6};
//+
Plane Surface(1) = {1};

Point(7) = {.5, 1.7, 0, lc2};
Point{7} In Surface{1};

