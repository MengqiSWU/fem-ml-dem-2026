width=2.0;
height=width*2.0;
lc1 = width/5.;
lc2 = 0.05*lc1;
//+
Point(1) = {0, 0, 0, lc1};
//+
Point(2) = {width, 0, 0, lc1};
//+
Point(3) = {width, height, 0, lc1};
//+
Point(4) = { 0, height, 0, lc1};

//+
Line(1) = {1, 2};
//+
Line(2) = {2, 3};
//+
Line(3) = {3, 4};
Line(4) = {4, 1};
//+
Curve Loop(1) = {1, 2, 3, 4};
//+
Plane Surface(1) = {1};


