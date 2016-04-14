/* algorithm.cpp - A simple demonstration algorithm used in Tests 1 & 2 

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

*/


#include <cmath>

#include "algorithm.h"
using adept::adouble;

// A simple demonstration algorithm used in the Adept paper. Note that
// this algorithm can be compiled with
// -DADEPT_NO_AUTOMATIC_DIFFERENTIATION to create a version that takes
// double arguments and returns a double result.
adouble algorithm(const adouble x[2]) {
  adouble y = 4.0;
  adouble s = 2.0*x[0] + 3.0*x[1]*x[1];
  double b=3.0;
  y = s + b;
  y *= sin(s);
  return y;
}
 
