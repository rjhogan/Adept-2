/* rosenbrock_banana_function.cpp - N-dimensional Rosenbrock function

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

// This function is an N-dimensional extension of Rosenbrock's banana
// function; it is actually the "2nd De Jong function" - see the
// Wikipedia entry for Rosenbrock's function.

#include "state.h"

using adept::adouble;
adouble State::calc_function_value(const adouble* x) {
  adouble sum = 0.0;
  for (unsigned int i = 0; i < nx()-1; i++) {
    adouble a = x[i+1]-x[i]*x[i];
    sum += (1.0-x[i])*(1.0-x[i]) + 100.0*a*a;
  }
  return sum;
}
