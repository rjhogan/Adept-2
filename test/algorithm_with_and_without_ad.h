/* algorithm_with_and_without_ad.h

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

// This header file defining the interface of the simple demonstration
// function "algorithm", and is included by
// test_adept_with_and_without_ad.cpp. It demonstrates the use of a
// single source file that is compiled twice to produce two overloaded
// versions of a function. The "original" version takes
// double-precision arguments and returns a double-precision answer,
// while the automatic differentiation version takes adouble arguments
// and returns an adouble answer. The two versions are compiled from
// the same source file algorithm.cpp by compiling it twice with and
// without the compiler option -DAUTOMATIC_DIFFERENTIATION.


#ifndef ALGORITHM_WITH_AND_WITHOUT_AD_H
#define ALGORITHM_WITH_AND_WITHOUT_AD_H 1

#include "adept.h"

// Declare the original version of the function
double algorithm(const double x[2]);

#ifndef ADEPT_NO_AUTOMATIC_DIFFERENTIATION
// Declare the automatic-differentiation version of the function
adept::adouble algorithm(const adept::adouble x[2]);
#endif

#endif
