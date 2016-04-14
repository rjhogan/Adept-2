/* algorithm.h - Header file for the simple example algorithm function

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#ifndef ALGORITHM_H
#define ALGORITHM_H 1

// This header file defining the interface of the simple demonstration
// function "algorithm".  This header file is included by both
// algorithm.cpp, which defines the body of the function, and
// test_adept.cpp, which calls algorithm. 

#include "adept.h"

// Declare the function
adept::adouble algorithm(const adept::adouble x[2]);

#endif
