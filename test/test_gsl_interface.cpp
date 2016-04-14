/* test_gsl_interface.cpp - "main" function for Test 4

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

// This program minimizes the N-dimensional Rosenbrock banana
// function, with the number of dimensions optionally provided on the
// command line

#include <iostream>
#include <vector>
#include <cstdlib>

#include "state.h"

int
main(int argc, char** argv)
{
  std::cout << "Testing Adept-GSL interface using N-dimensional Rosenbrock function\n";
  std::cout << "Usage: " << argv[0] << " [number_of_dimensions]\n";

  // Read number of dimensions from the command line (default 2)
  int nx = 2;
  if (argc > 1) {
    nx = std::atoi(argv[1]);
  }
   
  if (nx < 2) {
    std::cout << "Error: must have 2 or more dimensions, but "
	      << nx << " requested\n";
    return 1;
  }

  // Create minimization environment (see state.h) and then minimize
  // the function; note that initial values are set on construction.
  State state(nx);
  state.minimize();

  // Print out the result
  std::vector<double> x;
  state.x(x);
  std::cout << "Final state: x = [";
  for (int i = 0; i < nx; i++) {
    std::cout << " " << x[i];
  }
  std::cout << "]\n";
  
  return 0;
}
