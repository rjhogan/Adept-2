/* test_misc.cpp

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include "adept.h"
#include "algorithm.h"

// A straight implementation of the trivial example in Hogan (2014)

double algorithm_ad(const double x_val[2], // Input values
                    double* Y_ad,          // Input-output adjoint
                    double x_ad[2]) {      // Output adjoint
  using namespace adept;                   // Import Stack and adouble from adept
  Stack stack;                             // Where differential information is stored
  adouble x[2] = {x_val[0], x_val[1]};     // Initialize adouble inputs
  stack.new_recording();                   // Start recording derivatives
  adouble Y = algorithm(x);                // Version overloaded for adouble args
  Y.set_gradient(*Y_ad);                   // Load the input-output adjoint
  stack.reverse();                         // Run the adjoint algorithm
  x_ad[0] = x[0].get_gradient();           // Extract the output adjoint for x[0]
  x_ad[1] = x[1].get_gradient();           //   ...and x[1]
  *Y_ad   = Y.get_gradient();              // Input-output adjoint has changed too
  return Y.value();                        // Return result of simple computation
}   

int main()
{
  double x[2] = {2.0, 3.0};
  double y_ad = 1.0;
  double x_ad[2];
  double y = algorithm_ad(x, &y_ad, x_ad);
  std::cout << "x[0] = " << x[0] << "\n"
	    << "x[1] = " << x[1] << "\n"
	    << "y    = " << y    << "\n"
	    << "y_ad = " << y_ad << "\n"
	    << "x_ad[0]=" << x_ad[0] << "\n"
	    << "x_ad[1]=" << x_ad[1] << "\n";
  return 0;
}
