/* test_adept_with_and_without_ad.cpp

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

// Demonstration of the use of Adept with code (in this case,
// algorithm.cpp) that has been compiled twice, once with automatic
// differentiation enabled (the default) and once with it disabled
// (using -DADEPT_NO_AUTOMATIC_DIFFERENTIATION) to provide a faster
// version of a function that works with double rather than adouble
// objects.

#include <iostream>

#include "adept.h"

// Provide function prototypes for "algorithm"; see algorithm.cpp for
// the contents of the function
#include "algorithm_with_and_without_ad.h"

// Simple demonstration of automatic differentiation using Adept
int
main(int argc, char** argv)
{
  using adept::adouble;

  // Start an Adept stack before the first adouble object is
  // constructed
  adept::Stack s;

  adouble x[2]; // Our independent variables
  adouble y;    // Our dependent variable

  // Set the values of x
  x[0] = 2.0;
  x[1] = 3.0;


  // PART 1: NUMERICAL ADJOINT
  std::cout << "*** Computing numerical adjoint ***\n\n";

  // We will compare the Adept result to a numerically computed
  // adjoint, so define the perturbation size
  double dx = 1.0e-5;

  // Initialize a inactive version of x as double rather than adouble
  // variables
  double x_r[2];
  x_r[0] = x[0].value();
  x_r[1] = x[1].value();

  // Run the original version of the algorithm that takes real
  // arguments; this was compiled from algorithm.cpp using the
  // -DADEPT_NO_AUTOMATIC_DIFFERENTIATION flag to produce the
  // algorithm_noad.o object file
  double y_real = algorithm(x_r);

  // Now perturb x[0] and x[1] in turn and get a numerical estimate of
  // the gradient
  x_r[0] = x[0].value()+dx;
  x_r[1] = x[1].value();
  double dy_dx0 = (algorithm(x_r)-y_real)/dx;
  x_r[0] = x[0].value();
  x_r[1] = x[1].value()+dx;
  double dy_dx1 = (algorithm(x_r)-y_real)/dx;

  // Print information about the data held in the stack
  std::cout << "Stack status after numerical adjoint (number of operations should be zero):\n" 
	    << s << "\n";


  // PART 2: REVERSE-MODE AUTOMATIC DIFFERENTIATION
  std::cout << "*** Computing adjoint using automatic differentiation ***\n\n";

  // Start a new recording of derivative statements (note that this
  // must be done after the independent variables x[0] and x[1] are
  // initialized
  s.new_recording();

  // Now use Adept to do it - first run the algorithm overloaded for
  // adouble arguments
  y = algorithm(x);

  // Print information about the data held in the stack
  std::cout << "Stack status after algorithm run but adjoint not yet computed:\n"
	    << s << "\n";

  // If we set the adjoint of the dependent variable to 1 then the
  // resulting adjoints of the independent variables after
  // reverse-mode automatic differentiation will be comparable to the
  // outputs of the numerical differentiation
  y.set_gradient(1.0);

  // Print out some diagnostic information
  std::cout << "List of derivative statements:\n";
  s.print_statements();
  std::cout << "\n";

  std::cout << "Initial list of gradients:\n";
  s.print_gradients();
  std::cout << "\n";

  // Run the adjoint algorithm (reverse-mode differentiation)
  s.reverse();

  std::cout << "Final list of gradients:\n";
  s.print_gradients();
  std::cout << "\n";
  
  // Extract the adjoints of the independent variables
  double x0_ad = 0, x1_ad = 0; 
  x[0].get_gradient(x0_ad);
  x[1].get_gradient(x1_ad);


  // PART 3: JACOBIAN COMPUTATION

  // Here we use the same recording to compute the Jacobian matrix
  std::cout << "*** Computing Jacobian matrix ***\n\n";

  s.independent(x, 2); // Declare independents
  s.dependent(y);      // Declare dependents
  double jac[2];
  s.jacobian(jac);     // Compute Jacobian


  // PART 4: PRINT OUT RESULT

  // Print information about the data held in the stack
  std::cout << "Stack status after adjoint and Jacobian computed:\n"
	    << s << "\n";

  // Print memory information
  std::cout << "Memory usage: " << s.memory() << " bytes\n\n";

  std::cout << "Result of forward algorithm:\n";
  std::cout << "  y[from algorithm taking double arguments]  = " << y_real << "\n";
  std::cout << "  y[from algorithm taking adouble arguments] = " << y.value() << "\n\n";
  
  std::cout << "Comparison of gradients:\n";
  std::cout << "  dy_dx0[numerical] = " << dy_dx0 << "\n";
  std::cout << "  dy_dx0[adjoint]   = " << x0_ad  << "\n";
  std::cout << "  dy_dx0[jacobian]  = " << jac[0] << "\n";
  std::cout << "  dy_dx1[numerical] = " << dy_dx1 << "\n";
  std::cout << "  dy_dx1[adjoint]   = " << x1_ad  << "\n";
  std::cout << "  dy_dx1[jacobian]  = " << jac[1] << "\n";

  std::cout << "\nNote that the numerical gradients are less accurate since they use\n"
	    << "a finite difference and are also succeptible to round-off error.\n";

  return 0;

}
