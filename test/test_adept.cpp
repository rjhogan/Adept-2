/* test_adept.cpp - Demonstration of basic features of Adept

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include <iostream>

#include "adept.h"

// Provide function prototype for "algorithm"; see algorithm.cpp for
// the contents of the function
#include "algorithm.h"

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

  // We will provide an estimate of the adjoints by perturbing the
  // inputs by a small amount

  adouble x_perturbed[2]; // Perturbed independent variables

  // This version of the code uses the same algorithm function that
  // takes adouble arguments for doing the numerical adjoint, even
  // though we are not doing automatic differentiation. To make it
  // faster, we can turn off the recording of derivative information
  // using the pause_recording function.  This only works if all code
  // has been compiled with the -DADEPT_RECORDING_PAUSABLE flag;
  // otherwise it does nothing (so the program will still run
  // correctly, but will be less efficient). Note that another
  // approach if you want to call a function several times, sometimes
  // with automatic differentiation and sometimes without, is
  // demonstrated in
  // test_adept_with_without_automatic_differentiation.cpp.
  s.pause_recording();

  // We will compare the Adept result to a numerically computed
  // adjoint, so define the perturbation size
  double dx = 1.0e-5;

  // Run the algorithm
  y = algorithm(x);

  // Now perturb x[0] and x[1] in turn and get a numerical estimate of
  // the gradient
  x_perturbed[0] = x[0]+dx;
  x_perturbed[1] = x[1];
  double dy_dx0 = adept::value((algorithm(x_perturbed)-y)/dx);
  x_perturbed[0] = x[0];
  x_perturbed[1] = x[1]+dx;
  double dy_dx1 = adept::value((algorithm(x_perturbed)-y)/dx);

  // Turn the recording of deriviative information back on
  s.continue_recording();

  // Print information about the data held in the stack
  std::cout << "Stack status after numerical adjoint (if recording was successfully\n"
	    << "paused then the number of operations should be zero):\n" 
	    << s;
  // Print memory information
  std::cout << "Memory usage: " << s.memory() << " bytes\n\n";

  // PART 2: REVERSE-MODE AUTOMATIC DIFFERENTIATION

  // Now we use Adept to do the automatic differentiation
  std::cout << "*** Computing adjoint using automatic differentiation ***\n\n";

  // Start a new recording of derivative statements; note that this
  // must be done after the independent variables x[0] and x[1] are
  // defined and after they have been given their initial values
  s.new_recording();

  // Run the algorithm again
  y = algorithm(x);

  // Print information about the data held in the stack
  std::cout << "Stack status after algorithm run but adjoint not yet computed:\n"
	    << s;
  // Print memory information
  std::cout << "Memory usage: " << s.memory() << " bytes\n\n";

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

  // Some more diagnostic information
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
  double jac[2];       // Where the Jacobian will be stored
  s.jacobian(jac);     // Compute Jacobian


  // PART 4: PRINT OUT RESULTS

  // Print information about the data held in the stack
  std::cout << "Stack status after adjoint and Jacobian computed:\n"
	    << s;
  // Print memory information
  std::cout << "Memory usage: " << s.memory() << " bytes\n\n";

  std::cout << "Result of forward algorithm:\n";
  std::cout << "  y = " << y.value() << "\n";
  
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
