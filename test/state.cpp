/* state.cpp - An object-oriented interface to an Adept-based minimizer

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

// Note that this implementation uses the GNU Scientific Library (GSL)
// to provide the quasi-Newton minimization capability

#include <iostream>
#include <gsl/gsl_multimin.h>

#include "state.h"

// C functions needed by GSL

// Return function value given a vector of state variables x
extern "C" 
double my_function_value(const gsl_vector* x, void* params) {
  State* state = reinterpret_cast<State*>(params);
  return state->calc_function_value(x->data);
}
// Return gradient of function with respect to each state variable x
extern "C"
void my_function_gradient(const gsl_vector* x, void* params, gsl_vector* gradJ) { 
  State* state = reinterpret_cast<State*>(params);
  state->calc_function_value_and_gradient(x->data, gradJ->data);
}
// Return both function and its gradient
extern "C"
void my_function_value_and_gradient(const gsl_vector* x, void* params,
				    double* J, gsl_vector* gradJ) { 
  State* state = reinterpret_cast<State*>(params);
  *J = state->calc_function_value_and_gradient(x->data, gradJ->data);
}

using adept::adouble;

// "State" member function for returning the value of the function; it
// does this by calling the underlying calc_function_value(const
// adouble&) function, which is defined in
// rosenbrock_banana_function.cpp.  Since the gradient is not
// required, the recording of automatic differentiation is "paused"
// while this function is called.
double State::calc_function_value(const double* x) {
  stack_.pause_recording();
  for (unsigned int i = 0; i < nx(); ++i) active_x_[i] = x[i];
  double result = value(calc_function_value(&active_x_[0]));
  stack_.continue_recording();
  return result;
}

// Member function for returning both the value of the function and
// its gradient - here Adept is used to compute the gradient
double State::calc_function_value_and_gradient(const double* x, double* dJ_dx) {
  for (unsigned int i = 0; i < nx(); ++i) active_x_[i] = x[i];
  stack_.new_recording();
  adouble J = calc_function_value(&active_x_[0]);
  J.set_gradient(1.0);
  stack_.compute_adjoint();
  adept::get_gradients(&active_x_[0], nx(), dJ_dx);
  return value(J);
}

// Minimize the function, returning true if minimization successful,
// false otherwise
bool State::minimize() {
  // Minimizer settings
  const double initial_step_size = 0.01;
  const double line_search_tolerance = 1.0e-4;
  const double converged_gradient_norm = 1.0e-3;
  // Use the "limited-memory BFGS" quasi-Newton minimizer
  const gsl_multimin_fdfminimizer_type* minimizer_type
    = gsl_multimin_fdfminimizer_vector_bfgs2;
  
  // Declare and populate structure containing function pointers
  gsl_multimin_function_fdf my_function;
  my_function.n = nx();
  my_function.f = my_function_value;
  my_function.df = my_function_gradient;
  my_function.fdf = my_function_value_and_gradient;
  my_function.params = reinterpret_cast<void*>(this);
   
  // Set initial state variables using GSL's vector type: use -5.0 for
  // every value
  gsl_vector *x;
  x = gsl_vector_alloc(nx());
  for (unsigned int i = 0; i < nx(); ++i) gsl_vector_set(x, i, -5.0);

  // Configure the minimizer, and call function once
  gsl_multimin_fdfminimizer* minimizer
    = gsl_multimin_fdfminimizer_alloc(minimizer_type, nx());
  gsl_multimin_fdfminimizer_set(minimizer, &my_function, x,
				initial_step_size, line_search_tolerance);

  // Print out the result of the first function call with the initial
  // state
  std::cout << "Initial state: x = [";
  for (unsigned int i = 0; i < nx(); i++) {
    std::cout << active_x_[i].value() << " ";
  }
  std::cout << "], cost_function = " << minimizer->f << "\n";

  // Begin loop
  size_t iter = 0;
  int status;
  do {
    ++iter;
    // Perform one iteration
    status = gsl_multimin_fdfminimizer_iterate(minimizer);
    
    // Quit loop if iteration failed
    if (status != GSL_SUCCESS) break;
    
    // Test for convergence
    status = gsl_multimin_test_gradient(minimizer->gradient,
					converged_gradient_norm);
     
    // Print out limited number of state variables from this
    // iteration, and the corresponding cost function
    std::cout << "Iteration " << iter << ": x = [";
    for (unsigned int i = 0; i < nx(); i++) {
      std::cout << active_x_[i].value() << " ";
      if (i >= 5) {
	std::cout << "...";
	break;
      }
    }
    std::cout << "], cost_function = " << minimizer->f << "\n";
  }
  while (status == GSL_CONTINUE && iter < 1000);

  // Free memory
  gsl_multimin_fdfminimizer_free(minimizer);
  gsl_vector_free(x);

  // Return true if successfully minimized function, false otherwise
  if (status == GSL_SUCCESS) {
    std::cout << "Minimum found after " << iter << " iterations\n";
    return true;
  }
  else {
    std::cout << "Minimizer failed after " << iter << " iterations: "
	      << gsl_strerror(status) << "\n";
    return false;
  }
}

// Enquiry function to return the current value of the state
// variables, called after minimize() has been run.
void
State::x(std::vector<double>& x_out) const
{
  x_out.resize(nx());
  for (unsigned int i = 0; i < nx(); i++) {
    x_out[i] = active_x_[i].value();
  }
}
