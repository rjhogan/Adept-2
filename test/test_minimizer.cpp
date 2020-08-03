/* test_minimizer.cpp - Test Adept minimizer with N-dimensional Rosenbrock function

  Copyright (C) 2020 ECMWF

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

//#include <string> // for std::stoi in C++11
#include <cstdio>  // for std::sscanf in C++98
#include <iostream>
#include <adept_optimize.h>

using namespace adept;

class RosenbrockN : public Optimizable {

  // N-dimensional Rosenbrock function can be expressed as the sum of
  // the squared elements of vector y(x) defined as follows.  This
  // form facilitates the calculation of the Hessian from the Jacobian
  // dy/dx.  It is templated so that can be called either with a
  // passive "Vector" or active "aVector" argument.
  template <bool IsActive>
  Array<1,Real,IsActive> calc_y(const Array<1,Real,IsActive>& x) {
    int nx = x.size();
    Array<1,Real,IsActive> y((nx-1)*2);
    for (int ix = 0; ix < nx-1; ++ix) {
      y(ix*2)   = 10.0 * (x(ix+1)-x(ix)*x(ix));
      y(ix*2+1) = 1.0 - x(ix);
    }
    y *= sqrt(2.0);
    return y;
  }
    
  virtual void report_progress(int niter, const adept::Vector& x,
			       Real cost, Real gnorm) {
    std::cout << "Iteration " << niter
	      << ": cost=" << cost << ", gnorm=" << gnorm << "\n";
    // For plotting progress, direct standard error to a text file
    for (int ix = 0; ix < x.size(); ++ix) {
      std::cerr << x(ix) << " ";
    }
    std::cerr << cost << "\n";
  }

  virtual bool provides_derivative(int order) {
    if (order >= 0 && order <= 2) {
      return true;
    }
    else {
      return false;
    }
  }

  virtual Real calc_cost_function(const Vector& x) {
    //std::cout << "  test x: " << x << "\n";
    Vector y = calc_y(x);
    return 0.5*sum(y*y);
  }

  virtual Real calc_cost_function_gradient(const Vector& x,
					   Vector gradient) {
    Stack stack;
    aVector xactive = x;
    stack.new_recording();
    aVector y = calc_y(xactive);
    aReal cost = 0.5*sum(y*y);
    cost.set_gradient(1.0);
    stack.reverse();
    gradient = xactive.get_gradient();
    return value(cost);
  }

  virtual Real calc_cost_function_gradient_hessian(const Vector& x,
						   Vector gradient,
						   SymmMatrix& hessian) {
    Stack stack;
    aVector xactive = x;
    stack.new_recording();
    aVector y = calc_y(xactive);
    aReal cost = 0.5*sum(y*y);
    stack.independent(xactive);
    stack.dependent(y);
    Matrix jac = stack.jacobian();
    hessian  = jac.T() ** jac;
    gradient = jac.T() ** value(y);
    return value(cost);
  }

};

int
main(int argc, const char* argv[])
{
  RosenbrockN rosenbrock;
  Minimizer minimizer(MINIMIZER_ALGORITHM_LEVENBERG_MARQUARDT);
  int nx = 2;
  if (argc > 1) {
    // nx = std::stoi(argv[1]);
    std::sscanf(argv[1], "%d", &nx);
    if (argc > 2) {
      minimizer.set_algorithm(argv[2]);
      if (argc > 3) {
	int max_it;
	// max_it = std::stof(argv[3]);
	std::sscanf(argv[3], "%d", &max_it);
	minimizer.set_max_iterations(max_it);
	if (argc > 4) {
	  double converged_grad_norm;
	  //converged_grad_norm = std::stof(argv[4]);
	  std::sscanf(argv[4], "%lf", &converged_grad_norm);
	  minimizer.set_converged_gradient_norm(converged_grad_norm);
	}
      }
    }
  }
  else {
    std::cout << "Usage: " << argv[0] << " [nx] [Levenberg|Levenberg-Marquardt|L-BFGS] [max_iterations] [converged_gradient_norm]\n";
  }

  //minimizer.set_levenberg_damping_start(0.0);
  //minimizer.set_max_step_size(1.0);
  //minimizer.set_levenberg_damping_multiplier(2.0, 5.0);
  minimizer.ensure_updated_state(2);

  std::cout << "Minimizing " << nx << "-dimensional Rosenbrock function\n";
  std::cout << "Algorithm: " << minimizer.algorithm_name() << "\n";
  std::cout << "Maximum iterations: " << minimizer.max_iterations() << "\n";
  std::cout << "Converged gradient norm: " << minimizer.converged_gradient_norm() << "\n";

  // Initial state vector
  Vector x(nx);
  x = -3.0;
  //  x = {-3.0, 10.0};

  bool is_bounded = false;
  MinimizerStatus status;

  if (is_bounded) {
    Vector x_lower, x_upper;
    adept::minimizer_initialize_bounds(nx, x_lower, x_upper);
    x_upper(1) = 2.0;
    x_lower(1) = 0.2;
    status = minimizer.minimize(rosenbrock, x, x_lower, x_upper);
  }
  else {
    status = minimizer.minimize(rosenbrock, x);
  }

  std::cout << "Status: " << minimizer_status_string(status) << "\n";
  std::cout << "Solution: x=" << x << "\n";
  std::cout << "Number of samples: " << minimizer.n_samples() << "\n";

  return static_cast<int>(status);
}
