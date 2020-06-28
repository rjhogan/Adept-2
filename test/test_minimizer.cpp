/* test_minimizer.cpp - Test Adept minimizer with N-dimensional Rosenbrock function

  Copyright (C) 2020 ECMWF

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include <string>
#include <iostream>
#include <adept_optimize.h>

using namespace adept;

class RosenbrockN : public Optimizable2 {

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
    
  virtual void record_progress(int niter, const adept::Vector& x,
			       Real cost, Real gnorm) {
    std::cout << "Iteration " << niter
	      << ": cost=" << cost << ", gnorm=" << gnorm << "\n";
    // For plotting progress, direct standard error to a text file
    for (int ix = 0; ix < x.size(); ++ix) {
      std::cerr << x(ix) << " ";
    }
    std::cerr << cost << "\n";
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
						   SymmMatrix hessian) {
    Stack stack;
    aVector xactive = x;
    stack.new_recording();
    aVector y = calc_y(xactive);
    aReal cost = 0.5*sum(y*y);
    Matrix jac;
    jac.resize_column_major(dimensions(y.size(), x.size()));
    stack.independent(xactive);
    stack.dependent(y);
    stack.jacobian(jac.data());
    hessian = jac.T() ** jac;
    cost.set_gradient(1.0);
    stack.reverse();
    gradient = xactive.get_gradient();
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
    nx = std::stoi(argv[1]);
    if (argc > 2) {
      Real max_it = std::stof(argv[2]);
      minimizer.max_iterations(max_it);
      std::cout << "Maximum iterations: " << max_it << "\n";
      if (argc > 3) {
	Real converged_grad_norm = std::stof(argv[3]);
	minimizer.converged_gradient_norm(converged_grad_norm);
	std::cout << "Converged gradient norm: " << converged_grad_norm << "\n";
      }
    }
  }
  else {
    std::cout << "Usage: " << argv[0] << " [nx] [max_iterations] [converged_gradient_norm]\n";
  }

  std::cout << "Minimizing " << nx << "-dimensional Rosenbrock function\n";

  // Initial state vector
  Vector x(nx);
  x = -4.0;

  bool is_bounded = true;
  MinimizerStatus status;

  if (is_bounded) {
    Vector x_lower, x_upper;
    adept::initialize_bounds(nx, x_lower, x_upper);
    x_upper(1) = 2.0;
    x_lower(1) = 0.2;
    status = minimizer.minimize(rosenbrock, x, x_lower, x_upper);
  }
  else {
    status = minimizer.minimize(rosenbrock, x);
  }

  std::cout << "Status: " << minimizer_status_string(status) << "\n";
  std::cout << "Solution: x=" << x << "\n";

  return static_cast<int>(status);
}
