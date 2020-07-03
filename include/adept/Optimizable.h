/* Optimizable.h -- abstract base classes representing an optimization problem

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptOptimizable_H
#define AdeptOptimizable_H 1

#include <adept_arrays.h>

namespace adept {

  // A class representing an optimization problem that can be solved
  // by Adept's Minimizer class. The user should define their own
  // class that publicly inherits from Optimizable and overrides the
  // member functions calc_cost_function and provides_derivative.
  // This is the minimum requirement to use in gradient-free
  // minimization algorithms (e.g. Nelder-Mead). To use in
  // quasi-Newton and conjugate-gradient minimization algorithms, the
  // user should also override the member function
  // calc_cost_function_gradient. To use in Newton-type minimization
  // algorithms such as Gauss-Newton and Levenberg-Marquardt, the user
  // should also override the member function
  // calc_cost_function_gradient_hessian.  The user may optionally
  // override report_progress.
  class Optimizable {
  public:
    virtual ~Optimizable() { }

    // Return the cost function corresponding to the state vector x.
    virtual Real calc_cost_function(const adept::Vector& x) = 0;

    // Return the cost function corresponding to the state vector x,
    // and also set the "gradient" argument to the gradient of the
    // cost function with respect to each element of x.
    virtual Real calc_cost_function_gradient(const adept::Vector& x,
					     adept::Vector gradient) {
      // If we get here then a gradient-based minimizer has been
      // applied to this class but the user has not implemented a
      // function to compute the gradient.
      throw optimization_exception("Gradient calculation has not been implemented");
    }
   
    // Return the cost function corresponding to the state vector x,
    // and set the "gradient" argument to the gradient of the cost
    // function with respect to each element of x, and "hessian" to
    // the second derivative of the cost function with respect to x.
    virtual Real calc_cost_function_gradient_hessian(const adept::Vector& x,
		     adept::Vector gradient, adept::SymmMatrix& hessian) {
      // If we get here then a Newton-type minimizer has been applied
      // to this class but the user has not implemented a function to
      // compute the Hessian matrix.
      throw optimization_exception("Hessian calculation has not been implemented");
    }

    // This function is called at every iteration, and can be
    // overridden by child classes to report or store the progress at
    // each iteration, if required. By default it does nothing.
    virtual void report_progress(int niter, const adept::Vector& x,
				 Real cost, Real gnorm) { }

    // Child classes should override this function to provide a
    // run-time mechanism to check which of the first and second
    // derivative (i.e. gradient and Hessian, respectively) are
    // available.  If only the gradient is available then it could be
    // implemented as: if (order == 0 || order == 1) { return true; }
    // else { return false; }
    virtual bool provides_derivative(int order) = 0;

  };

};

#endif
