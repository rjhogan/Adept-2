/* Optimizable.h -- abstract base classes representing an optimization problem

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptOptimizable_H
#define AdeptOptimizable_H 1

#include <adept_arrays.h>

namespace adept {

  // A class representing an optimization problem where only the cost
  // function can be computed, but not its derivative.  This is
  // suitable for gradient-free minimization algorithms
  // (e.g. Nelder-Mead).  The user should define their own class that
  // publicly inherits from Optimizable0 and overrides
  // calc_cost_function.
  class Optimizable0 {
  public:
    virtual ~Optimizable0() { }
    // Return the cost function corresponding to the state vector x.
    virtual Real calc_cost_function(const adept::Vector& x) = 0;
    // This function is called at every iteration, and can be
    // overridden by child classes to report or store the progress at
    // each iteration, if required. By default it does nothing.
    virtual void record_progress(int niter, const adept::Vector& x,
				 Real cost, Real gnorm) { }
    virtual void initialize_bounds(int nx) {
      x_lower_bound_.resize(nx);
      x_upper_bound_.resize(nx);
      x_lower_bound_ = -std::numeric_limits<Real>::infinity();
      x_upper_bound_ =  std::numeric_limits<Real>::infinity();
    }
    void set_lower_bound(int ix, Real val) {
      x_lower_bound_(ix) = val;
    }
    void set_upper_bound(int ix, Real val) {
      x_upper_bound_(ix) = val;
    }
    void set_lower_bounds(int ix1, int ix2, Real val) {
      x_lower_bound_(range(ix1,ix2)) = val;
    }
    void set_upper_bounds(int ix1, int ix2, Real val) {
      x_upper_bound_(range(ix1,ix2)) = val;
    }
    const Vector& lower_bounds() const { return x_lower_bound_; }
    const Vector& upper_bounds() const { return x_upper_bound_; }
    Real lower_bound(int ix)    const { return x_lower_bound_(ix); }
    Real upper_bound(int ix)    const { return x_upper_bound_(ix); }
    bool have_bounds()          const {
      return !x_lower_bound_.empty() && !x_upper_bound_.empty();
    }
  protected:
    // Data
    Vector x_lower_bound_, x_upper_bound_;
  };

  // A class representing an optimization problem where the cost
  // function can be computed, as well as the gradient of the cost
  // function with respect to each element of the state vector. This
  // is suitable for quasi-Newton and conjugate gradient minimization
  // algorithms. The user should define their own class that publicly
  // inherits from Optimizable1 and overrides both calc_cost_function
  // and calc_cost_function_gradient.
  class Optimizable1 : public Optimizable0 {
  public:
    // Return the cost function corresponding to the state vector x,
    // and also set the "gradient" argument to the gradient of the
    // cost function with respect to each element of x.
    virtual Real calc_cost_function_gradient(const adept::Vector& x,
					     adept::Vector gradient) = 0;
  };

  // A class representing an optimization problem where the cost
  // function, the gradient of the cost function with respect to each
  // element of the state vector, and the Hessian matrix can be
  // computed. This is suitable for Newton-type minimization
  // algorithms such as Gauss-Newton and Levenberg-Marquardt. The user
  // should define their own class that publicly inherits from
  // Optimizable2 and overrides calc_cost_function,
  // calc_cost_function_gradient and
  // calc_cost_function_gradient_hessian.
  class Optimizable2 : public Optimizable1 {
  public:
    // Return the cost function corresponding to the state vector x,
    // and set the "gradient" argument to the gradient of the cost
    // function with respect to each element of x, and "hessian" to
    // the second derivative of the cost function with respect to x.
    virtual Real calc_cost_function_gradient_hessian(const adept::Vector& x,
	     adept::Vector gradient, adept::SymmMatrix hessian) = 0;
  };

};

#endif
