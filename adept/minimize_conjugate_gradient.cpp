/* minimize_conjugate_gradient.cpp -- Minimize function using Conjugate Gradient algorithm

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <limits>
#include <cmath>
#include <adept/Minimizer.h>

namespace adept {

  MinimizerStatus
  Minimizer::minimize_conjugate_gradient(Optimizable& optimizable, Vector x)
  {
    int nx = x.size();

    // Initial values
    n_iterations_ = 0;
    n_samples_ = 0;
    status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
    cost_function_ = std::numeric_limits<Real>::infinity();

    Real new_cost;
    Vector new_x(nx);
    Vector gradient(nx);
    Vector previous_gradient(nx);
    Vector direction(nx);
    Vector test_x(nx);
    int state_up_to_date = -1;
    Real step_size = 1.0;
    if (max_step_size_ > 0.0) {
      step_size = max_step_size_;
    }
    bool do_restart = true;

    do {
      // At this point we have either just started or have just
      // reduced the cost function
      if (!do_restart) {
	previous_gradient = gradient;
      }
      cost_function_ = optimizable.calc_cost_function_gradient(x, gradient);
      state_up_to_date = 1;
      ++n_samples_;
      if (n_iterations_ == 0) {
	start_cost_function_ = cost_function_;
      }

      // Check cost function and gradient are finite
      if (!std::isfinite(cost_function_)) {
	status_ = MINIMIZER_STATUS_INVALID_COST_FUNCTION;
	break;
      }
      else if (any(!isfinite(gradient))) {
	status_ = MINIMIZER_STATUS_INVALID_GRADIENT;
	break;
      }
      // Compute L2 norm of gradient to see how "flat" the environment
      // is
      gradient_norm_ = norm2(gradient);
      // Report progress using user-defined function
      optimizable.report_progress(n_iterations_, x, cost_function_, gradient_norm_);
      // Convergence has been achieved if the L2 norm has been reduced
      // to a user-specified threshold
      if (gradient_norm_ <= converged_gradient_norm_) {
	status_ = MINIMIZER_STATUS_SUCCESS;
	break;
      }

      // Find search direction
      if (do_restart) {
	direction = -gradient;
	do_restart = false;
      }
      else {
	Real beta;
	if (cg_use_fletcher_reeves_) {
	  beta = dot_product(gradient, gradient) 
	    / dot_product(previous_gradient, previous_gradient);
	}
	else {
	  beta = std::max(sum(gradient * (gradient - previous_gradient))
			  / dot_product(previous_gradient, previous_gradient),
			  0.0);
	}
	direction = beta*direction - gradient;
      }

      // Perform line search
      status_ = line_search(optimizable, x, direction,
			   new_x, step_size, gradient, state_up_to_date);

      n_iterations_++;
      if (n_iterations_ >= max_iterations_) {
	status_ = MINIMIZER_STATUS_MAX_ITERATIONS_REACHED;
      }
    }
    while (status_ == MINIMIZER_STATUS_NOT_YET_CONVERGED);
     
    if (state_up_to_date < ensure_updated_state_) {
      // The last call to calc_cost_function* was not with the state
      // vector returned to the user, and they want it to be.
      if (ensure_updated_state_ > 0) {
	// User wants at least the first derivative
	cost_function_ = optimizable.calc_cost_function_gradient(x, gradient);
      }
      else {
	// User does not need derivatives to have been computed
	cost_function_ = optimizable.calc_cost_function(x);
      }
    }

    return status_;
  }

};
