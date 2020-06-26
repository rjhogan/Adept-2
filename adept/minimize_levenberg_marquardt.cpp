/* minimize_levenberg_marquardt.h -- Minimize function using Levenberg-Marquardt algorithm

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <limits>
#include <cmath>
#include <adept/Minimizer.h>

namespace adept {

  // Minimize the cost function embodied in "optimizable" using the
  // Levenberg-Marquardt algorithm, where "x" is the initial state
  // vector and also where the solution is stored.
  MinimizerStatus
  minimize_levenberg_marquardt(Optimizable2& optimizable,
			       Vector x,
			       int max_iterations,
			       Real converged_gradient_norm,
			       Real max_step_size)
  {
    // Parameters controlling some aspects of algorithm behaviour
    static const Real min_positive_gamma = 1.0/128.0;
    static const Real gamma_multiplier = 4.0;
    static const Real max_gamma = 1000.0;

    int nx = x.size();

    // Initial values
    int iteration = 0;
    MinimizerStatus status = MINIMIZER_STATUS_NOT_YET_CONVERGED;
    Real cost = std::numeric_limits<Real>::infinity();

    Real new_cost;
    Vector new_x(nx);
    Vector gradient(nx);
    Vector dx(nx);
    SymmMatrix hessian(nx);
    hessian = 0.0;
    Real gamma = 1.0;
    Real gnorm;

    do {
      // At this point we have either just started or have just
      // reduced the cost function
      cost = optimizable.calc_cost_function_gradient_hessian(x, gradient, hessian);
      // Check cost function and gradient are finite
      if (!std::isfinite(cost)) {
	return MINIMIZER_STATUS_INVALID_COST_FUNCTION;
      }
      else if (any(!isfinite(gradient))) {
	return MINIMIZER_STATUS_INVALID_GRADIENT;
      }
      // Compute L2 norm of gradient to see how "flat" the environment
      // is
      gnorm = norm2(gradient);
      // Report progress using user-defined function
      optimizable.record_progress(iteration, x, cost, gnorm);
      // Convergence has been achieved if the L2 norm has been reduced
      // to a user-specified threshold
      if (gnorm <= converged_gradient_norm) {
	status = MINIMIZER_STATUS_SUCCESS;
	break;
      }

      // Try to minimize cost function 
      while(true) {
	Real previous_diag_scaling = 1.0;
	if (gamma > 1000.0) {
	  // Try steepest descent instead
	  dx = -gradient/gamma;
	}
	else {
	  // Levenberg-Marquardt formula: scale the diagonal of the
	  // Hessian, where the larger the value of gamma, the closer
	  // the resulting behaviour is to steepest descent
	  hessian.diag_vector() *= (1.0 + gamma)/previous_diag_scaling;
	  previous_diag_scaling = 1.0 + gamma;
	  dx = -adept::solve(hessian, gradient);
	}

	// Limit the maximum step size, if required
	if (max_step_size > 0.0) {
	  Real max_dx = maxval(abs(dx));
	  if (max_dx > max_step_size) {
	    dx *= (max_step_size/max_dx);
	  }
	}

	// Compute new cost state vector and cost function, but not
	// gradient or Hessian for efficiency
	new_x = x+dx;
	new_cost = optimizable.calc_cost_function(new_x);

	// If cost function is not finite it may be possible to
	// recover by trying smaller step sizes
	bool cost_invalid = !std::isfinite(new_cost);

	if (new_cost >= cost || cost_invalid) {
	  // We haven't managed to reduce the cost function: increase
	  // gamma value to take smaller steps
	  if (gamma <= 0.0) {
	    gamma = min_positive_gamma;
	  }
	  else if (gamma < max_gamma) {
	    gamma *= gamma_multiplier;
	  }
	  else {
	    // The gamma value is now larger than max_gamma so we can
	    // get no further
	    if (cost_invalid) {
	      status = MINIMIZER_STATUS_INVALID_COST_FUNCTION;
	    }
	    else {
	      status = MINIMIZER_STATUS_FAILED_TO_CONVERGE;
	    }
	    break;
	  }
	}
	else {
	  // Managed to reduce cost function
	  x = new_x;
	  iteration++;
	  // Reduce gamma for next iteration
	  if (gamma > min_positive_gamma) {
	    gamma /= gamma_multiplier;
	  }
	  else {
	    gamma = 0.0;
	  }
	  if (iteration >= max_iterations) {
	    status = MINIMIZER_STATUS_MAX_ITERATIONS_REACHED;
	  }
	  break;
	}
      } // Inner loop
    }
    while (status == MINIMIZER_STATUS_NOT_YET_CONVERGED);
     
    return status;
  }

};
