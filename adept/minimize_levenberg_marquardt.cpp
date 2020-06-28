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
      Real previous_diag_scaling = 1.0;
      while(true) {
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


  // Minimize the cost function embodied in "optimizable" using the
  // Levenberg-Marquardt algorithm, where "x" is the initial state
  // vector and also where the solution is stored.
  MinimizerStatus
  minimize_levenberg_marquardt_bounded(Optimizable2& optimizable,
				       Vector x,
				       const Vector& min_x,
				       const Vector& max_x,
				       int max_iterations,
				       Real converged_gradient_norm,
				       Real max_step_size)
  {
    // Parameters controlling some aspects of algorithm behaviour
    static const Real min_positive_gamma = 1.0/128.0;
    static const Real gamma_multiplier = 4.0;
    static const Real max_gamma = 1000.0;

    if (any(min_x >= max_x)) {
      return MINIMIZER_STATUS_INVALID_BOUNDS;
    }

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
    SymmMatrix modified_hessian(nx);
    SymmMatrix sub_hessian;
    Vector sub_gradient;
    Vector sub_dx;
    hessian = 0.0;
    Real gamma = 1.0;
    Real gnorm;

    // Which state variables are at the minimum bound (-1), maximum
    // bound (1) or free (0)?
    intVector bound_status(nx);
    bound_status = 0;

    // Ensure that initial x lies within the specified bounds
    bound_status.where(x >= max_x) =  1;
    bound_status.where(x <= min_x) = -1;
    x = max(min_x, min(x, max_x));

    int nbound = count(bound_status != 0);
    int nfree  = nx - nbound;

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

      // Find which dimensions are in play
      if (nbound > 0) {
	// We release any dimensions from being at a minimum or
	// maximum bound if two conditions are met: (1) the gradient
	// in that dimension slopes away from the bound, and (2) the
	// Levenberg-Marquardt formula to compute dx using the current
	// value of gamma leads to a point on the valid side of the
	// bound
	modified_hessian = hessian;
	modified_hessian.diag_vector() *= (1.0 + gamma);
	dx = -adept::solve(modified_hessian, gradient);
	// Release points at the minimum bound
	bound_status.where(bound_status == -1
			   && gradient < 0.0
			   && dx > 0.0) = 0;
	// Release points at the maximum bound
	bound_status.where(bound_status == 1
			   && gradient > 0.0
			   && dx < 0.0) = 0;
      }

      nbound = count(bound_status != 0);
      nfree  = nx - nbound;

      // List of indices of free state variables
      intVector ifree(nfree);
      if (nbound > 0) {
	ifree = find(bound_status == 0);
      }
      else {
	ifree = range(0, nx-1);
      }

      // Compute L2 norm of gradient to see how "flat" the environment
      // is, restricting ourselves to the dimensions currently in play
      if (nfree > 0) {
	gnorm = norm2(gradient(ifree));
      }
      else {
	// If no dimensions are in play we are at a corner of the
	// bounds and the gradient is pointing into the corner: we
	// have reached a minimum in the cost function subject to the
	// bounds so have converged
	gnorm = 0.0;
      }
      // Report progress using user-defined function
      optimizable.record_progress(iteration, x, cost, gnorm);
      // Convergence has been achieved if the L2 norm has been reduced
      // to a user-specified threshold
      if (gnorm <= converged_gradient_norm) {
	status = MINIMIZER_STATUS_SUCCESS;
	break;
      }

      sub_gradient.clear();
      sub_hessian.clear();
      if (nbound > 0) {
	sub_gradient = gradient(ifree);
	sub_hessian  = SymmMatrix(Matrix(hessian)(ifree,ifree));
      }
      else {
	sub_gradient >>= gradient;
	sub_hessian  >>= hessian;
      }

      // FIX reuse dx if possible below...

      // Try to minimize cost function 
      Real previous_diag_scaling = 1.0;
      while(true) {
	sub_dx.resize(nfree);
	if (gamma > 1000.0) {
	  // Try steepest descent instead
	  sub_dx = -sub_gradient/gamma;
	}
	else {
	  // Levenberg-Marquardt formula: scale the diagonal of the
	  // Hessian, where the larger the value of gamma, the closer
	  // the resulting behaviour is to steepest descent
	  sub_hessian.diag_vector() *= (1.0 + gamma)/previous_diag_scaling;
	  previous_diag_scaling = 1.0 + gamma;
	  sub_dx = -adept::solve(sub_hessian, sub_gradient);
	}

	// Limit the maximum step size, if required
	if (max_step_size > 0.0) {
	  Real max_dx = maxval(abs(dx));
	  if (max_dx > max_step_size) {
	    sub_dx *= (max_step_size/max_dx);
	  }
	}

	// Check for collision with new bounds
	intVector new_min_bounds = find(x(ifree)+sub_dx <= min_x(ifree));
	intVector new_max_bounds = find(x(ifree)+sub_dx >= max_x(ifree));
	Real mmin_frac = 2.0;
	Real mmax_frac = 2.0;
	int imin = 0, imax = 0;
	if (!new_min_bounds.empty()) {
	  Vector min_frac = -(x(ifree(new_min_bounds)) - min_x(ifree(new_min_bounds)))
	    / sub_dx(new_min_bounds);
	  mmin_frac = minval(min_frac);
	  imin = new_min_bounds(minloc(min_frac));
	}
	if (!new_max_bounds.empty()) {
	  Vector max_frac = (max_x(ifree(new_max_bounds)) - x(ifree(new_max_bounds)))
	    / sub_dx(new_max_bounds);
	  mmax_frac = minval(max_frac);
	  imax = new_max_bounds(maxloc(max_frac));
	}

	Real frac = 1.0;
	int bound_type = 0;
	int ibound = 0;
	if (mmin_frac <= 1.0 || mmax_frac <= 1.0) {
	  if (mmin_frac < mmax_frac) {
	    frac = mmin_frac;
	    ibound = imin;
	    bound_type = -1;
	  }
	  else {
	    frac = mmax_frac;
	    ibound = imax;
	    bound_type = 1;
	  }	  
	  sub_dx *= frac;
	}

	// Compute new cost state vector and cost function, but not
	// gradient or Hessian for efficiency
	new_x = x;
	new_x(ifree) += sub_dx;
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
	  if (frac < 1.0) {
	    // Found a new bound
	    bound_status(ifree(ibound)) = bound_type;
	  }
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
