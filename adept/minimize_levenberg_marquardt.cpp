/* minimize_levenberg_marquardt.cpp -- Minimize function using Levenberg-Marquardt algorithm

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
  Minimizer::minimize_levenberg_marquardt(Optimizable& optimizable, Vector x,
					  bool use_additive_damping)
  {
    int nx = x.size();

    // Initial values
    n_iterations_ = 0;
    n_samples_ = 0;
    status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
    cost_function_ = std::numeric_limits<Real>::infinity();

    Real new_cost;

    // The main memory storage for the Levenberg family of methods
    // consists of the following three vectors...
    Vector new_x(nx);
    Vector gradient(nx);
    Vector dx(nx);

    // ...and the Hessian matrix, which is stored explicitly
    SymmMatrix hessian(nx);
    hessian = 0.0;

    Real damping = levenberg_damping_start_;
    gradient_norm_ = -1.0;

    // Original Levenberg is additive to the diagonal of the Hessian
    // so to make the performance insensitive to an arbitrary scaling
    // of the cost function, we scale the damping factor by the mean
    // of the diagonal of the Hessian
    Real diag_scaling;

    // Does the last calculation of the cost function in "optimizable"
    // match the current contents of the state vector x? -1=no, 0=yes,
    // 1=yes and the last calculation included the gradient, 2=yes and
    // the last calculation included gradient and Hessian.
    int state_up_to_date = -1;

    do {
      // At this point we have either just started or have just
      // reduced the cost function
      cost_function_ = optimizable.calc_cost_function_gradient_hessian(x, gradient, hessian);
      diag_scaling = mean(hessian.diag_vector());
      state_up_to_date = 2;
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

      // Try to minimize cost function 
      Real previous_diag_scaling  = 1.0; // Used in Levenberg-Marquardt version
      Real previous_diag_modifier = 0.0; // Used in Levenberg version
      while(true) {
	if (!use_additive_damping) {
	  // Levenberg-Marquardt formula: scale the diagonal of the
	  // Hessian, where the larger the value of "damping", the
	  // closer the resulting behaviour is to steepest descent
	  hessian.diag_vector() *= (1.0 + damping)/previous_diag_scaling;
	  previous_diag_scaling = 1.0 + damping;
	}
	else {
	  // Older Levenberg approach: add to the diagonal instead
	  hessian.diag_vector() += damping*diag_scaling - previous_diag_modifier;
	  previous_diag_modifier = damping*diag_scaling;
	}
	dx = -adept::solve(hessian, gradient);

	// Limit the maximum step size, if required
	if (max_step_size_ > 0.0) {
	  Real max_dx = maxval(abs(dx));
	  if (max_dx > max_step_size_) {
	    dx *= (max_step_size_/max_dx);
	  }
	}

	// Compute new cost state vector and cost function, but not
	// gradient or Hessian for efficiency
	new_x = x+dx;
	new_cost = optimizable.calc_cost_function(new_x);
	state_up_to_date = -1;
	++n_samples_;

	// If cost function is not finite it may be possible to
	// recover by trying smaller step sizes
	bool cost_invalid = !std::isfinite(new_cost);

	if (new_cost >= cost_function_ || cost_invalid) {
	  // We haven't managed to reduce the cost function: increase
	  // damping value to take smaller steps
	  if (damping <= 0.0) {
	    damping = levenberg_damping_restart_;
	  }
	  else if (damping < levenberg_damping_max_) {
	    damping *= levenberg_damping_multiplier_;
	  }
	  else {
	    // The damping value is now larger than the maximum so we
	    // can get no further
	    if (cost_invalid) {
	      status_ = MINIMIZER_STATUS_INVALID_COST_FUNCTION;
	    }
	    else {
	      status_ = MINIMIZER_STATUS_FAILED_TO_CONVERGE;
	    }
	    break;
	  }
	}
	else {
	  // Managed to reduce cost function
	  x = new_x;
	  n_iterations_++;
	  // Reduce damping for next iteration
	  if (damping > levenberg_damping_min_) {
	    damping /= levenberg_damping_divider_;
	  }
	  else {
	    damping = 0.0;
	  }
	  if (n_iterations_ >= max_iterations_) {
	    status_ = MINIMIZER_STATUS_MAX_ITERATIONS_REACHED;
	  }
	  break;
	}
      } // Inner loop
    }
    while (status_ == MINIMIZER_STATUS_NOT_YET_CONVERGED);
     
    if (state_up_to_date < ensure_updated_state_) {
      // The last call to calc_cost_function* was not with the state
      // vector returned to the user, and they want it to be.  Note
      // that the cost function and gradient norm ought to be
      // up-to-date already at this point.
      if (ensure_updated_state_ > 0) {
	// User wants at least the first derivative, but
	// calc_cost_function_gradient() is not guaranteed to be
	// present so we call the hessain function
	cost_function_ = optimizable.calc_cost_function_gradient_hessian(x, gradient,
									 hessian);
      }
      else {
	// User does not need derivatives to have been computed
	cost_function_ = optimizable.calc_cost_function(x);
      }
    }

    return status_;
  }


  // Minimize the cost function embodied in "optimizable" using the
  // Levenberg-Marquardt algorithm, where "x" is the initial state
  // vector and also where the solution is stored, subject to the
  // constraint that x lies between min_x and max_x.
  MinimizerStatus
  Minimizer::minimize_levenberg_marquardt_bounded(Optimizable& optimizable,
						  Vector x,
						  const Vector& min_x,
						  const Vector& max_x,
						  bool use_additive_damping)
  {
    if (any(min_x >= max_x)
	|| min_x.size() != x.size()
	|| max_x.size() != x.size()) {
      return MINIMIZER_STATUS_INVALID_BOUNDS;
    }

    int nx = x.size();

    // Initial values
    n_iterations_ = 0;
    n_samples_ = 0;
    status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
    cost_function_ = std::numeric_limits<Real>::infinity();

    Real new_cost;

    // The main memory storage for the Levenberg family of methods
    // consists of the following three vectors...
    Vector new_x(nx);
    Vector gradient(nx);
    Vector dx(nx);

    // ...and the Hessian matrix, which is stored explicitly
    SymmMatrix hessian(nx);
    SymmMatrix modified_hessian(nx);
    SymmMatrix sub_hessian;
    Vector sub_gradient;
    Vector sub_dx;
    hessian = 0.0;
    Real damping = levenberg_damping_start_;

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
    gradient_norm_ = -1.0;

    // Original Levenberg is additive to the diagonal of the Hessian
    // so to make the performance insensitive to an arbitrary scaling
    // of the cost function, we scale the damping factor by the mean
    // of the diagonal of the Hessian
    Real diag_scaling;

    // Does the last calculation of the cost function in "optimizable"
    // match the current contents of the state vector x? -1=no, 0=yes,
    // 1=yes and the last calculation included the gradient, 2=yes and
    // the last calculation included gradient and Hessian.
    int state_up_to_date = -1;

    do {
      // At this point we have either just started or have just
      // reduced the cost function
      cost_function_ = optimizable.calc_cost_function_gradient_hessian(x, gradient, hessian);
      diag_scaling = mean(hessian.diag_vector());
      state_up_to_date = 2;
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

      // Find which dimensions are in play
      if (nbound > 0) {
	// We release any dimensions from being at a minimum or
	// maximum bound if two conditions are met: (1) the gradient
	// in that dimension slopes away from the bound, and (2) the
	// Levenberg-Marquardt formula to compute dx using the current
	// value of "damping" leads to a point on the valid side of the
	// bound
	modified_hessian = hessian;
	if (!use_additive_damping) {
	  modified_hessian.diag_vector() *= (1.0 + damping);
	}
	else {
	  modified_hessian.diag_vector() += damping*diag_scaling;
	}
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
	gradient_norm_ = norm2(gradient(ifree));
      }
      else {
	// If no dimensions are in play we are at a corner of the
	// bounds and the gradient is pointing into the corner: we
	// have reached a minimum in the cost function subject to the
	// bounds so have converged
	gradient_norm_ = 0.0;
      }
      // Report progress using user-defined function
      optimizable.report_progress(n_iterations_, x, cost_function_, gradient_norm_);
      // Convergence has been achieved if the L2 norm has been reduced
      // to a user-specified threshold
      if (gradient_norm_ <= converged_gradient_norm_) {
	status_ = MINIMIZER_STATUS_SUCCESS;
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
      Real previous_diag_scaling  = 1.0; // Used in Levenberg-Marquardt version
      Real previous_diag_modifier = 0.0; // Used in Levenberg version
      while(true) {
	sub_dx.resize(nfree);
	if (!use_additive_damping) {
	  // Levenberg-Marquardt formula: scale the diagonal of the
	  // Hessian, where the larger the value of "damping", the
	  // closer the resulting behaviour is to steepest descent
	  sub_hessian.diag_vector() *= (1.0 + damping)/previous_diag_scaling;
	  previous_diag_scaling = 1.0 + damping;
	}
	else {
	  // Older Levenberg approach: add to the diagonal instead
	  sub_hessian.diag_vector() += damping*diag_scaling - previous_diag_modifier;
	  previous_diag_modifier = damping*diag_scaling;
	}
	sub_dx = -adept::solve(sub_hessian, sub_gradient);

	// Limit the maximum step size, if required
	if (max_step_size_ > 0.0) {
	  Real max_dx = maxval(abs(sub_dx));
	  if (max_dx > max_step_size_) {
	    sub_dx *= (max_step_size_/max_dx);
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

	// Compute new state vector and cost function, but not
	// gradient or Hessian for efficiency
	new_x = x;
	new_x(ifree) += sub_dx;
	new_cost = optimizable.calc_cost_function(new_x);
	state_up_to_date = -1;
	++n_samples_;

	// If cost function is not finite it may be possible to
	// recover by trying smaller step sizes
	bool cost_invalid = !std::isfinite(new_cost);

	if (new_cost >= cost_function_ || cost_invalid) {
	  // We haven't managed to reduce the cost function: increase
	  // damping value to take smaller steps
	  if (damping <= 0.0) {
	    damping = levenberg_damping_restart_;
	  }
	  else if (damping < levenberg_damping_max_) {
	    damping *= levenberg_damping_multiplier_;
	  }
	  else {
	    // The damping value is now larger than the maximum so we
	    // can get no further
	    if (cost_invalid) {
	      status_ = MINIMIZER_STATUS_INVALID_COST_FUNCTION;
	    }
	    else {
	      status_ = MINIMIZER_STATUS_FAILED_TO_CONVERGE;
	    }
	    break;
	  }
	}
	else {
	  // Managed to reduce cost function
	  x = new_x;
	  n_iterations_++;
	  if (frac < 1.0) {
	    // Found a new bound
	    bound_status(ifree(ibound)) = bound_type;
	  }
	  // Reduce damping for next iteration
	  if (damping > levenberg_damping_min_) {
	    damping /= levenberg_damping_divider_;
	  }
	  else {
	    damping = 0.0;
	  }
	  if (n_iterations_ >= max_iterations_) {
	    status_ = MINIMIZER_STATUS_MAX_ITERATIONS_REACHED;
	  }
	  break;
	}
      } // Inner loop
    }
    while (status_ == MINIMIZER_STATUS_NOT_YET_CONVERGED);
    
    if (state_up_to_date < ensure_updated_state_) {
      // The last call to calc_cost_function* was not with the state
      // vector returned to the user, and they want it to be.  Note
      // that the cost function and gradient norm ought to be
      // up-to-date already at this point.
      if (ensure_updated_state_ > 0) {
	// User wants at least the first derivative, but
	// calc_cost_function_gradient() is not guaranteed to be
	// present so we call the hessain function
	cost_function_ = optimizable.calc_cost_function_gradient_hessian(x, gradient,
									 hessian);
      }
      else {
	// User does not need derivatives to have been computed
	cost_function_ = optimizable.calc_cost_function(x);
      }
    }

    return status_;
  }

};
