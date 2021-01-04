/* minimize_conjugate_gradient.cpp -- Minimize function using Conjugate Gradient algorithm

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <limits>
#include <cmath>
#include <adept/Minimizer.h>

namespace adept {

  // Minimize the cost function embodied in "optimizable" using the
  // Conjugate-Gradient algorithm, where "x" is the initial state
  // vector and also where the solution is stored. By default the
  // Polak-Ribiere method is used to compute the new search direction,
  // but Fletcher-Reeves is also available.
  MinimizerStatus
  Minimizer::minimize_conjugate_gradient(Optimizable& optimizable, Vector x,
					 bool use_fletcher_reeves)
  {
    int nx = x.size();

    // Initial values
    n_iterations_ = 0;
    n_samples_ = 0;
    status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
    cost_function_ = std::numeric_limits<Real>::infinity();

    // The Conjugate-Gradient method is the most efficient
    // gradient-based method in terms of memory usage, requiring a
    // working memory of just 4*nx, making it suitable for large state
    // vectors.
    Vector gradient(nx);
    Vector previous_gradient(nx);
    Vector direction(nx);
    Vector test_x(nx); // Used by the line search only

    // Does the last calculation of the cost function in "optimizable"
    // match the current contents of the state vector x? -1=no, 0=yes,
    // 1=yes and the last calculation included the gradient, 2=yes and
    // the last calculation included gradient and Hessian.
    int state_up_to_date = -1;

    // Initial step size
    Real step_size = 1.0;
    if (max_step_size_ > 0.0) {
      step_size = max_step_size_;
    }

    // A restart is performed every nx+1 iterations
    bool do_restart = true;
    int iteration_at_last_restart = n_iterations_;

    // Main loop
    while (status_ == MINIMIZER_STATUS_NOT_YET_CONVERGED) {

      // If the last line search found a minimum along the lines
      // satisfying the Wolfe conditions, then the current cost
      // function and gradient will be consistent with the current
      // state vector.  Otherwise we need to compute them.
      if (state_up_to_date < 1) {
	cost_function_ = optimizable.calc_cost_function_gradient(x, gradient);
	state_up_to_date = 1;
	++n_samples_;
      }

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

      // Restart every nx+1 iterations
      if (n_iterations_ - iteration_at_last_restart > nx) {
	do_restart = true;
      }

      // Find search direction
      if (do_restart) {
	// Simple gradient descent after a restart
	direction = -gradient;
	do_restart = false;
	iteration_at_last_restart = n_iterations_;
      }
      else {
	// The brains of the Conjugate-Gradient method - note that
	// generally the Polak-Ribiere method is believed to be
	// superior to Fletcher-Reeves
	Real beta;
	if (use_fletcher_reeves) {
	  // Fletcher-Reeves method
	  beta = dot_product(gradient, gradient) 
	    / dot_product(previous_gradient, previous_gradient);
	}
	else {
	  // Default: Polak-Ribiere method
	  beta = std::max(sum(gradient * (gradient - previous_gradient))
			  / dot_product(previous_gradient, previous_gradient),
			  0.0);
	}
	// beta==0 is equivalent to gradient descent (i.e. a restart)
	if (beta <= 0) {
	  iteration_at_last_restart = n_iterations_;
	}
	// Compute new direction
	direction = beta*direction - gradient;
      }

      // Store gradient for computing beta in next iteration
      previous_gradient = gradient;

      // Perform line search, storing new state vector in x
      MinimizerStatus ls_status
	= line_search(optimizable, x, direction,
		      test_x, step_size, gradient, state_up_to_date,
		      cg_curvature_coeff_);

      if (ls_status == MINIMIZER_STATUS_SUCCESS) {
	// Successfully minimized along search direction: continue to
	// next iteration
	status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
      }
      else if (iteration_at_last_restart != n_iterations_) {
	// Line search either made no progress or encountered a
	// non-finite cost function or gradient, and this was not a
	// restart; try restarting once
	do_restart = true;
	status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
      }
      else {
	// Unrecoverable failure in line-search: return status to
	// calling function
	status_ = ls_status;
      }

      // Better convergence if first step size on next line search is
      // larger than the actual step size on the last line search
      step_size *= 2.0;

      ++n_iterations_;
      if (status_ == MINIMIZER_STATUS_NOT_YET_CONVERGED
	  && n_iterations_ >= max_iterations_) {
	status_ = MINIMIZER_STATUS_MAX_ITERATIONS_REACHED;
      }

      // End of main loop: if status_ is anything other than
      // MINIMIZER_STATUS_NOT_YET_CONVERGED then no more iterations
      // are performed
    }
     
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

  // Minimize the cost function embodied in "optimizable" using the
  // Conjugate-Gradient algorithm, where "x" is the initial state
  // vector and also where the solution is stored, subject to the
  // constraint that x lies between min_x and max_x. By default the
  // Polak-Ribiere method is used to compute the new search direction,
  // but Fletcher-Reeves is also available.
  MinimizerStatus
  Minimizer::minimize_conjugate_gradient_bounded(Optimizable& optimizable, Vector x,
					 const Vector& min_x,
					 const Vector& max_x,
					 bool use_fletcher_reeves)
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

    // The Conjugate-Gradient method is the most efficient
    // gradient-based method in terms of memory usage, requiring a
    // working memory of just 4*nx, making it suitable for large state
    // vectors.
    Vector gradient(nx);
    Vector previous_gradient(nx);
    Vector direction(nx);
    Vector test_x(nx); // Used by the line search only

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

    // Floating-point number containing 1.0 if unbound and 0.0 if
    // bound
    Vector unbound_status(nx);
    unbound_status = 1.0-fabs(bound_status);

    // Does the last calculation of the cost function in "optimizable"
    // match the current contents of the state vector x? -1=no, 0=yes,
    // 1=yes and the last calculation included the gradient, 2=yes and
    // the last calculation included gradient and Hessian.
    int state_up_to_date = -1;

    // Initial step size
    Real step_size = 1.0;
    if (max_step_size_ > 0.0) {
      step_size = max_step_size_;
    }

    // A restart is performed every nx+1 iterations
    bool do_restart = true;
    int iteration_at_last_restart = n_iterations_;

    // Main loop
    while (status_ == MINIMIZER_STATUS_NOT_YET_CONVERGED) {

      // If the last line search found a minimum along the lines
      // satisfying the Wolfe conditions, then the current cost
      // function and gradient will be consistent with the current
      // state vector.  Otherwise we need to compute them.
      if (state_up_to_date < 1) {
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

      }

      // Check whether the bound status of each state variable is
      // consistent with the gradient if a steepest descent were to be
      // taken, and if not flag a restart
      if (any(bound_status == -1 && gradient < 0.0)
	  || any(bound_status == 1 && gradient > 0.0)) {
	bound_status.where(bound_status == -1 && gradient < 0.0) = 0;
	bound_status.where(bound_status ==  1 && gradient > 0.0) = 0;
	unbound_status = 1.0-fabs(bound_status);
	do_restart = true;
      }
      nbound = count(bound_status != 0);
      nfree = nx - nbound;

      // Set gradient at bound points to zero
      gradient.where(bound_status != 0) = 0.0;

      // Compute L2 norm of gradient to see how "flat" the environment
      // is
      if (nfree > 0) {
	gradient_norm_ = norm2(gradient);
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

      // Restart every nx+1 iterations
      if (n_iterations_ - iteration_at_last_restart > nx) {
	do_restart = true;
      }

      // Find search direction
      if (do_restart) {
	// Simple gradient descent after a restart
	direction = -gradient;
	do_restart = false;
	iteration_at_last_restart = n_iterations_;
      }
      else {
	// The brains of the Conjugate-Gradient method - note that
	// generally the Polak-Ribiere method is believed to be
	// superior to Fletcher-Reeves
	Real beta;
	if (use_fletcher_reeves) {
	  // Fletcher-Reeves method
	  beta = dot_product(gradient, gradient) 
	    / dot_product(previous_gradient, previous_gradient);
	}
	else {
	  // Default: Polak-Ribiere method
	  beta = std::max(sum(gradient * (gradient - previous_gradient))
			  / dot_product(previous_gradient, previous_gradient),
			  0.0);
	}
	// beta==0 is equivalent to gradient descent (i.e. a restart)
	if (beta <= 0) {
	  iteration_at_last_restart = n_iterations_;
	}
	// Compute new direction
	direction = beta*direction - gradient;
      }

      // Store gradient for computing beta in next iteration
      previous_gradient = gradient;

      // Distance to the nearest bound
      Real dir_scaling = norm2(direction);
      Real bound_step_size = std::numeric_limits<Real>::max();
      int i_nearest_bound = -1;
      int i_bound_type = 0;
      // Work out the maximum step size along "direction" before a
      // bound is met... there must be a faster way to do this
      for (int ix = 0; ix < nx; ++ix) {
	if (direction(ix) > 0.0 && max_x(ix) < std::numeric_limits<Real>::max()) {
	  Real local_bound_step_size = dir_scaling*(max_x(ix)-x(ix))/direction(ix);
	  if (bound_step_size >= local_bound_step_size) {
	    bound_step_size = local_bound_step_size;
	    i_nearest_bound = ix;
	    i_bound_type = 1;
	  }				   
	}
	else if (direction(ix) < 0.0 && min_x(ix) > -std::numeric_limits<Real>::max()) {
	  Real local_bound_step_size = dir_scaling*(min_x(ix)-x(ix))/direction(ix);
	  if (bound_step_size >= local_bound_step_size) {
	    bound_step_size = local_bound_step_size;
	    i_nearest_bound = ix;
	    i_bound_type = -1;
	  }
	}
      }

      MinimizerStatus ls_status; // line-search outcome
      if (i_nearest_bound >= 0) {
	// Perform line search, storing new state vector in x
	ls_status = line_search(optimizable, x, direction,
			       test_x, step_size, gradient, state_up_to_date,
			       cg_curvature_coeff_, bound_step_size);
	if (ls_status == MINIMIZER_STATUS_BOUND_REACHED) {
	  bound_status(i_nearest_bound) = i_bound_type;
	  do_restart = true;
	  ls_status = MINIMIZER_STATUS_SUCCESS;
	}
      }
      else {
	// Perform line search, storing new state vector in x
	ls_status = line_search(optimizable, x, direction,
				test_x, step_size, gradient, state_up_to_date,
				cg_curvature_coeff_);
      }

      if (ls_status == MINIMIZER_STATUS_SUCCESS) {
	// Successfully minimized along search direction: continue to
	// next iteration
	status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
      }
      else if (iteration_at_last_restart != n_iterations_) {
	// Line search either made no progress or encountered a
	// non-finite cost function or gradient, and this was not a
	// restart; try restarting once
	do_restart = true;
	status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
      }
      else {
	// Unrecoverable failure in line-search: return status to
	// calling function
	status_ = ls_status;
      }

      // Better convergence if first step size on next line search is
      // larger than the actual step size on the last line search
      step_size *= 2.0;

      ++n_iterations_;
      if (status_ == MINIMIZER_STATUS_NOT_YET_CONVERGED
	  && n_iterations_ >= max_iterations_) {
	status_ = MINIMIZER_STATUS_MAX_ITERATIONS_REACHED;
      }

      // End of main loop: if status_ is anything other than
      // MINIMIZER_STATUS_NOT_YET_CONVERGED then no more iterations
      // are performed
    }
     
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
