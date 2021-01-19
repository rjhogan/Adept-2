/* minimize_limited_memory_bfgs.cpp -- Minimize function using Limited-Memory BFGS algorithm

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <limits>

#include <adept/Minimizer.h>

namespace adept {

  // Structure for storing data from previous iterations used by
  // L-BFGS minimization algorithm
  class LbfgsData {

  public:
    LbfgsData(int nx, int ni)
      : nx_(nx), ni_(ni), iteration_(0) {
      x_diff_.resize(ni,nx);
      gradient_diff_.resize(ni,nx);
      rho_.resize(ni);
      alpha_.resize(ni);
      gamma_.resize(ni);
    }

    // Return false if the dot product of x_diff and gradient_diff is
    // zero, true otherwise
    void store(int iter, const Vector& x_diff, const Vector& gradient_diff) {
      int index = (iter-1) % ni_;
      x_diff_[index] = x_diff;
      gradient_diff_[index] = gradient_diff;
      Real dp = dot_product(x_diff, gradient_diff);
      if (std::fabs(dp) > 10.0*std::numeric_limits<Real>::min()) {
	rho_[index] = 1.0 / dp;
      }
      else if (dp >= 0.0) {
	rho_[index] = 1.0 / std::max(dp, 10.0*std::numeric_limits<Real>::min());
      }
      else {
	rho_[index] = 1.0 / std::min(dp, -10.0*std::numeric_limits<Real>::min());
      }
    }

    // Return read-only vectors containing the differences between
    // state vectors and gradients at sequential iterations, by
    // slicing off the appropriate row of the matrix
    Vector x_diff(int iter) {
      return x_diff_[iter % ni_];
    };
    Vector gradient_diff(int iter) {
      return gradient_diff_[iter % ni_];
    };

    Real& alpha(int iter) { return alpha_[iter % ni_]; }
    Real rho(int iter) const { return rho_[iter % ni_]; }
    Real gamma(int iter) const { return gamma_[iter % ni_]; }

  private:
    // Data
    int nx_; // Number of state variables
    int ni_; // Number of iterations to store
    int iteration_; // Current iteration
    Matrix x_diff_;
    Matrix gradient_diff_;
    Vector rho_;
    Vector alpha_;
    Vector gamma_;
  };


  // Minimize the cost function embodied in "optimizable" using the
  // Limited-Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
  // algorithm, where "x" is the initial state vector and also where
  // the solution is stored.
  MinimizerStatus
  Minimizer::minimize_limited_memory_bfgs(Optimizable& optimizable, Vector x)
  {

    int nx = x.size();

    // Initial values
    n_iterations_ = 0;
    n_samples_ = 0;
    status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
    cost_function_ = std::numeric_limits<Real>::infinity();

    Vector previous_x(nx);
    Vector gradient(nx);
    Vector previous_gradient(nx);
    Vector direction(nx);
    Vector test_x(nx); // Used by the line search only

    // Previous states needed by the L-BFGS algorithm
    int n_states = std::min(nx, lbfgs_n_states_);
    LbfgsData data(nx, n_states);

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

      // Store state and gradient differences
      if (n_iterations_ > 0) {
	data.store(n_iterations_, x-previous_x, gradient-previous_gradient);
      }

      // Find search direction: see page 779 of Nocedal (1980):
      // Updating quasi-Newton matrices with limited
      // storage. Mathematics of Computation, 35, 773-782.
      direction = gradient;
      if (n_iterations_ > 0) {

	for (int ii = n_iterations_-1;
	     ii >= std::max(0,n_iterations_-n_states);
	     --ii) {
	  data.alpha(ii) = data.rho(ii) 
	    * dot_product(data.x_diff(ii), direction);
	  direction -= data.alpha(ii) * data.gradient_diff(ii);
	}

	Real gamma = dot_product(x-previous_x, gradient-previous_gradient)
	  / std::max(10.0*std::numeric_limits<Real>::min(),
		     dot_product(gradient-previous_gradient, gradient-previous_gradient));
	direction *= gamma;

	for (int ii = std::max(0,n_iterations_-n_states);
	     ii < n_iterations_;
	     ++ii) {
	  Real beta = data.rho(ii) * dot_product(data.gradient_diff(ii), direction);
	  direction += data.x_diff(ii) * (data.alpha(ii)-beta);
	}

	direction = -direction;
      }
      else {
	direction = -gradient * (step_size / norm2(gradient));
      }

      // Store state and gradient
      previous_x = x;
      previous_gradient = gradient;

      // Perform line search, storing new state vector in x, and
      // returning MINIMIZER_STATUS_NOT_YET_CONVERGED on success
      Real curvature_coeff = lbfgs_curvature_coeff_;
      if (n_iterations_ < n_states) {
	// In the early iterations we require the line search to be
	// more accurate since the L-BFGS update will have fewer
	// states to make a good estimate of the minimum; interpolate
	// between the Conjugate Gradient and L-BFGS curvature
	// coefficients
	curvature_coeff = (cg_curvature_coeff_ * (n_states-n_iterations_)
			   + lbfgs_curvature_coeff_ * n_iterations_)
	  / n_states;
      }

      // Direction points to the best estimate of the actual location
      // of the minimum, so the step size is the norm of the direction
      // vector
      step_size = norm2(direction);
      MinimizerStatus ls_status
	= line_search(optimizable, x, direction,
		      test_x, step_size, gradient, state_up_to_date,
		      curvature_coeff);

      if (ls_status == MINIMIZER_STATUS_SUCCESS) {
	// Successfully minimized along search direction: continue to
	// next iteration
	status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
      }
      else {
	// Unrecoverable failure in line-search: return status to
	// calling function
	status_ = ls_status;
      }

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
  // Limited-Memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
  // algorithm, where "x" is the initial state vector and also where
  // the solution is stored.
  MinimizerStatus
  Minimizer::minimize_limited_memory_bfgs_bounded(Optimizable& optimizable, Vector x,
						  const Vector& min_x,
						  const Vector& max_x)
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

    Vector previous_x(nx);
    Vector gradient(nx);
    Vector previous_gradient(nx);
    Vector direction(nx);
    Vector test_x(nx); // Used by the line search only

    // Previous states needed by the L-BFGS algorithm
    int n_states = std::min(nx, lbfgs_n_states_);
    LbfgsData data(nx, n_states);

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

    // If we reach a bound we need to restart the L-BFGS storage, so
    // store the iteration at the last restart
    int iteration_last_restart = 0;

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
	iteration_last_restart = n_iterations_;
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

      // Store state and gradient differences
      if (n_iterations_ > iteration_last_restart) {
	data.store(n_iterations_, x-previous_x, gradient-previous_gradient);
      }

      // Find search direction: see page 779 of Nocedal (1980):
      // Updating quasi-Newton matrices with limited
      // storage. Mathematics of Computation, 35, 773-782.
      direction = gradient;
      if (n_iterations_ > iteration_last_restart) {

	for (int ii = n_iterations_-1;
	     ii >= std::max(iteration_last_restart,n_iterations_-n_states);
	     --ii) {
	  data.alpha(ii) = data.rho(ii) 
	    * dot_product(data.x_diff(ii), direction);
	  direction -= data.alpha(ii) * data.gradient_diff(ii);
	}

	Real gamma = dot_product(x-previous_x, gradient-previous_gradient)
	  / std::max(10.0*std::numeric_limits<Real>::min(),
		     dot_product(gradient-previous_gradient, gradient-previous_gradient));
	direction *= gamma;

	for (int ii = std::max(iteration_last_restart,n_iterations_-n_states);
	     ii < n_iterations_;
	     ++ii) {
	  Real beta = data.rho(ii) * dot_product(data.gradient_diff(ii), direction);
	  direction += data.x_diff(ii) * (data.alpha(ii)-beta);
	}

	direction = -direction;
      }
      else {
	// We are either at the first iteration or have restarted
	// having changed the bound dimensions: use steepest descent
	direction = -gradient * (step_size / norm2(gradient));
      }

      // Store state and gradient
      previous_x = x;
      previous_gradient = gradient;

      // Perform line search, storing new state vector in x, and
      // returning MINIMIZER_STATUS_NOT_YET_CONVERGED on success
      Real curvature_coeff = lbfgs_curvature_coeff_;
      int n_stored_iterations = n_iterations_ - iteration_last_restart;
      if (n_stored_iterations < n_states) {
	// In the early iterations we require the line search to be
	// more accurate since the L-BFGS update will have fewer
	// states to make a good estimate of the minimum; interpolate
	// between the Conjugate Gradient and L-BFGS curvature
	// coefficients
	curvature_coeff = (cg_curvature_coeff_ * (n_states-n_stored_iterations)
			   + lbfgs_curvature_coeff_ * n_stored_iterations)
	  / n_states;
      }

      // Direction points to the best estimate of the actual location
      // of the minimum, so the step size is the norm of the direction
      // vector
      step_size = norm2(direction);

      // Distance to the nearest bound
      Real dir_scaling = step_size;
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
				curvature_coeff, bound_step_size);
	if (ls_status == MINIMIZER_STATUS_BOUND_REACHED) {
	  bound_status(i_nearest_bound) = i_bound_type;
	  // Restart the L-BFGS storage
	  iteration_last_restart = n_iterations_+1;
	  ls_status = MINIMIZER_STATUS_SUCCESS;
	}
      }
      else {
	// Perform line search, storing new state vector in x
	ls_status = line_search(optimizable, x, direction,
				test_x, step_size, gradient, state_up_to_date,
				curvature_coeff);
      }

      if (ls_status == MINIMIZER_STATUS_SUCCESS) {
	// Successfully minimized along search direction: continue to
	// next iteration
	status_ = MINIMIZER_STATUS_NOT_YET_CONVERGED;
      }
      else {
	// Unrecoverable failure in line-search: return status to
	// calling function
	status_ = ls_status;
      }

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
