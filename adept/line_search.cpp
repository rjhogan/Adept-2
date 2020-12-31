/* line_search.cpp -- Approximate minimization of function along a line

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <limits>
#include <cmath>
#include <adept/Minimizer.h>

namespace adept {

  bool
  Minimizer::line_search_gradient_check(Optimizable& optimizable, Vector x, Vector direction,
					Vector test_x, Real& abs_step_size,
					Vector gradient, int& state_up_to_date,
					Real step_size, Real grad0, Real dir_scaling,
					Real& cf, Real& grad)
  {
    test_x = x + (step_size * dir_scaling) * direction;
    cf = optimizable.calc_cost_function_gradient(test_x, gradient);
    ++n_samples_;
    state_up_to_date = 1;
    grad = dot_product(direction, gradient) * dir_scaling;

    // Check Wolfe conditions
    if (cf <= cost_function_ + armijo_coeff_*step_size*grad0 // Armijo condition
	&& std::fabs(grad) <= -cg_curvature_coeff_*grad0) { // Curvature condition
      x = test_x;
      abs_step_size = step_size;
      cost_function_ = cf;
      return true;
    }
    else {
      return false;
    }
  }

  MinimizerStatus
  Minimizer::line_search(Optimizable& optimizable, Vector x, const Vector& direction,
			 Vector test_x, Real& abs_step_size,
			 Vector gradient, int& state_up_to_date)
  {
    Real dir_scaling = 1.0 / norm2(direction);

    // Numerical suffixes to variables indicate different locations
    // along the line:
    // 0 = initial point of line search, constant within this function
    // 1 = point at which gradient has been calculated (initially the same as 0)
    // 2 = test point
    // 3 = test point

    // Step sizes
    const Real ss0 = 0.0;
    Real ss1 = ss0;
    Real ss2 = abs_step_size;
    Real ss3;

    // Gradients in search direction
    Real grad0 = dot_product(direction, gradient) * dir_scaling;
    Real grad1 = grad0;
    Real grad2, grad3;

    // Cost function values
    Real cf0 = cost_function_;
    Real cf1 = cf0;
    Real cf2, cf3;

    typedef enum {
      TASK_LEAP_OF_FAITH,
      TASK_SECANT,
      TASK_WAVE
    } Task;
    
    Task task = TASK_LEAP_OF_FAITH;

    int iterations_remaining = cg_max_line_search_iterations_;

    if (grad0 >= 0.0) {
      return MINIMIZER_STATUS_DIRECTION_UPHILL;
    }

    while (iterations_remaining > 0) {

      //      std::cout << "  LS " << iterations_remaining << " " << static_cast<int>(task) << " " << ss1 << "-" << ss2 << " " << cf1 << "," << cf2 << " " << grad1 << "," << grad2 << "\n";

      if (task == TASK_LEAP_OF_FAITH) {

	if (line_search_gradient_check(optimizable, x, direction, test_x,
				       abs_step_size, gradient, state_up_to_date,
				       ss2, grad0, dir_scaling,
				       cf2, grad2)) {
	  return MINIMIZER_STATUS_NOT_YET_CONVERGED;
	}

	if (grad2 > 0.0) {
	  // Positive gradient -> bounded between points 1 and 2
	  task = TASK_SECANT;
	}
	else if (cf2 < cf1) {
	  // Reduced cost function but not yet bounded -> look further
	  // ahead
	  ss1 = ss2;
	  cf1 = cf2;
	  grad1 = grad2;
	  ss2 *= 4.0;	  
	  task = TASK_LEAP_OF_FAITH;
	}
	else {
	  // Cost function increased but gradient still negative so
	  // some kind of wave structure
	  task = TASK_WAVE;
	}

      }
      else if (task == TASK_SECANT) {
	// Minimizer of cubic function
	{
	  Real step_diff = ss2-ss1;
	  Real theta = (cf1-cf2) * 3.0 / step_diff + grad1 + grad2;
	  Real max_grad = std::fmax(std::fabs(theta),
				    std::fmax(std::fabs(grad1), std::fabs(grad2)));
	  Real scaled_theta = theta / max_grad;
	  Real gamma = max_grad * std::sqrt(scaled_theta*scaled_theta
				       - (grad1/max_grad) * (grad2/max_grad));
	  ss3 = ss1 + ((gamma - grad1 + theta) / (2.0*gamma + grad2 - grad1)) * step_diff;
	}
	// Bound the step size to be at least 5% away from each end
	ss3 = std::max(0.95*ss1+0.05*ss2,
		       std::min(0.05*ss1+0.95*ss2, ss3));

	if (line_search_gradient_check(optimizable, x, direction, test_x,
				       abs_step_size, gradient, state_up_to_date,
				       ss3, grad0, dir_scaling,
				       cf3, grad3)) {
	  return MINIMIZER_STATUS_NOT_YET_CONVERGED;
	}

	if (grad3 > 0.0) {
	  // Positive gradient -> bounded between points 1 and 3
	  ss2 = ss3;
	  cf2 = cf3;
	  grad2 = grad3;
	  task = TASK_SECANT;
	}
	else if (cf3 < cf1) {
	  // Reduced cost function, negative gradient
	  ss1 = ss3;
	  cf1 = cf3;
	  grad1 = grad3;
	  task = TASK_SECANT;
	}
	else {
	  // Increased cost function, negative gradient
	  ss2 = ss3;
	  cf2 = cf3;
	  grad2 = grad3;
	  task = TASK_WAVE;
	}	

      }
      else if (task == TASK_WAVE) {
	
	// Minimizer of cubic function
	{
	  Real step_diff = ss2-ss1;
	  Real theta = (cf1-cf2) * 3.0 / step_diff + grad1 + grad2;
	  Real max_grad = std::fmax(std::fabs(theta),
				    std::fmax(std::fabs(grad1), std::fabs(grad2)));
	  Real scaled_theta = theta / max_grad;
	  Real gamma = max_grad * std::sqrt(scaled_theta*scaled_theta
				       - (grad1/max_grad) * (grad2/max_grad));
	  ss3 = ss1 + ((gamma - grad1 + theta) / (2.0*gamma + grad2 - grad1)) * step_diff;
	}
	// Simple bisection:
	//	ss3 = 0.5 * (ss1 + ss2);
	// Bound the step size to be at least 5% away from each end
	ss3 = std::max(0.95*ss1+0.05*ss2,
		       std::min(0.05*ss1+0.95*ss2, ss3));

	if (line_search_gradient_check(optimizable, x, direction, test_x,
				       abs_step_size, gradient, state_up_to_date,
				       ss3, grad0, dir_scaling,
				       cf3, grad3)) {
	  return MINIMIZER_STATUS_NOT_YET_CONVERGED;
	}
	
	if (grad3 > 0.0) {
	  // Positive gradient -> bounded between points 1 and 3
	  ss2 = ss3;
	  cf2 = cf3;
	  grad2 = grad3;
	  task = TASK_SECANT;
	}
	else if (cf3 < cf1) {
	  // Reduced cost function, negative gradient
	  ss1 = ss3;
	  cf1 = cf3;
	  grad1 = grad3;
	  task = TASK_WAVE;
	}
	else {
	  // Increased cost function, negative gradient
	  ss2 = ss3;
	  cf2 = cf3;
	  grad2 = grad3;
	  task = TASK_WAVE;
	}

      }

      --iterations_remaining;
    }

    // Maximum iterations reached: check if cost function has been
    // reduced at all
    state_up_to_date = false;
    if (cf2 < cf1) {
      // Return value at point 2
      x += (ss2 * dir_scaling) * direction;
      cost_function_ = cf2;  
    }
    else if (cf1 < cf0) {
      // Return value at point 1
      x += (ss1 * dir_scaling) * direction;
      cost_function_ = cf1;  
    }
    else {
      return MINIMIZER_STATUS_FAILED_TO_CONVERGE;
    }

    return MINIMIZER_STATUS_NOT_YET_CONVERGED;

  }

}
