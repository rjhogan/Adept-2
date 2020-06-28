/* Minimizer.h -- class for minimizing the cost function of an optimizable object

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/


#include <adept/Minimizer.h>
#include <adept/exception.h>

namespace adept {

  const char*
  minimizer_status_string(MinimizerStatus status)
  {
    switch (status) {
    case MINIMIZER_STATUS_SUCCESS:
      return "Converged";
      break;
    case MINIMIZER_STATUS_EMPTY_STATE:
      return "Empty state vector, no minimization performed";
      break;
    case MINIMIZER_STATUS_MAX_ITERATIONS_REACHED:
      return "Maximum iterations reached";
      break;
    case MINIMIZER_STATUS_FAILED_TO_CONVERGE:
      return "Failed to converge";
      break;
    case MINIMIZER_STATUS_INVALID_COST_FUNCTION:
      return "Non-finite cost function";
      break;
    case MINIMIZER_STATUS_INVALID_GRADIENT:
      return "Non-finite gradient";
      break;
    case MINIMIZER_STATUS_INVALID_BOUNDS:
      return "Invalid bounds for bounded minimization";
      break;
    case MINIMIZER_STATUS_NOT_YET_CONVERGED:
      return "Minimization still in progress";
      break;
    default:
      return "Status unrecognized";
    }
  }

  MinimizerStatus
  Minimizer::minimize(Optimizable1& optimizable, Vector x)
  {
    if (minimizer_algorithm_order(algorithm_) > 1) {
      throw optimization_exception("2nd-order minimization algorithm requires optimizable that can provide 2nd derivatives");
    }
    else if (algorithm_ == MINIMIZER_ALGORITHM_LIMITED_MEMORY_BFGS) {
      return minimize_limited_memory_bfgs(optimizable, x,
					  max_iterations_,
					  converged_gradient_norm_);
    }
    else {
      throw optimization_exception("Minimization algorithm not recognized");
    }
  }

  MinimizerStatus
  Minimizer::minimize(Optimizable2& optimizable, Vector x)
  {
    if (algorithm_ == MINIMIZER_ALGORITHM_LIMITED_MEMORY_BFGS) {
      return minimize_limited_memory_bfgs(optimizable, x,
					  max_iterations_,
					  converged_gradient_norm_,
					  max_step_size_);
    }
    else if (algorithm_ == MINIMIZER_ALGORITHM_LEVENBERG_MARQUARDT) {
      if (is_bounded_ && optimizable.have_bounds()) {
	return minimize_levenberg_marquardt_bounded(optimizable, x,
					    optimizable.lower_bounds(),
					    optimizable.upper_bounds(),
					    max_iterations_,
					    converged_gradient_norm_,
					    max_step_size_);
      }
      else {
	return minimize_levenberg_marquardt(optimizable, x,
					    max_iterations_,
					    converged_gradient_norm_,
					    max_step_size_);
      }
    }
    else {
      throw optimization_exception("Minimization algorithm not recognized");
    }
  }

};
