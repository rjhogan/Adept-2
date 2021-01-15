/* Minimizer.h -- class for minimizing the cost function of an optimizable object

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <cctype>

#include <adept/Minimizer.h>
#include <adept/exception.h>

namespace adept {

  // List of the names of available minimizer algorithms
  static const char* minimizer_algorithm_names_[]
    = {"L-BFGS",
       "Conjugate-Gradient",
       "Conjugate-Gradient-FR",
       "Levenberg",
       "Levenberg-Marquardt"};

  // Lower-case versions of the list above
  static const char* minimizer_algorithm_lower_names_[]
    = {"l-bfgs",
       "conjugate-gradient",
       "conjugate-gradient-fr",
       "levenberg",
       "levenberg-marquardt"};

  // Convert to lower case, and convert spaces and underscores to
  // hyphens. This function is used to do a case-insensitive
  // string-based selection of the minimizer algorithm to use.
  static void to_lower_in_place(std::string& str) {
    for (std::string::size_type istr = 0; istr < str.size(); ++istr) {
      str[istr] = std::tolower(str[istr]);
      if (str[istr] == ' ' || str[istr] == '_') {
	str[istr] = '-';
      }
    }
  }

  // Return a C string describing the minimizer status
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
    case MINIMIZER_STATUS_DIRECTION_UPHILL:
      return "Search direction points uphill";
      break;
    case MINIMIZER_STATUS_BOUND_REACHED:
      return "Bound reached"; // Should not be returned from a minimize function
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

  // Case-insensitive setting of the miminization algorithm given its
  // name
  void
  Minimizer::set_algorithm(const std::string& algo) {
    std::string algo_lower = algo;
    to_lower_in_place(algo_lower);

    std::cout << "Checking \"" << algo_lower << "\"\n";

    for (int ialgo = 0;
	 ialgo < static_cast<int>(MINIMIZER_ALGORITHM_NUMBER_AVAILABLE);
	 ++ialgo) {
      if (algo_lower == minimizer_algorithm_lower_names_[ialgo]) {
	set_algorithm(static_cast<MinimizerAlgorithm>(ialgo));
	return;
      }
    }
    throw optimization_exception("Algorithm name not understood");
  }

  std::string
  Minimizer::algorithm_name() {
    int ialgo = static_cast<MinimizerAlgorithm>(algorithm_);
    if (ialgo >= 0 && ialgo < MINIMIZER_ALGORITHM_NUMBER_AVAILABLE) {
      return minimizer_algorithm_names_[ialgo];
    }
    else {
      return "Unknown";
    }
  }

  // Unconstrained minimization
  MinimizerStatus
  Minimizer::minimize(Optimizable& optimizable, Vector x)
  {
    if (minimizer_algorithm_order(algorithm_) > 1
	&& !optimizable.provides_derivative(2)) {
      throw optimization_exception("2nd-order minimization algorithm requires optimizable that can provide 2nd derivatives");
    }
    else if (algorithm_ == MINIMIZER_ALGORITHM_LIMITED_MEMORY_BFGS) {
      return minimize_limited_memory_bfgs(optimizable, x);
    }
    else if (algorithm_ == MINIMIZER_ALGORITHM_CONJUGATE_GRADIENT) {
      return minimize_conjugate_gradient(optimizable, x);
    }
    else if (algorithm_ == MINIMIZER_ALGORITHM_CONJUGATE_GRADIENT_FR) {
      return minimize_conjugate_gradient(optimizable, x, true);
    }
    else if (algorithm_ == MINIMIZER_ALGORITHM_LEVENBERG) {
      return minimize_levenberg_marquardt(optimizable, x, true);
    }
    else if (algorithm_ == MINIMIZER_ALGORITHM_LEVENBERG_MARQUARDT) {
      return minimize_levenberg_marquardt(optimizable, x, false);
    }
    else {
      throw optimization_exception("Minimization algorithm not recognized");
    }
  }

  // Constrained minimization
  MinimizerStatus
  Minimizer::minimize(Optimizable& optimizable, Vector x,
		      const Vector& x_lower, const Vector& x_upper)
  {
    if (minimizer_algorithm_order(algorithm_) > 1
	&& !optimizable.provides_derivative(2)) {
      throw optimization_exception("2nd-order minimization algorithm requires optimizable that can provide 2nd derivatives");
    }
    if (algorithm_ == MINIMIZER_ALGORITHM_LIMITED_MEMORY_BFGS) {
      return minimize_limited_memory_bfgs_bounded(optimizable, x,
						  x_lower, x_upper);
    }
    else if (algorithm_ == MINIMIZER_ALGORITHM_CONJUGATE_GRADIENT) {
      return minimize_conjugate_gradient_bounded(optimizable, x,
						 x_lower, x_upper);
    }
    else if (algorithm_ == MINIMIZER_ALGORITHM_CONJUGATE_GRADIENT_FR) {
      return minimize_conjugate_gradient_bounded(optimizable, x,
						 x_lower, x_upper, true);
    }
    if (algorithm_ == MINIMIZER_ALGORITHM_LEVENBERG) {
      return minimize_levenberg_marquardt_bounded(optimizable, x,
						  x_lower, x_upper, true);
    }
    if (algorithm_ == MINIMIZER_ALGORITHM_LEVENBERG_MARQUARDT) {
      return minimize_levenberg_marquardt_bounded(optimizable, x,
						  x_lower, x_upper, false);
    }
    else {
      throw optimization_exception("Constrained minimization algorithm not recognized");
    }
  }

};
