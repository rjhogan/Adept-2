/* Minimizer.h -- class for minimizing the cost function of an optimizable object

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptMinimizer_H
#define AdeptMinimizer_H 1

#include <adept/Optimizable.h>

namespace adept {

  enum MinimizerAlgorithm {
    MINIMIZER_ALGORITHM_LIMITED_MEMORY_BFGS = 0,
    MINIMIZER_ALGORITHM_LEVENBERG_MARQUARDT,
    MINIMIZER_ALGORITHM_NUMBER_AVAILABLE
  };

  enum MinimizerStatus {
    MINIMIZER_STATUS_SUCCESS = 0,
    MINIMIZER_STATUS_EMPTY_STATE,
    MINIMIZER_STATUS_MAX_ITERATIONS_REACHED,
    MINIMIZER_STATUS_FAILED_TO_CONVERGE,
    MINIMIZER_STATUS_INVALID_COST_FUNCTION,
    MINIMIZER_STATUS_INVALID_GRADIENT,
    MINIMIZER_STATUS_INVALID_BOUNDS,
    MINIMIZER_STATUS_NUMBER_AVAILABLE,
    MINIMIZER_STATUS_NOT_YET_CONVERGED
  };

  // Return a C string describing the minimizer status
  const char* minimizer_status_string(MinimizerStatus status);

  // Return the order of a minimization algorithm: 0 indicates only
  // the cost function is required, 1 indicates the first derivative
  // is required, 2 indicates the second derivative is required, while
  // -1 indicates that the algorithm is not recognized.
  inline int minimizer_algorithm_order(MinimizerAlgorithm algo) {
    switch (algo) {
    case MINIMIZER_ALGORITHM_LIMITED_MEMORY_BFGS:
      return 1;
      break;
    case MINIMIZER_ALGORITHM_LEVENBERG_MARQUARDT:
      return 2;
      break;
    default:
      return -1;
    }
  }

  // Direct access to specific minimization algorithms
  MinimizerStatus
  minimize_limited_memory_bfgs(Optimizable1& optimizable, Vector x,
			       int max_iterations, Real converged_gradient_norm,
			       Real max_step_size = -1.0);
  MinimizerStatus
  minimize_levenberg_marquardt(Optimizable2& optimizable, Vector x,
			       int max_iterations, Real converged_gradient_norm,
			       Real max_step_size = -1.0);
  MinimizerStatus
  minimize_levenberg_marquardt_bounded(Optimizable2& optimizable, Vector x,
				       const Vector& min_x,
				       const Vector& max_x,
				       int max_iterations,
				       Real converged_gradient_norm,
				       Real max_step_size = -1.0);

  // A class that can minimize a function using various algorithms
  class Minimizer {

  public:

    Minimizer(MinimizerAlgorithm algo) : algorithm_(algo) { }

    void set_algorithm(MinimizerAlgorithm algo) {
      algorithm_ = algo;
    }

    // Unconstrained minimization
    MinimizerStatus minimize(Optimizable1& optimizable, Vector x);
    MinimizerStatus minimize(Optimizable2& optimizable, Vector x);
    // Constrained minimization
    //    MinimizerStatus minimize(Optimizable1& optimizable, Vector x,
    //			     const Vector& x_lower, const Vector& x_upper);
    MinimizerStatus minimize(Optimizable2& optimizable, Vector x,
			     const Vector& x_lower, const Vector& x_upper);

    void max_iterations(int mi) { max_iterations_ = mi; }
    void converged_gradient_norm(Real cgn) { converged_gradient_norm_ = cgn; }
    void max_step_size(Real mss) { max_step_size_ = mss; }

  private:
    // Minimizer type
    MinimizerAlgorithm algorithm_;

    // Variables controling the behaviour of the minimizer
    int max_iterations_ = 100; // <=0 means no limit
    adept::Real max_step_size_ = -1.0;
    adept::Real converged_gradient_norm_ = 0.1;
  };


  


};

#endif
