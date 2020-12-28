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
    MINIMIZER_ALGORITHM_CONJUGATE_GRADIENT,
    MINIMIZER_ALGORITHM_LEVENBERG,
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
    case MINIMIZER_ALGORITHM_CONJUGATE_GRADIENT:
      return 1;
      break;
    case MINIMIZER_ALGORITHM_LEVENBERG:
    case MINIMIZER_ALGORITHM_LEVENBERG_MARQUARDT:
      return 2;
      break;
    default:
      return -1;
    }
  }

  // Convenience function for initializing vectors representing the
  // lower and upper bounds on state variables
  inline void minimizer_initialize_bounds(int nx, adept::Vector& x_lower,
					  adept::Vector& x_upper) {
    x_lower.resize(nx);
    x_upper.resize(nx);
    x_lower = -std::numeric_limits<Real>::max();
    x_upper =  std::numeric_limits<Real>::max();
  }

  // A class that can minimize a function using various algorithms
  class Minimizer {

  public:

    // Tedious C++98 initializations
    Minimizer(MinimizerAlgorithm algo)
      : algorithm_(algo),
	max_iterations_(100), // <=0 means no limit
	max_step_size_(-1.0),
	converged_gradient_norm_(0.1),
	ensure_updated_state_(-1),
	levenberg_damping_min_(1.0/128.0),
	levenberg_damping_max_(100000.0),
	levenberg_damping_multiplier_(2.0),
	levenberg_damping_divider_(5.0),
	levenberg_damping_start_(0.0),
	levenberg_damping_restart_(1.0/4.0) { }

    Minimizer(const std::string& algo)
      : max_iterations_(100), // <=0 means no limit
	max_step_size_(-1.0),
	converged_gradient_norm_(0.1),
	ensure_updated_state_(-1),
	levenberg_damping_min_(1.0/128.0),
	levenberg_damping_max_(100000.0),
	levenberg_damping_multiplier_(2.0),
	levenberg_damping_divider_(5.0),
	levenberg_damping_start_(0.0),
	levenberg_damping_restart_(1.0/4.0)
    { set_algorithm(algo); }

    // Unconstrained minimization
    MinimizerStatus minimize(Optimizable& optimizable, Vector x);
    // Constrained minimization
    MinimizerStatus minimize(Optimizable& optimizable, Vector x,
			     const Vector& x_lower, const Vector& x_upper);

    // Functions to set parameters defining the general behaviour of
    // minimization algorithms
    void set_algorithm(MinimizerAlgorithm algo) { algorithm_ = algo; }
    void set_algorithm(const std::string& algo);
    void set_max_iterations(int mi)             { max_iterations_ = mi; }
    void set_converged_gradient_norm(Real cgn)  { converged_gradient_norm_ = cgn; }
    void set_max_step_size(Real mss)            { max_step_size_ = mss; }
    // Ensure that the last call to compute the cost function uses the
    // "solution" state vector returned by minimize. This ensures that
    // any variables in user classes that inherit from Optimizable are
    // up to date with the returned state vector. The "order" argument
    // indicates which the order of derivatives required (provided
    // they are supported by the minimizing algorithm):
    // 0=cost_function, 1=cost_function_gradient,
    // 2=cost_function_gradient_hessian.
    void ensure_updated_state(int order = 2)    { ensure_updated_state_ = order; }
    
    // Return parameters defining behaviour of minimization algorithms
    MinimizerAlgorithm algorithm() { return algorithm_; }
    std::string algorithm_name();
    int max_iterations() { return max_iterations_; }
    Real converged_gradient_norm() { return converged_gradient_norm_; }      

    // Functions to set parameters defining the behaviour of the
    // Levenberg and Levenberg-Marquardt algorithm
    void set_levenberg_damping_limits(Real damp_min, Real damp_max);
    void set_levenberg_damping_start(Real damp_start);
    void set_levenberg_damping_restart(Real damp_restart);
    void set_levenberg_damping_multiplier(Real damp_multiply, Real damp_divide);

    // Query aspects of the algorithm progress after it has completed
    int  n_iterations()        const { return n_iterations_; }
    int  n_samples()           const { return n_samples_; }
    Real cost_function()       const { return cost_function_; }
    Real gradient_norm()       const { return gradient_norm_; }
    Real start_cost_function() const { return start_cost_function_; }
    MinimizerStatus status()   const { return status_; }

  protected:

    // Specific minimization algorithms

    MinimizerStatus
    minimize_limited_memory_bfgs(Optimizable& optimizable, Vector x);

    MinimizerStatus
    minimize_conjugate_gradient(Optimizable& optimizable, Vector x);

    // Call the Levenberg-Marquardt algorithm; if use_additive_damping
    // is true then the Levenberg algorithm is used instead
    MinimizerStatus
    minimize_levenberg_marquardt(Optimizable& optimizable, Vector x,
				 bool use_additive_damping = false);
    MinimizerStatus
    minimize_levenberg_marquardt_bounded(Optimizable& optimizable, Vector x,
					 const Vector& min_x,
					 const Vector& max_x,
					 bool use_additive_damping = false);

    // DATA

    // Minimizer type
    MinimizerAlgorithm algorithm_;

    // Variables controling the general behaviour of the minimizer,
    // used by all gradient-based algorithms
    int max_iterations_; // <=0 means no limit
    Real max_step_size_;
    Real converged_gradient_norm_;
    int ensure_updated_state_;

    // Variables controling the specific behaviour of the
    // Levenberg-Marquardt minimizer
    Real levenberg_damping_min_;
    Real levenberg_damping_max_;
    Real levenberg_damping_multiplier_;
    Real levenberg_damping_divider_;
    Real levenberg_damping_start_;
    Real levenberg_damping_restart_;

    // Variables set during the running of an algorithm and available
    // to the user afterwards

    // Number of iterations that successfully reduced the cost function
    int n_iterations_;

    // Number of calculations of the cost function
    int n_samples_;

    Real start_cost_function_;
    Real cost_function_;
    Real gradient_norm_;
    MinimizerStatus status_;
  };

  // Implement inline member functions

  // Functions to set parameters defining the behaviour of the
  // Levenberg and Levenberg-Marquardt algorithm
  inline void 
  Minimizer::set_levenberg_damping_limits(Real damp_min, Real damp_max) {
    if (damp_min <= 0.0) {
      throw optimization_exception("Minimum damping factor in Levenberg-Marquardt algorithm must be positive");
    }
    else if (damp_max <= damp_min) {
      throw optimization_exception("Maximum damping factor must be greater than minimum in Levenberg-Marquardt algorithm");
    }
    levenberg_damping_min_ = damp_min;
    levenberg_damping_max_ = damp_max;
  }
  inline void 
  Minimizer::set_levenberg_damping_start(Real damp_start) {
    if (damp_start < 0.0) {
      throw optimization_exception("Start damping factor in Levenberg-Marquardt algorithm must be positive or zero");
    }
    levenberg_damping_start_ = damp_start;
  }
  inline void 
  Minimizer::set_levenberg_damping_restart(Real damp_restart) {
    if (damp_restart <= 0.0) {
      throw optimization_exception("Restart damping factor in Levenberg-Marquardt algorithm must be positive");
    }
    levenberg_damping_restart_ = damp_restart;
  }
  inline void 
  Minimizer::set_levenberg_damping_multiplier(Real damp_multiply,
					      Real damp_divide) {
    if (damp_multiply <= 1.0 || damp_divide <= 1.0) {
      throw optimization_exception("Damping multipliers in Levenberg-Marquardt algorithm must be greater than one");
    }
    levenberg_damping_multiplier_ = damp_multiply;
    levenberg_damping_divider_    = damp_divide;
  }

};

#endif
