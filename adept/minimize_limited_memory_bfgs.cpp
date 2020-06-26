/* minimize_limited_memory_bfgs.h -- Minimize function using Limited-Memory BFGS algorithm

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <adept/Minimizer.h>

namespace adept {

  MinimizerStatus
  minimize_limited_memory_bfgs(Optimizable1& optimizable, Vector x,
			       int max_iterations, Real converged_gradient_norm,
			       Real max_step_size)
  {
    return MINIMIZER_STATUS_SUCCESS;
  }

};
