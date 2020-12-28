/* minimize_conjugate_gradient.cpp -- Minimize function using Conjugate Gradient algorithm

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <adept/Minimizer.h>

namespace adept {

  MinimizerStatus
  Minimizer::minimize_conjugate_gradient(Optimizable& optimizable, Vector x)
  {
    throw optimization_exception("Conjugate Gradient algorithm not yet implemented");
    //return MINIMIZER_STATUS_SUCCESS;
  }

};
