/* vector_utilities.h -- Vector utility functions

    Copyright (C) 2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptVectorUtilities_H
#define AdeptVectorUtilities_H

#include <adept/Array.h>

namespace adept {

  Array<1,Real,false> linspace(Real x1, Real x2, Index n);

}

#endif
