/* vector_utilities.cpp -- Vector utility functions

    Copyright (C) 2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <adept/vector_utilities.h>

namespace adept {

  Array<1,Real,false>
  linspace(Real x1, Real x2, Index n) {
    Array<1,Real,false> ans(n);
    if (n > 1) {
      for (Index i = 0; i < n; ++i) {
	ans(i) = x1 + (x2-x1)*i / static_cast<Real>(n-1);
      }
    }
    else if (n == 1 && x1 == x2) {
      ans(0) = x1;
      return ans;
    }
    else if (n == 1) {
      throw(invalid_operation("linspace(x1,x2,n) with n=1 only valid if x1=x2"));
    }
    return ans;
  }

}

