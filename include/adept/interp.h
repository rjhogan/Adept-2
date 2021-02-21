/* interp.h -- 1D interpolation


    Copyright (C) 2015-2021 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptInterp_H
#define AdeptInterp_H

#include <adept/Array.h>

namespace adept {

  enum ExtrapolatePolicy {
    ADEPT_EXTRAPOLATE_LINEAR = 0,
    ADEPT_EXTRAPOLATE_CLAMP,
  };

  template <typename XType, typename YType, bool YIsActive, typename XiType, int YDims>
  Array<YDims,YType,YIsActive>
  interp(const Array<1,XType,false>& x,
	 const Array<YDims,YType,YIsActive>& y,
	 const Array<1,XiType,false>& xi,
	 ExtrapolatePolicy ep = ADEPT_EXTRAPOLATE_LINEAR) {
    ExpressionSize<YDims> ans_dims = y.dimensions();
    ans_dims[0] = xi.size();
    Array<YDims,YType,YIsActive> ans(ans_dims);
    if (x.size() != y.size(0)) {
      throw(size_mismatch("Interpolation vector x must have same length of first dimension of y in interp"));
    }
    else if (x.size() == 0) {
      throw(size_mismatch("Interpolation from empty vectors"));
    }
    else if (x.size() == 1) {
      // Input arrays are at a single point: copy this point into all
      // output points regardless of their x coordinate
      for (int ii = 0; ii < xi.size(); ++ii) {
	ans[ii] = y[0];
      }
    }

    if (x(0) < x(1)) {
      // Normal ordering
      for (Index i = 0; i < xi.size(); i++) {
	Real xii = xi(i);
	Index jmin = 0;
	Index jmax = x.size()-1;
	if (xii <= x(0)) {
	  if (ep == ADEPT_EXTRAPOLATE_LINEAR) {
	    // Extrapolate leftwards
	    jmax = 1;
	  }
	  else {
	    // Clamp at first value
	    ans[i] = y[0];
	    continue;
	  }
	}
	else if (xii >= x(jmax)) {
	  if (ep == ADEPT_EXTRAPOLATE_LINEAR) {
	    // Extrapolate rightwards
	    jmin = jmax-1;
	  }
	  else {
	    // Clamp at final value
	    ans[i] = y[jmax];
	    continue;
	  }
	}
	else {
	  // xii lies within x
	  // Find pair in which xi sits
	  while (jmax > jmin+1) {
	    Index jmid = jmin + (jmax-jmin)/2;
	    if (xii > x(jmid)) {
	      jmin = jmid;
	    }
	    else {
	      jmax = jmid;
	    }
	  }
	}
	// Found value: linearly interpolate
	ans[i] = ((xii-x(jmin))*y[jmax] + (x(jmax)-xii)*y[jmin])
	  / (x(jmax)-x(jmin));
      }
    }
    else {
      // Reverse ordering
      for (Index i = 0; i < xi.size(); i++) {
	Real xii = xi(i);
	Index jmin = 0;
	Index jmax = x.size()-1;
	if (xii >= x(0)) {
	  if (ep == ADEPT_EXTRAPOLATE_LINEAR) {
	    // Extrapolate leftwards
	    jmax = 1;
	  }
	  else {
	    // Clamp at first value
	    ans[i] = y[0];
	    continue;
	  }
	}
	else if (xii <= x(jmax)) {
	  if (ep == ADEPT_EXTRAPOLATE_LINEAR) {
	    // Extrapolate rightwards
	    jmin = jmax-1;
	  }
	  else {
	    // Clamp at last value
	    ans[i] = y[jmax];
	    continue;
	  }
	}
	else {
	  // xii lies within x
	  // Find pair in which xi sits
	  while (jmax > jmin+1) {
	    Index jmid = jmin + (jmax-jmin)/2;
	    if (xii < x(jmid)) {
	      jmin = jmid;
	    }
	    else {
	      jmax = jmid;
	    }
	  }
	}
	// Found value: linearly interpolate
	ans[i] = ((xii-x(jmin))*y[jmax] + (x(jmax)-xii)*y[jmin])
	  / (x(jmax)-x(jmin));
      }
    }
    return ans;
  }

  template <typename XType, typename YType, typename XiType,
	    class X, class Y, class Xi>
  Array<Y::rank,YType,Y::is_active>
  interp(const Expression<XType,X>& x,
	 const Expression<YType,Y>& y,
	 const Expression<XiType,Xi>& xi,
	 ExtrapolatePolicy ep = ADEPT_EXTRAPOLATE_LINEAR) {
    const Array<1,XType,false> x2(x.cast());
    const Array<Y::rank,YType,Y::is_active> y2(y.cast());
    const Array<1,XiType,false> xi2(xi.cast());
    return interp(x2, y2, xi2, ep);
  }


  template <typename XType, typename YType, bool YIsActive, typename XiType>
  Array<1,YType,YIsActive>
  log_interp(const Array<1,XType,false>& x,
	 const Array<1,YType,YIsActive>& y,
	 const Array<1,XiType,false>& xi) {
    using std::exp;
    using std::log;

    int length = xi.size();
    Array<1,YType,YIsActive> ans(length);
    if (x.size() != y.size()) {
      throw(size_mismatch("Interpolation vectors must be the same length in log_interp"));
    }

    if (x(0) < x(1)) {
      // Normal ordering
      for (Index i = 0; i < length; i++) {
	Real xii = xi(i);
	Index jmin = 0;
	Index jmax = x.size()-1;
	if (xii <= x(0)) {
	  // Extrapolate leftwards
	  jmax = 1;
	}
	else if (xii >= x(jmax)) {
	  // Extrapolate rightwards
	  jmin = jmax-1;
	}
	else {
	  // xii lies within x
	  // Find pair in which xi sits
	  while (jmax > jmin+1) {
	    Index jmid = jmin + (jmax-jmin)/2;
	    if (xii > x(jmid)) {
	      jmin = jmid;
	    }
	    else {
	      jmax = jmid;
	    }
	  }
	}
	// Found value: logarithmically interpolate
	if (y(jmax) > 0.0 && y(jmin) > 0.0) {
	  YType log_y_jmax = log(y(jmax));
	  YType log_y_jmin = log(y(jmin));
	  ans(i) = exp(((xii-x(jmin))*log_y_jmax + (x(jmax)-xii)*log_y_jmin)
		       / (x(jmax)-x(jmin)));
	}
	else {
	  // Interpolate linearly since one or both values is zero
	  ans(i) = ((xii-x(jmin))*y(jmax) + (x(jmax)-xii)*y(jmin))
	    / (x(jmax)-x(jmin));
	}
      }
    }
    else {
      // Reverse ordering
      for (Index i = 0; i < length; i++) {
	Real xii = xi(i);
	Index jmin = 0;
	Index jmax = x.size()-1;
	if (xii >= x(0)) {
	  // Extrapolate leftwards
	  jmax = 1;
	}
	else if (xii <= x(jmax)) {
	  // Extrapolate rightwards
	  jmin = jmax-1;
	}
	else {
	  // xii lies within x
	  // Find pair in which xi sits
	  while (jmax > jmin+1) {
	    Index jmid = jmin + (jmax-jmin)/2;
	    if (xii < x(jmid)) {
	      jmin = jmid;
	    }
	    else {
	      jmax = jmid;
	    }
	  }
	}
	// Found value: logarithmically interpolate
	if (y(jmax) > 0.0 && y(jmin) > 0.0) {
	  YType log_y_jmax = log(y(jmax));
	  YType log_y_jmin = log(y(jmin));
	  ans(i) = exp(((xii-x(jmin))*log_y_jmax + (x(jmax)-xii)*log_y_jmin)
		       / (x(jmax)-x(jmin)));
	}
	else {
	  // Interpolate linearly since one or both values is zero
	  ans(i) = ((xii-x(jmin))*y(jmax) + (x(jmax)-xii)*y(jmin))
	    / (x(jmax)-x(jmin));
	}
      }
    }
    return ans;
  }
  
} // End namespace adept

#endif
