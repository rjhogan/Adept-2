/* interp.h -- 1D interpolation

    Copyright (C) 2015- European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptInterp_H
#define AdeptInterp_H

#include <adept/Array.h>

namespace adept {

  namespace internal {
    typedef unsigned int uint;
  };
  
  // The interpolation scheme and extrapolation behaviours are passed
  // in as one "options" argument with a bitwise OR. The lowest four
  // bits specify the extrapolation policy and the remaining bits the
  // interpolation scheme.
  static const internal::uint ADEPT_INTERPOLATE_LINEAR  = 0u; // Default
  static const internal::uint ADEPT_INTERPOLATE_NEAREST = (1u<<4);

  static const internal::uint ADEPT_EXTRAPOLATE_DEFAULT  = 0u;
  static const internal::uint ADEPT_EXTRAPOLATE_LINEAR   = 1u; // Default for linear interp 
  static const internal::uint ADEPT_EXTRAPOLATE_CLAMP    = 2u; // Default for nearest-neighbour
  // Return a constant for out-of-bounds inputs, or NaN if the
  // constant is not specified
  static const internal::uint ADEPT_EXTRAPOLATE_CONSTANT = 3u;

  // A bitwise AND of the "options" argument with one of the following
  // will extract the component associated with interpolation and
  // extrapolation
  namespace internal {
    static const internal::uint ADEPT_EXTRAPOLATE_MASK = 15; // Binary 1111
    static const internal::uint ADEPT_INTERPOLATE_MASK = ~ADEPT_EXTRAPOLATE_MASK;

    inline void extract_interp_extrap(uint options, uint& interp_scheme, uint& extrap_policy) {
      interp_scheme = options & ADEPT_INTERPOLATE_MASK;
      extrap_policy = options & ADEPT_EXTRAPOLATE_MASK;
      if (interp_scheme != ADEPT_INTERPOLATE_LINEAR
	  && interp_scheme != ADEPT_INTERPOLATE_NEAREST) {
	throw array_exception("Interpolation scheme not understood");
      }
      else if (extrap_policy > ADEPT_EXTRAPOLATE_CONSTANT) {
	throw array_exception("Extrapolation policy not understood");
      }
      else if (interp_scheme == ADEPT_INTERPOLATE_NEAREST
	       && extrap_policy == ADEPT_EXTRAPOLATE_LINEAR) {
	throw array_exception("Linear extrapolation not available with nearest-neighbour interpolation");
      }
      else if (extrap_policy == ADEPT_EXTRAPOLATE_DEFAULT) {
	if (interp_scheme == ADEPT_INTERPOLATE_LINEAR) {
	  extrap_policy = ADEPT_EXTRAPOLATE_LINEAR;
	}
	else {
	  extrap_policy = ADEPT_EXTRAPOLATE_CLAMP;
	}
      }
    }

    // The dimensions of an array containing the data to be
    // interpolated may be described either by a vector of real
    // numbers, or by a regular range; any other type will not
    // compile.  A regular range (which could be expressed by a
    // LinSpace object) has not yet been defined.
    template <typename T>
    struct InterpHelper {
      static const bool is_valid = false;
    };

    // Specialization for a vector of real numbers    
    template <typename XType>
    struct InterpHelper<Array<1,XType,false> > {
      static const bool is_valid = is_floating_point<XType>::value;
      template <typename XiType>
      static void interp_get_indices_weights(const Array<1,XType,false>& x,
				 const Array<1,XiType,false>& xi,
				 internal::uint interp_scheme,
				 internal::uint extrap_policy,
				 Array<1,Index>& ind0, Array<1,Real,false>& weight0,
				 Array<1,bool>& is_valid) {
	if (x(1) > x(0)) {
	  // Normal ordering; loop over points to be interpolated
	  for (Index i = 0; i < xi.size(); ++i) {
	    const XiType xii = xi(i);
	    if (xii >= x(0) && xii <= x(end)) {
	      // Point is in the range of the interpolated function
	      Index jj = 0;
	      while (jj < x.size()-2 && x(jj+1) < xii) {
		++jj;
	      }
	      ind0(i) = jj;
	      weight0(i) = (x(jj+1)-xii)/(x(jj+1)-x(jj));
	    }
	    else if (xii < x(0)) {
	      // Point is off the low end of the scale
	      ind0(i) = 0;
	      if (extrap_policy == ADEPT_EXTRAPOLATE_LINEAR) {
		weight0(i) = (x(1)-xii)/(x(1)-x(0));
	      }
	      else if (extrap_policy == ADEPT_EXTRAPOLATE_CLAMP) {
		weight0(i) = 1.0;
	      }
	      else {
		is_valid(i) = false;
	      }
	    }
	    else {
	      // Point is off the high end of the scale
	      ind0(i) = x.size()-2;
	      if (extrap_policy == ADEPT_EXTRAPOLATE_LINEAR) {
		weight0(i) = (x(end)-xii)/(x(end)-x(end-1));
	      }
	      else if (extrap_policy == ADEPT_EXTRAPOLATE_CLAMP) {
		weight0(i) = 0.0;
	      }
	      else {
		is_valid(i) = false;
	      }
	    }
	  }
	}
	else {
	  // Reverse ordering; loop over points to be interpolated
	  for (Index i = 0; i < xi.size(); ++i) {
	    const XiType xii = xi(i);
	    if (xii <= x(0) && xii >= x(end)) {
	      // Point is in the range of the interpolated function
	      Index jj = x.size()-2;
	      while (jj > 0 && x(jj) < xii) {
		--jj;
	      }
	      ind0(i) = jj;
	      weight0(i) = (x(jj+1)-xii)/(x(jj+1)-x(jj));
	    }
	    else if (xii > x(0)) {
	      // Point is off the scale (high in x, low in index)
	      ind0(i) = 0;
	      if (extrap_policy == ADEPT_EXTRAPOLATE_LINEAR) {
		weight0(i) = (x(1)-xii)/(x(1)-x(0));
	      }
	      else if (extrap_policy == ADEPT_EXTRAPOLATE_CLAMP) {
		weight0(i) = 1.0;
	      }
	      else {
		is_valid(i) = false;
	      }
	    }
	    else {
	      // Point is off the scale (low in x, high in index)
	      ind0(i) = x.size()-2;
	      if (extrap_policy == ADEPT_EXTRAPOLATE_LINEAR) {
		weight0(i) = (x(end)-xii)/(x(end)-x(end-1));
	      }
	      else if (extrap_policy == ADEPT_EXTRAPOLATE_CLAMP) {
		weight0(i) = 0.0;
	      }
	      else {
		is_valid(i) = false;
	      }
	    }	    
	  }
	}
	// Not very efficient implementation of nearest-neighbour
	// interpolation: round the weights from linear interpolation
	if (interp_scheme == ADEPT_INTERPOLATE_NEAREST) {
	  weight0 = round(weight0);
	}
      }
    };
  }
  
  // 1D interpolation: interp1(x,y,xi) interpolates to obtain values of
  // y (whose first dimension is at the points in vector x)
  // interpolated to the values in vector xi. If y has more than one
  // dimension then multiple values are interpolated for every point
  // in xi, and the returned array has a size equal to y except that
  // the first dimension is of the same length as xi. If the
  // extrapolate policy is specified and is ADEPT_EXTRAPOLATE_CLAMP
  // then values outside the range will be clampted at the first or
  // last point. If it is ADEPT_EXTRAPOLATE_CONSTANT then a constant
  // value will be used which can be specified as the final argument,
  // or is a signaling NaN by default.  Otherwise, linear
  // extrapolation is performed (the default). Note that x and xi must
  // be inactive variables, but y can be active in which case the
  // returned array will be too.
  template <typename XType, typename YType, bool YIsActive, typename XiType, int YDims>
  Array<YDims,YType,YIsActive>
  interp(const Array<1,XType,false>& x,
	 const Array<YDims,YType,YIsActive>& y,
	 const Array<1,XiType,false>& xi,
	 internal::uint options = ADEPT_INTERPOLATE_LINEAR | ADEPT_EXTRAPOLATE_DEFAULT,
	 YType extrap_value = std::numeric_limits<YType>::signaling_NaN()) {
    
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
      return ans;
    }

    internal::uint interp_scheme, extrap_policy;
    internal::extract_interp_extrap(options, interp_scheme, extrap_policy);
    
    if (x(0) < x(1)) {
      // Normal ordering
      for (Index i = 0; i < xi.size(); i++) {
	Real xii = xi(i);
	Index jmin = 0;
	Index jmax = x.size()-1;
	if (xii <= x(0)) {
	  if (extrap_policy == ADEPT_EXTRAPOLATE_LINEAR) {
	    // Extrapolate leftwards
	    jmax = 1;
	  }
	  else if (extrap_policy == ADEPT_EXTRAPOLATE_CLAMP) {
	    // Clamp at first value
	    ans[i] = y[0];
	    continue;
	  }
	  else {
	    ans[i] = extrap_value;
	    continue;
	  }
	}
	else if (xii >= x(jmax)) {
	  if (extrap_policy == ADEPT_EXTRAPOLATE_LINEAR) {
	    // Extrapolate rightwards
	    jmin = jmax-1;
	  }
	  else if (extrap_policy == ADEPT_EXTRAPOLATE_CLAMP) {
	    // Clamp at final value
	    ans[i] = y[jmax];
	    continue;
	  }
	  else {
	    ans[i] = extrap_value;
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
	if (interp_scheme == ADEPT_INTERPOLATE_LINEAR) {
	  // Found value: linearly interpolate. Note that we need
	  // square brackets here because ans and y may have more than
	  // one dimension in which case we want to slice them
	  // returning a lower dimensional array
	  ans[i] = ((xii-x(jmin))*y[jmax] + (x(jmax)-xii)*y[jmin])
	    / (x(jmax)-x(jmin));
	}
	else if (xii-x(jmin) > x(jmax)-xii) {
	  // Nearest neighbour is at next point
	  ans[i] = y[jmax];
	}
	else {
	  // Nearest neighbour is at previous point
	  ans[i] = y[jmin];
	}
      }
    }
    else {
      // Reverse ordering
      for (Index i = 0; i < xi.size(); i++) {
	Real xii = xi(i);
	Index jmin = 0;
	Index jmax = x.size()-1;
	if (xii >= x(0)) {
	  if (extrap_policy == ADEPT_EXTRAPOLATE_LINEAR) {
	    // Extrapolate leftwards
	    jmax = 1;
	  }
	  else if (extrap_policy == ADEPT_EXTRAPOLATE_CLAMP) {
	    // Clamp at first value
	    ans[i] = y[0];
	    continue;
	  }
	  else {
	    ans[i] = extrap_value;
	    continue;
	  }
	}
	else if (xii <= x(jmax)) {
	  if (extrap_policy == ADEPT_EXTRAPOLATE_LINEAR) {
	    // Extrapolate rightwards
	    jmin = jmax-1;
	  }
	  else if (extrap_policy == ADEPT_EXTRAPOLATE_CLAMP) {
	    // Clamp at last value
	    ans[i] = y[jmax];
	    continue;
	  }
	  else {
	    ans[i] = extrap_value;
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
	if (interp_scheme == ADEPT_INTERPOLATE_LINEAR) {
	  // Found value: linearly interpolate (all weights here are
	  // negative)
	  ans[i] = ((xii-x(jmin))*y[jmax] + (x(jmax)-xii)*y[jmin])
	    / (x(jmax)-x(jmin));
	}
	else if (xii-x(jmin) < x(jmax)-xii) {
	  // Nearest neighbour is at next point
	  ans[i] = y[jmax];
	}
	else {
	  // Nearest neighbour is at previous point
	  ans[i] = y[jmin];
	}
      }
    }
    return ans;
  }

  // Ensure that 1D interpolation works if expressions are provided
  // for any of the arguments; these are converted to temporary
  // arrays.
  template <typename XType, typename YType, typename XiType,
	    class X, class Y, class Xi>
  Array<Y::rank,YType,Y::is_active>
  interp(const Expression<XType,X>& x,
	 const Expression<YType,Y>& y,
	 const Expression<XiType,Xi>& xi,
	 internal::uint options = ADEPT_INTERPOLATE_LINEAR | ADEPT_EXTRAPOLATE_DEFAULT,
	 YType extrap_value = std::numeric_limits<YType>::signaling_NaN()) {
    const Array<1,XType,false> x2(x.cast());
    const Array<Y::rank,YType,Y::is_active> y2(y.cast());
    const Array<1,XiType,false> xi2(xi.cast());
    return interp(x2, y2, xi2, options, extrap_value);
  }

  // 1D logarithmic interpolation: interpolate log(Y) and then
  // exponentiate the result.
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

  // 2D interpolation: as 1D interpolation but with two vectors
  // describing the dimensions of the interpolation array and two
  // vectors providing points at which interpolated values are
  // required
  template <typename XType, typename YType,
	    int MDims, typename MType, bool MIsActive,
	    typename XiType, typename YiType>
  Array<MDims-1,MType,MIsActive>
  interp2d(const XType& x,
	   const YType& y,
	   const Array<MDims,MType,MIsActive>& M,
	   const Array<1,XiType,false>& xi,
	   const Array<1,YiType,false>& yi,
	   internal::uint options = ADEPT_INTERPOLATE_LINEAR | ADEPT_EXTRAPOLATE_DEFAULT,
	   MType extrap_value = std::numeric_limits<MType>::signaling_NaN()) {

    ADEPT_STATIC_ASSERT(MDims >= 2, TWO_DIMENSIONAL_INTERPOLATION_REQUIRES_2D_ARRAY);
    
    if (x.size() != M.size(0)) {
      throw(size_mismatch("Interpolation vector x must have same length as first dimension of M in interp2d"));
    }
    if (y.size() != M.size(1)) {
      throw(size_mismatch("Interpolation vector y must have same length as second dimension of M in interp2d"));
    }
    else if (x.size() < 2 || y.size() < 2) {
      throw(size_mismatch("Interpolation array must have at least two elements in each direction in interp2d"));
    }
    else if (xi.dimensions() != yi.dimensions()) {
      throw(size_mismatch("Indexing arrays must be the same shape in interp2d"));
    }

    internal::uint interp_scheme, extrap_policy;
    internal::extract_interp_extrap(options, interp_scheme, extrap_policy);
    
    Index ni = xi.size();
    ExpressionSize<MDims-1> ans_dims;
    ans_dims[0] = xi.size();
    for (int ii = 2; ii < MDims; ++ii) {
      ans_dims[ii-1] = M.size(ii);
    }

    Array<MDims-1,MType,MIsActive> ans(ans_dims);
    
    // Indices to the first of the two elements in each dimension, and
    // the weight of the first element
    IntVector xind0(ni);
    Vector xweight0(ni);
    IntVector yind0(ni);
    Vector yweight0(ni);
    boolVector is_valid(ni);
    is_valid = true;
    internal::InterpHelper<XType>::interp_get_indices_weights(x, xi, interp_scheme, extrap_policy,
							      xind0, xweight0, is_valid);
    internal::InterpHelper<YType>::interp_get_indices_weights(y, yi, interp_scheme, extrap_policy,
							      yind0, yweight0, is_valid);
    /*
    std::cout << "xind0 " << xind0 << "\n";
    std::cout << "xweight00 " << xweight0 << "\n";
    std::cout << "yind0 " << yind0 << "\n";
    std::cout << "yweight00 " << yweight0 << "\n";
    */
    for (Index ii = 0; ii < ni; ++ii) {
      if (is_valid(ii)) {
	// Bi-linear interpolation
	ans[ii] = yweight0(ii) * (      xweight0(ii)  * M[xind0(ii)][yind0(ii)]
				  +(1.0-xweight0(ii)) * M[xind0(ii)+1][yind0(ii)])
	  + (1.0-yweight0(ii)) * (      xweight0(ii)  * M[xind0(ii)][yind0(ii)+1]
				  +(1.0-xweight0(ii)) * M[xind0(ii)+1][yind0(ii)+1]);
      }
      else {
	ans[ii] = extrap_value;
      }
    }
    return ans;
  }

  // Ensure that 2D interpolation works if expressions are provided
  // for any of the arguments; these are converted to temporary
  // arrays.
  template <typename XType, typename YType, typename MType, typename XiType, class YiType,
	    class X, class Y, class M, class Xi, class Yi>
  Array<M::rank-1,MType,M::is_active>
  interp2d(const Expression<XType,X>& x,
	   const Expression<YType,Y>& y,
	   const Expression<MType,M>& m,
	   const Expression<XiType,Xi>& xi,
	   const Expression<YiType,Yi>& yi,
	   internal::uint options = ADEPT_INTERPOLATE_LINEAR | ADEPT_EXTRAPOLATE_DEFAULT,
	   MType extrap_value = std::numeric_limits<MType>::signaling_NaN()) {
    const Array<1,XType,false> x2(x.cast());
    const Array<1,YType,false> y2(y.cast());
    const Array<M::rank,MType,M::is_active> m2(m.cast());
    const Array<1,XiType,false> xi2(xi.cast());
    const Array<1,YiType,false> yi2(yi.cast());
    return interp2d(x2, y2, m2, xi2, yi2, options, extrap_value);
  }
  
  // 3D interpolation: as 1D interpolation but with two vectors
  // describing the dimensions of the interpolation array and two
  // vectors providing points at which interpolated values are
  // required
  template <typename XType, typename YType, typename ZType,
	    int MDims, typename MType, bool MIsActive,
	    typename XiType, typename YiType, typename ZiType>
  Array<MDims-2,MType,MIsActive>
  interp3d(const XType& x,
	   const YType& y,
	   const ZType& z,
	   const Array<MDims,MType,MIsActive>& M,
	   const Array<1,XiType,false>& xi,
	   const Array<1,YiType,false>& yi,
	   const Array<1,ZiType,false>& zi,
	   internal::uint options = ADEPT_INTERPOLATE_LINEAR | ADEPT_EXTRAPOLATE_DEFAULT,
	   MType extrap_value = std::numeric_limits<MType>::signaling_NaN()) {

    ADEPT_STATIC_ASSERT(MDims >= 3, THREE_DIMENSIONAL_INTERPOLATION_REQUIRES_3D_ARRAY);
    
    if (x.size() != M.size(0)) {
      throw(size_mismatch("Interpolation vector x must have same length as first dimension of M in interp3d"));
    }
    if (y.size() != M.size(1)) {
      throw(size_mismatch("Interpolation vector y must have same length as second dimension of M in interp3d"));
    }
    if (z.size() != M.size(2)) {
      throw(size_mismatch("Interpolation vector z must have same length as third dimension of M in interp3d"));
    }
    else if (x.size() < 2 || y.size() < 2 || z.size() < 2) {
      throw(size_mismatch("Interpolation array must have at least two elements in each direction in interp3d"));
    }
    else if (xi.dimensions() != yi.dimensions() || xi.dimensions() != zi.dimensions()) {
      throw(size_mismatch("Indexing arrays must be the same shape in interp3d"));
    }

    internal::uint interp_scheme, extrap_policy;
    internal::extract_interp_extrap(options, interp_scheme, extrap_policy);
    
    Index ni = xi.size();
    ExpressionSize<MDims-2> ans_dims;
    ans_dims[0] = xi.size();
    for (int ii = 3; ii < MDims; ++ii) {
      ans_dims[ii-2] = M.size(ii);
    }

    Array<MDims-2,MType,MIsActive> ans(ans_dims);
    
    // Indices to the first of the two elements in each dimension, and
    // the weight of the first element
    IntVector xind0(ni);
    Vector xweight0(ni);
    IntVector yind0(ni);
    Vector yweight0(ni);
    IntVector zind0(ni);
    Vector zweight0(ni);
    boolVector is_valid(ni);
    is_valid = true;
    internal::InterpHelper<XType>::interp_get_indices_weights(x, xi, interp_scheme, extrap_policy,
							      xind0, xweight0, is_valid);
    internal::InterpHelper<YType>::interp_get_indices_weights(y, yi, interp_scheme, extrap_policy,
							      yind0, yweight0, is_valid);
    internal::InterpHelper<ZType>::interp_get_indices_weights(z, zi, interp_scheme, extrap_policy,
							      zind0, zweight0, is_valid);
    for (Index ii = 0; ii < ni; ++ii) {
      if (is_valid(ii)) {
	// Tri-linear interpolation
	ans[ii] = xweight0(ii) *
	  (yweight0(ii) * (zweight0(ii) * M[xind0(ii)][yind0(ii)][zind0(ii)]
			   +(1.0-zweight0(ii)) * M[xind0(ii)][yind0(ii)][zind0(ii)+1])
	   + (1.0-yweight0(ii)) * (zweight0(ii)  * M[xind0(ii)][yind0(ii)+1][zind0(ii)]
				   +(1.0-zweight0(ii)) * M[xind0(ii)][yind0(ii)+1][zind0(ii)+1]))
	  + (1.0 - xweight0(ii)) *
	  (yweight0(ii) * (zweight0(ii) * M[xind0(ii)+1][yind0(ii)][zind0(ii)]
			   +(1.0-zweight0(ii)) * M[xind0(ii)+1][yind0(ii)][zind0(ii)+1])
	   + (1.0-yweight0(ii)) * (zweight0(ii)  * M[xind0(ii)+1][yind0(ii)+1][zind0(ii)]
				   +(1.0-zweight0(ii)) * M[xind0(ii)+1][yind0(ii)+1][zind0(ii)+1]));
      }
      else {
	ans[ii] = extrap_value;
      }
    }
    return ans;
  }

  // Ensure that 3D interpolation works if expressions are provided
  // for any of the arguments; these are converted to temporary
  // arrays.
  template <typename XType, typename YType, typename ZType, typename MType,
	    typename XiType, class YiType, class ZiType,
	    class X, class Y, class Z, class M, class Xi, class Yi, class Zi>
  Array<M::rank-2,MType,M::is_active>
  interp3d(const Expression<XType,X>& x,
	   const Expression<YType,Y>& y,
	   const Expression<ZType,Z>& z,
	   const Expression<MType,M>& m,
	   const Expression<XiType,Xi>& xi,
	   const Expression<YiType,Yi>& yi,
	   const Expression<ZiType,Zi>& zi,
	   internal::uint options = ADEPT_INTERPOLATE_LINEAR | ADEPT_EXTRAPOLATE_DEFAULT,
	   MType extrap_value = std::numeric_limits<MType>::signaling_NaN()) {
    const Array<1,XType,false> x2(x.cast());
    const Array<1,YType,false> y2(y.cast());
    const Array<1,ZType,false> z2(z.cast());
    const Array<M::rank,MType,M::is_active> m2(m.cast());
    const Array<1,XiType,false> xi2(xi.cast());
    const Array<1,YiType,false> yi2(yi.cast());
    const Array<1,ZiType,false> zi2(zi.cast());
    return interp3d(x2, y2, z2, m2, xi2, yi2, zi2, options, extrap_value);
  }
  
} // End namespace adept

#endif
