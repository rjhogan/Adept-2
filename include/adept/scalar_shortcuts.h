/* shortcuts.h -- Definitions of "shortcut" typedefs for scalar types

    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptScalarShortcuts_H
#define AdeptScalarShortcuts_H

#include <complex>

#ifndef ADEPT_NO_AUTOMATIC_DIFFERENTIATION
// First the case when automatic differentiation is ON

#include <adept/Active.h>

namespace adept {

  typedef Active<Real> aReal;
  typedef Active<float> afloat;
  typedef Active<double> adouble;

  typedef Active<std::complex<Real> > aComplex;
  typedef Active<std::complex<float> > aComplexFloat;
  typedef Active<std::complex<double> > aComplexDouble;

  inline Real value(Real x) { return x; }

} // End namespace adept


#else
// Second the case when automatic differentiation is OFF

#include <adept/base.h>

namespace adept {

  typedef Real aReal;
  typedef float afloat;
  typedef double adouble;

  typedef std::complex<Real> aComplex;
  typedef std::complex<float> aComplexFloat;
  typedef std::complex<double> aComplexDouble;

  // Normally value(x) returns the inactive part of x, so if x is
  // inactive we simply return a constant reference to x
  template <typename T>
  inline const T& value(const T& x) { return x; }

  inline Real value(Real x) { return x; }

} // End namespace adept

#endif

#endif
