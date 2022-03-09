/* adept_half.h -- Half-precision support on ARM-NEON platforms

    Copyright (C) 2022 European Centre for Medium-Range Weather Forecasts

    Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptHalf_H
#define AdeptHalf_H 1

#include <iostream>

#if defined(__ARM_NEON) && defined(__clang__)
#include <arm_neon.h>
#else
#error "Half-precision currently only available on ARM NEON with the Clang compiler"
#endif

namespace adept {

  // Define "adept::half" to be a half-precision float
  typedef float16_t half;
  //  typedef _Float16 half;

  // To work in array containers we need to be able to send a
  // half-precision scalar to a stream
  std::ostream& operator<<(std::ostream& os, adept::half x) {
    os << static_cast<float>(x);
    return os;
  }
  
  // For the moment implement mathematical functions by first
  // converting to single precision
  half log(half x)        { return std::log(static_cast<float>(x)); }
  half log10(half x)      { return std::log10(static_cast<float>(x)); }
  half sin(half x)        { return std::sin(static_cast<float>(x)); }
  half cos(half x)        { return std::cos(static_cast<float>(x)); }
  half tan(half x)        { return std::tan(static_cast<float>(x)); }
  half asin(half x)       { return std::asin(static_cast<float>(x)); }
  half acos(half x)       { return std::acos(static_cast<float>(x)); }
  half atan(half x)       { return std::atan(static_cast<float>(x)); }
  half sinh(half x)       { return std::sinh(static_cast<float>(x)); }
  half cosh(half x)       { return std::cosh(static_cast<float>(x)); }
  half tanh(half x)       { return std::tanh(static_cast<float>(x)); }
  half abs(half x)        { return std::abs(static_cast<float>(x)); }
  half fabs(half x)       { return std::fabs(static_cast<float>(x)); }
  half exp(half x)        { return std::exp(static_cast<float>(x)); }
  half sqrt(half x)       { return std::sqrt(static_cast<float>(x)); }
  half ceil(half x)       { return std::ceil(static_cast<float>(x)); }
  half floor(half x)      { return std::floor(static_cast<float>(x)); }

  half pow(half x, half y)   { return std::pow(static_cast<float>(x), static_cast<float>(y)); }
  half atan2(half x, half y) { return std::atan2(static_cast<float>(x), static_cast<float>(y)); }
  half min(half x, half y)   { return std::min(static_cast<float>(x), static_cast<float>(y)); }
  half max(half x, half y)   { return std::max(static_cast<float>(x), static_cast<float>(y)); }
    
  // Functions defined in the std namespace in C++11 but only in the
  // global namespace before that
#ifdef ADEPT_CXX11_FEATURES
  half log2(half x)       { return std::log2(static_cast<float>(x)); }
  half expm1(half x)      { return std::expm1(static_cast<float>(x)); }
  half exp2(half x)       { return std::exp2(static_cast<float>(x)); }
  half log1p(half x)      { return std::log1p(static_cast<float>(x)); }
  half asinh(half x)      { return std::asinh(static_cast<float>(x)); }
  half acosh(half x)      { return std::acosh(static_cast<float>(x)); }
  half atanh(half x)      { return std::atanh(static_cast<float>(x)); }
  half erf(half x)        { return std::erf(static_cast<float>(x)); }
  half erfc(half x)       { return std::erfc(static_cast<float>(x)); }
  half cbrt(half x)       { return std::cbrt(static_cast<float>(x)); }
  half round(half x)      { return std::round(static_cast<float>(x)); }
  half trunc(half x)      { return std::trunc(static_cast<float>(x)); }
  half rint(half x)       { return std::rint(static_cast<float>(x)); }
  half nearbyint(half x)  { return std::nearbyint(static_cast<float>(x)); }
  half fmin(half x, half y)   { return std::fmin(static_cast<float>(x), static_cast<float>(y)); }
  half fmax(half x, half y)   { return std::fmax(static_cast<float>(x), static_cast<float>(y)); }
#else
  half log2(half x)       { return ::log2(static_cast<float>(x)); }
  half expm1(half x)      { return ::expm1(static_cast<float>(x)); }
  half exp2(half x)       { return ::exp2(static_cast<float>(x)); }
  half log1p(half x)      { return ::log1p(static_cast<float>(x)); }
  half asinh(half x)      { return ::asinh(static_cast<float>(x)); }
  half acosh(half x)      { return ::acosh(static_cast<float>(x)); }
  half atanh(half x)      { return ::atanh(static_cast<float>(x)); }
  half erf(half x)        { return ::erf(static_cast<float>(x)); }
  half erfc(half x)       { return ::erfc(static_cast<float>(x)); }
  half cbrt(half x)       { return ::cbrt(static_cast<float>(x)); }
  half round(half x)      { return ::round(static_cast<float>(x)); }
  half trunc(half x)      { return ::trunc(static_cast<float>(x)); }
  half rint(half x)       { return ::rint(static_cast<float>(x)); }
  half nearbyint(half x)  { return ::nearbyint(static_cast<float>(x)); }
  half fmin(half x, half y)   { return std::min(static_cast<float>(x), static_cast<float>(y)); }
  half fmax(half x, half y)   { return std::max(static_cast<float>(x), static_cast<float>(y)); }
#endif
  
}

#include <adept_arrays.h>

namespace adept {

  // For reduction functions (e.g. maxval) to work we need to know
  // positive and negative infinity of the half-precision type
  namespace internal {
    template <>
    struct numeric_limits<adept::half> {
      static adept::half min_inf() { unsigned short x = 0xfc00;
	return *reinterpret_cast<adept::half*>(&x); }
      static adept::half max_inf() { unsigned short x = 0x7c00;
	return *reinterpret_cast<adept::half*>(&x); }
    };
  }
  
  // Shortcuts
  typedef Array<1,half> halfVector;
  typedef Array<2,half> halfMatrix;
  typedef Array<3,half> halfArray3D;
  typedef Array<4,half> halfArray4D;
  typedef Array<5,half> halfArray5D;
  typedef Array<6,half> halfArray6D;
  typedef Array<7,half> halfArray7D;

}

#endif
