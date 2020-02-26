/* test_fastexp.cpp - Test Adept's fast exponential for correctness 

  Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

  This file tests Adept's fast exponential function "fastexp", which
  is vectorizable.
*/

#include <iostream>
#include <limits>
#include "adept_arrays.h"

using namespace adept;

int main(int argc, const char** argv)
{
  {
    std::cout << "DOUBLE PRECISION\n";
    std::cout << "Packet<double>::size = " << internal::Packet<double>::size << "\n";
    Vector x = linspace(-750.0,750.0,128);
    x(end) = std::numeric_limits<double>::quiet_NaN();
    Vector exponential = exp(x);
    Vector fast_exponential = fastexp(x);
    Vector fractional_error = (fast_exponential - exponential) / exponential;
    //    std::cout << fractional_error << "\n";
    Matrix M(128,4);
    M(__,0) = x;
    M(__,1) = exponential;
    M(__,2) = fast_exponential;
    M(__,3) = fractional_error;
    std::cout << "x  exp(x)  fastexp(x)  fractional-error";
    std::cout << M << "\n";
  }
  {
    std::cout << "\nSINGLE PRECISION\n";
    std::cout << "Packet<float>::size = " << internal::Packet<float>::size << "\n";
    floatVector x = linspace(-100.0,100.0,128);
    x(end) = std::numeric_limits<float>::quiet_NaN();
    floatVector exponential = exp(x);
    floatVector fast_exponential = fastexp(x);
    floatVector fractional_error = (fast_exponential - exponential) / exponential;
    floatMatrix M(128,4);
    M(__,0) = x;
    M(__,1) = exponential;
    M(__,2) = fast_exponential;
    M(__,3) = fractional_error;
    std::cout << "x  exp(x)  fastexp(x)  fractional-error";
    std::cout << M << "\n";
  }
  return 0;
}
