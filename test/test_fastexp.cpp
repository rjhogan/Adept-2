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
#include "adept_arrays.h"

using namespace adept;

int main(int argc, const char** argv)
{
  {
    std::cout << "DOUBLE PRECISION\n";
    std::cout << "Packet<double>::size = " << internal::Packet<double>::size << "\n";
    Vector x = linspace(-700.0,700.0,128);
    Vector exponential = exp(x);
    Vector fast_exponential = fastexp(x);
    Vector fractional_error = (fast_exponential - exponential) / exponential;
    std::cout << fractional_error << "\n";
    //std::cout << (fastexp(x(stride(0,end,2))) - exponential(stride(0,end,2)))
    //  / exponential(stride(0,end,2)) << "\n";
  }
  {
    std::cout << "SINGLE PRECISION\n";
    std::cout << "Packet<float>::size = " << internal::Packet<float>::size << "\n";
    floatVector x = linspace(-87.0,87.0,128);
    floatVector exponential = exp(x);
    floatVector fast_exponential = fastexp(x);
    floatVector fractional_error = (fast_exponential - exponential) / exponential;
    std::cout << fractional_error << "\n";
    //    std::cout << (fastexp(x(stride(0,end,2))) - exponential(stride(0,end,2)))
    //  / exponential(stride(0,end,2)) << "\n";
  }
  return 0;
}
