/* test_fastexp.cpp - Test Adept's fast exponential for correctness and speed

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
  Vector x = linspace(-700.0,700.0,2801);
  Vector exponential = exp(x);
  Vector fast_exponential = fastexp(x);
  Vector fractional_error = (fast_exponential - exponential) / exponential;
  std::cout << fractional_error;
  return 0;
}
