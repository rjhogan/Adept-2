/* math_benchmark.cpp - Benchmark mathematical functions

  Copyright (C) 2023 ECMWF

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

*/

#include <iostream>
#include <adept_arrays.h>
#include "Timer.h"

int main(int argc, const char** argv)
{
  using namespace adept;
  static const int N = 1024;
  int nrepeat = 1024*16;
  Vector x(N), y(N);

  Timer timer;
  timer.print_on_exit(true);
  int add_id = timer.new_activity("addition");
  int sub_id = timer.new_activity("subtraction");
  int mul_id = timer.new_activity("multiplication");
  int div_id = timer.new_activity("division");
  int exp_id = timer.new_activity("exp");
  int fastexp_id = timer.new_activity("fastexp");
  int log_id = timer.new_activity("log");
  int sin_id = timer.new_activity("sin");

  x = 1.001;
  y = x*x;
  y = 0.0;
  
  timer.start(add_id);
  for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
    y += x;
  }
  timer.stop();

  y = 0.0;
  
  timer.start(sub_id);
  for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
    y -= x;
  }
  timer.stop();

  y = 1.0;
  
  timer.start(mul_id);
  for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
    y *= x;
  }
  timer.stop();

  std::cout << "y=" << y(0) << "\n";
  
  timer.start(div_id);
  for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
    y /= x;
  }
  timer.stop();

  x = 0.001;
  
  timer.start(exp_id);
  for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
    y = exp(x);
    x = y-1.001;
  }
  timer.stop();

  std::cout << "y=" << y(0) << "\n";
  
  x = 0.001;

 
  timer.start(fastexp_id);
  for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
    y = fastexp(x);
    x = y-1.001;
  }
  timer.stop();

  std::cout << "y=" << y(0) << "\n";
  
  x = 1.001;
  
  timer.start(log_id);
  for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
    y = log(x);
    x = y+1.0;
  }
  timer.stop();

  std::cout << "y=" << y(0) << "\n";
  
  x = 1.001;
  
  timer.start(sin_id);
  for (int irepeat = 0; irepeat < nrepeat; ++irepeat) {
    y = sin(x);
    y = x;
  }
  timer.stop();

  std::cout << "y=" << y(0) << "\n";

  std::cout << "RELATIVE COSTS\n";
  std::cout << "div/mul = " << timer.timing(div_id)/timer.timing(mul_id) << "\n";
  std::cout << "exp/mul = " << timer.timing(exp_id)/timer.timing(mul_id) << "\n";
  std::cout << "fastexp/mul = " << timer.timing(fastexp_id)/timer.timing(mul_id) << "\n";
  std::cout << "log/mul = " << timer.timing(log_id)/timer.timing(mul_id) << "\n";
  std::cout << "sin/mul = " << timer.timing(sin_id)/timer.timing(mul_id) << "\n";
  
}
