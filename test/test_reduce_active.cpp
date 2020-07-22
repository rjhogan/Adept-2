/* test_reduce_active.cpp

  Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

  This file tests reduce operations on active vectors
*/

#include <iostream>
#include "adept_arrays.h"

using namespace adept;

#define TEST_REDUCE(FUNC)			\
  {						\
    std::cout << "\nTESTING REDUCE FUNCTION "	\
	      << #FUNC << "\n";			\
    stack.new_recording();			\
    aReal J = FUNC(x);				\
    Real Jp = FUNC(value(x));			\
    J.set_gradient(1.0);			\
    stack.reverse();				\
    Vector dJdx = x.get_gradient();		\
    std::cout << #FUNC << "(x) = "		\
	      << J << "\n";			\
    std::cout << #FUNC << "(value(x)) = "       \
	      << Jp << "\n";	\
    std::cout << "d(" << #FUNC << "(x))/dx = "	\
              << dJdx << "\n";			\
    if (J != Jp) { ++status; }		        \
    stack.print_statements();			\
  }


int
main(int argc, const char** argv)
{
  Stack stack;

  aVector x(5);
  x << -2.0, -3.0, -1.0, -50.0, 7.0;

  std::cout << "x = " << x << "\n";

  int status = 0;

  TEST_REDUCE(sum);
  TEST_REDUCE(mean);
  TEST_REDUCE(maxval);
  TEST_REDUCE(minval);
  TEST_REDUCE(product);
  TEST_REDUCE(norm2);

  // Test product by hand
  {
    std::cout << "\nTESTING MANUAL PRODUCT\n";
    stack.new_recording();
    //aReal J = x(0)*x(1)*x(2)*x(3)*x(4);
    aReal J = x(0)*x(1);
    J *= x(2);
    J *= x(3);
    J *= x(4);
    J.set_gradient(1.0);
    stack.reverse();
    Vector dJdx = x.get_gradient();
    std::cout << "manual_product(x) = " << J << "\n";
    std::cout << "d(manual_product(x))/x = " << dJdx << "\n";
    stack.print_statements();
  }

  // Test norm2 by hand
  {
    std::cout << "\nTESTING MANUAL NORM2\n";
    stack.new_recording();
    aReal J = sqrt(x(0)*x(0) + x(1)*x(1) + x(2)*x(2)
		   + x(3)*x(3) + x(4)*x(4));
    J.set_gradient(1.0);
    stack.reverse();
    Vector dJdx = x.get_gradient();
    std::cout << "manual_norm2(x) = " << J << "\n";
    std::cout << "d(manual_norm2(x))/x = " << dJdx << "\n";
    stack.print_statements();
  }

  if (status != 0) {
    std::cout << "Error: " << status << " of the active/passive reduce operations are different\n";
  }

  return status;
}

