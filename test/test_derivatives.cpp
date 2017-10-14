/* test_derivatives.cpp - Test derivatives of mathematical functions 

    Copyright (C) 2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

*/

#include <adept_arrays.h>


#define TEST_UNARY_FUNC(FUNC)					\
  {								\
    std::cout << "  Checking " << #FUNC << "... \t";		\
    aVector x = x_save;					\
    stack.new_recording();					\
    aVector y = FUNC(x);					\
    Vector dy_dx_num  = (FUNC(x_save+dx)-FUNC(x_save)) / dx;	\
    Vector dy_dx_adept(N);					\
    for (int i = 0; i < N; ++i) {				\
      x[i].set_gradient(1.0);					\
      stack.forward();						\
      y[i].get_gradient(dy_dx_adept[i]);			\
    }								\
    Real max_err						\
      = maxval(abs(dy_dx_adept-dy_dx_num));			\
    Real max_frac_err						\
      = maxval(abs(dy_dx_adept-dy_dx_num)/dy_dx_adept);		\
    if (max_err == 0) {						\
      std::cout << "max error = 0: PASSED\n";			\
    }								\
    if (max_frac_err <= MAX_FRAC_ERR) {				\
      std::cout << "max fractional error = " << max_frac_err	\
		<< ": PASSED\n";				\
    }								\
    else {							\
      std::cout << "max fractional error = "			\
		<< max_frac_err << ": FAILED\n";		\
      std::cout << "    Adept     dy/dx = "			\
		<< dy_dx_adept << "\n";				\
      std::cout << "    Numerical dy/dx = " << dy_dx_num << "\n";	\
      error_too_large = true;					\
    }								\
  }

#define TEST_BINARY_FUNC(FUNC)					\
  {								\
    std::cout << "  Checking " << #FUNC << "... \t";		\
    aVector x = x_save;					\
    aVector y = y_save;						\
    stack.new_recording();					\
    aVector z = FUNC(x,y);					\
    Vector dz_dx_num						\
      = (FUNC(x_save+dx,y_save)-FUNC(x_save,y_save)) / dx; \
    Vector dz_dy_num						\
      = (FUNC(x_save,y_save+dy)-FUNC(x_save,y_save)) / dy;	\
    Vector dz_dx_adept(N);					\
    Vector dz_dy_adept(N);					\
    for (int i = 0; i < N; ++i) {				\
      z[i].set_gradient(1.0);					\
      stack.reverse();						\
      x[i].get_gradient(dz_dx_adept[i]);			\
      y[i].get_gradient(dz_dy_adept[i]);			\
    }								\
    Real max_err						\
      = std::max(maxval(abs(dz_dx_adept-dz_dx_num)),		\
		 maxval(abs(dz_dy_adept-dz_dy_num)));		\
    Real max_frac_err						\
      = std::max(maxval(abs(dz_dx_adept-dz_dx_num)/dz_dx_adept),	\
		 maxval(abs(dz_dy_adept-dz_dy_num)/dz_dy_adept));	\
    if (max_err == 0) {						\
      std::cout << "max error = 0: PASSED\n";			\
    }								\
    if (max_frac_err <= MAX_FRAC_ERR) {				\
      std::cout << "max fractional error = " << max_frac_err	\
		<< ": PASSED\n";				\
    }								\
    else {							\
      std::cout << "max fractional error = "			\
		<< max_frac_err << ": FAILED\n";		\
      std::cout << "    Adept     dz/dx = " << dz_dx_adept << "\n";	\
      std::cout << "    Adept     dz/dy = " << dz_dy_adept << "\n";	\
      std::cout << "    Numerical dz/dx = " << dz_dx_num << "\n";	\
      std::cout << "    Numerical dz/dy = " << dz_dy_num << "\n";	\
      error_too_large = true;					\
    }								\
  }


int
main(int argc, const char** argv) {
  using namespace adept;

  Stack stack;

  static const int N             = 12;
  static const Real MAX_FRAC_ERR = 1.0e-5;

  Vector x_save(N);
  x_save = 0.2;
  x_save << 0.01, 0.4, 0.99;

  Vector y_save(N);
  y_save = 0.7;
  y_save << 0.9, 0.6, 0.1;

  Real dx = 1.0e-8;

  if (sizeof(Real) < 8) {
    // Single precision only works with larger perturbations
    dx = 1.0e-5;
  }

  Real dy = dx;

  bool error_too_large = false;  

  std::cout << "EVALUATING UNARY FUNCTIONS\n";
  std::cout << "For functions of the form y=FUNC(x), where x=" << x_save << ",\n";
  std::cout << "checking that fractional difference between dy/dx computed using Adept\n";
  std::cout << "and numerically by perturbing x by " << dx << " is less than " << MAX_FRAC_ERR << ".\n";    

  
  TEST_UNARY_FUNC(-); // Unary minus
  TEST_UNARY_FUNC(+); // Unary plus
  TEST_UNARY_FUNC(log);
  TEST_UNARY_FUNC(log10);
  TEST_UNARY_FUNC(sin);
  TEST_UNARY_FUNC(cos);
  TEST_UNARY_FUNC(tan);
  TEST_UNARY_FUNC(asin);
  TEST_UNARY_FUNC(acos);
  TEST_UNARY_FUNC(atan);
  TEST_UNARY_FUNC(sinh);
  TEST_UNARY_FUNC(cosh);
  TEST_UNARY_FUNC(tanh);
  TEST_UNARY_FUNC(abs);
  TEST_UNARY_FUNC(fabs);
  TEST_UNARY_FUNC(exp);
  TEST_UNARY_FUNC(sqrt);
  TEST_UNARY_FUNC(ceil);
  TEST_UNARY_FUNC(floor);
  TEST_UNARY_FUNC(log2);
  TEST_UNARY_FUNC(expm1);
  TEST_UNARY_FUNC(exp2);
  TEST_UNARY_FUNC(log1p);
  TEST_UNARY_FUNC(asinh);
  TEST_UNARY_FUNC(acosh);
  TEST_UNARY_FUNC(atanh);
  TEST_UNARY_FUNC(erf);
  TEST_UNARY_FUNC(erfc);
  TEST_UNARY_FUNC(cbrt);
  TEST_UNARY_FUNC(round);
  TEST_UNARY_FUNC(trunc);
  TEST_UNARY_FUNC(rint);
  TEST_UNARY_FUNC(nearbyint);

  std::cout << "EVALUATING BINARY FUNCTIONS\n";
  std::cout << "For functions of the form z=FUNC(x,y), where x=" << x_save << ",\n";
  std::cout << "and y=" << y_save << ", checking that fractional difference between\n";
  std::cout << "dz/dx and dz/dy computed using Adept and numerically by perturbing\n";
  std::cout << "x and y by " << dx << " is less than " << MAX_FRAC_ERR << ".\n";    

  TEST_BINARY_FUNC(pow);
  TEST_BINARY_FUNC(atan2);
  TEST_BINARY_FUNC(max);
  TEST_BINARY_FUNC(min);
  TEST_BINARY_FUNC(fmax);
  TEST_BINARY_FUNC(fmin);


  if (error_too_large) {
    std::cerr << "*** Error: fractional error in the derivatives of some functions too large\n";

    if (sizeof(Real) < 8) {
      std::cerr << "*** (but you are using less than double precision so it is not surprising)\n";
    }

    return 1;
  }
  else {
    return 0;
  }
}
