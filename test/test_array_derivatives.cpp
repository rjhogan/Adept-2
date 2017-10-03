/* test_array_derivatives.cpp - Test derivatives of array expressions

    Copyright (C) 2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

*/

#include <adept_arrays.h>


// Arbitrary algorithm converting array of general type A to scalar of
// type S, which may be active or passive
template <class A, class S>
void algorithm(const A& x, S& y) {
  using namespace adept;
  A tmp;
  intVector index(2);
  index << 1, 0;
  tmp = atan2((exp(x) * x), spread<0>(x(index,1),2)) / x(0,0);
  y = sum(tmp);
}


int
main(int argc, const char** argv) {
  using namespace adept;

  Stack stack;

  // Matrix dimension
  static const int N = 2;
  static const Real MAX_FRAC_ERR = 1.0e-5;

  // Perturbation size for numerical calculation
  Real dx = 1.0e-6;

  if (sizeof(Real) < 8) {
    // Single precision only works with larger perturbations
    dx = 1.0e-4;
  }

  // Maximum fractional error
  Real max_frac_err;
  bool error_too_large = false;

  // Input data
  Matrix X(N,N);
  X << 2, 3, 5, 7;
  
  // Numerical calculation 
  std::cout << "NUMERICAL CALCULATION\n";
  Matrix dJ_dx_num(N,N);
  {
    Real J;
    algorithm(X, J);
    std::cout << "J = " << J << "\n";

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
	Matrix Xpert(N,N);
	Xpert = X;
	Xpert(i,j) += dx;
	Real Jpert;
	algorithm(Xpert, Jpert);
	dJ_dx_num(i,j) = (Jpert - J) / dx;
      }
    }
  }

  std::cout << "dJ_dx_num = " << dJ_dx_num << "\n";

  std::cout << "\nNUMERICAL CALCULATION WITH \"FixedArray\"\n";
  Matrix22 dJ_dx_num_FixedArray;
  {
    Real J;
    algorithm(X, J);
    std::cout << "J = " << J << "\n";

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
	Matrix22 Xpert = X;
	Xpert(i,j) += dx;
	Real Jpert;
	algorithm(Xpert, Jpert);
	dJ_dx_num_FixedArray(i,j) = (Jpert - J) / dx;
      }
    }
  }

  std::cout << "dJ_dx_num_FixedArray = " << dJ_dx_num_FixedArray << "\n";

 // Adept calculation with aArray
  std::cout << "\nADEPT CALCULATION WITH \"aArray\"\n";
  Matrix dJ_dx_adept_Array(N,N);
  {
    aMatrix aX = X;
    stack.new_recording();
    aReal aJ;
    algorithm(aX, aJ);
    std::cout << "J = " << aJ << "\n";
    aJ.set_gradient(1.0);
    stack.reverse();
   
    dJ_dx_adept_Array = aX.get_gradient();
  }

  std::cout << "dJ_dx_adept_Array = " << dJ_dx_adept_Array << "\n";

  max_frac_err = maxval(abs(dJ_dx_adept_Array-dJ_dx_num)/dJ_dx_num);
  if (max_frac_err <= MAX_FRAC_ERR) {
    std::cout << "max fractional error = " << max_frac_err
		<< ": PASSED\n";
  }
  else {
    std::cout << "max fractional error = "
	      << max_frac_err << ": FAILED\n";
    error_too_large = true;
  }
  // Adept calculation with aFixedArray
  std::cout << "\nADEPT CALCULATION WITH \"aFixedArray\"\n";
  Matrix dJ_dx_adept_FixedArray;
  {
    aMatrix22 aX = X;
    stack.new_recording();
    aReal aJ;
    algorithm(aX, aJ);
    std::cout << "J = " << aJ << "\n";
    aJ.set_gradient(1.0);
    stack.reverse();
    dJ_dx_adept_FixedArray = aX.get_gradient();

  }
  std::cout << "dJ_dx_adept_FixedArray = " << dJ_dx_adept_FixedArray << "\n";

  max_frac_err = maxval(abs(dJ_dx_adept_FixedArray-dJ_dx_num)/dJ_dx_num);
  if (max_frac_err <= MAX_FRAC_ERR) {
    std::cout << "max fractional error = " << max_frac_err
		<< ": PASSED\n";
  }
  else {
    std::cout << "max fractional error = "
	      << max_frac_err << ": FAILED\n";
    error_too_large = true;
  }

  std::cout << "\n";

  if (error_too_large) {
    std::cerr << "*** Error: fractional error in the derivatives of some configurations too large\n";

    if (sizeof(Real) < 8) {
      std::cerr << "*** (but you are using less than double precision so it is not surprising)\n";
    }

    return 1;
  }
  else {
    return 0;
  }


}
