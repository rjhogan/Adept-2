#include <iostream>

#include <adept_arrays.h>

#include "Timer.h"

template<bool IsActive>
double
time_operation(int n, int nrepeat, bool is_col_major)
{
  adept::Array<2,double,IsActive> A, B, C;
  Timer timer;
  int matmul_timer_id = timer.new_activity("matmul");
  if (is_col_major) {
    A.resize_column_major(adept::expression_size(n,n));
    B.resize_column_major(adept::expression_size(n,n));
    C.resize_column_major(adept::expression_size(n,n));
  }
  else {
    A.resize(n,n);
    B.resize(n,n);
    C.resize(n,n);
  }
  for (int irepeat = -nrepeat/10; irepeat < nrepeat; ++irepeat) {
    A = 1.1;
    B = 2.2;
    if (IsActive) {
      adept::active_stack()->new_recording();
    }
    if (irepeat >= 0) {
      timer.start(matmul_timer_id);
    }
    C = A ** B;
    if (irepeat >= 0) {
      timer.stop();
    }
  }
  if (IsActive && n < 8) {
    std::cout << "C=" << C;
    std::cout << *adept::active_stack();
    adept::active_stack()->print_statements();
  }
  return timer.timing(matmul_timer_id) / nrepeat;
}


int
main(int argc, char* argv[])
{
  int ibegin = 1;
  int iend = 18;
  int nrepeat = 10;
  bool is_col_major = false;

  adept::Stack stack;
  int n = 2;
  std::cout << "Dense N-by-N matrix-matrix multiplication\n";
  std::cout << " N        inactive time (us)   inactive flops    active time (us)    active flops\n";
  for (int i = ibegin; i <= iend; ++i) {
    std::cout << n << "  ";

    double t = time_operation<false>(n, nrepeat, is_col_major);
    std::cout << t*1.0e6 << "  " << (n*n*n) / t << "  ";

    t = time_operation<true>(n, nrepeat, is_col_major);
    std::cout << t*1.0e6 << "  " << (n*n*n) / t;

    std::cout << "\n";

    n *= 2;
  }
  return 0;
}
