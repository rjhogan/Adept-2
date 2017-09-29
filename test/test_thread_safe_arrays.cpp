/* test_thread_safe_arrays.cpp - Tests that Adept arrays are thread-safe

  Copyright (C) 2017 ECMWF

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

*/

#ifdef _OPENMP
#include <omp.h>
#endif

//#define ADEPT_STORAGE_THREAD_SAFE 1

#include <adept_arrays.h>

int main(int argc, const char** argv)
{
  using namespace adept;

  int N = 2;
  Matrix A(N,N);
  SymmMatrix S(N);

  Matrix B;
  SymmMatrix T;
#ifdef ADEPT_STORAGE_THREAD_SAFE
  std::cout << "Storage should be thread safe\n";
  // B shares the data and increases the reference counter of the
  // shared Storage object. If A goes out of scope, B will "steal" the
  // data.
  B >>= A;
  T >>= S;
#else
  std::cout << "Storage is not thread safe: using soft_link()\n";
  // B points to the data but does not have access to the Storage
  // object. If A goes out of scope, B will most likely point to an
  // inaccessible memory location.
  B >>= A.soft_link();
  T >>= S.soft_link();
#endif

  A = 1.0; // Also seen by B
  S = 2.0; // Also seen by S

  int nthreads = 1;

#ifdef _OPENMP
  nthreads = omp_get_max_threads();
  std::cout << omp_get_num_procs() << " processors available running maximum of "
	    << nthreads << " threads\n";
#else
  std::cout << "Compiled without OpenMP support: 1 thread\n";
#endif

  // The following almost always causes a crash if the code is not
  // properly thread safe
#pragma omp parallel for
  for (int i = 0; i < N*1000; ++i) {

    for (int j = 0; j < N*1000; ++j) {
      B[j % N] = noalias(B(__,j)) + T.diag_vector();
    }

  }

  if (nthreads > 1) {
    std::cout << "Parallel subsetting of array zillions of times was successful\n";
  }
  else {
    std::cout << "Serial subsetting of array zillions of times was successful (unsurprisingly)\n";
  }

  return 0;

}
