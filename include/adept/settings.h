/* settings.h -- View/change the overall Adept settings

    Copyright (C) 2016-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptSettings_H
#define AdeptSettings_H 1

#include <string>

namespace adept {

  // -------------------------------------------------------------------
  // Get compiler settings
  // -------------------------------------------------------------------

  // Return the version of Adept at compile time
  std::string version();

  // Return the compiler used to compile the Adept library (e.g. "g++ 4.3.2")
  std::string compiler_version();

  // Return the compiler flags used when compiling the Adept library
  // (e.g. "-Wall -g -O3")
  std::string compiler_flags();
  
  // Return a multi-line string listing numerous aspects of the way
  // Adept has been configured.
  std::string configuration();

  // Was the library compiled with matrix multiplication support (from
  // BLAS)?
  bool have_matrix_multiplication();

  // Was the library compiled with linear algebra support (e.g. inv
  // and solve from LAPACK)
  bool have_linear_algebra();

  // -------------------------------------------------------------------
  // Get/set number of threads for array operations
  // -------------------------------------------------------------------

  // Get the maximum number of threads available for BLAS operations
  int max_blas_threads();

  // Set the maximum number of threads available for BLAS operations
  // (zero means use the maximum sensible number on the current
  // system), and return the number actually set.  Note that OpenBLAS
  // uses pthreads and the Jacobian calculation uses OpenMP - this can
  // lead to inefficient behaviour so if you are computing Jacobians
  // then you may get better performance by setting the number of
  // array threads to one.
  int set_max_blas_threads(int n);

} // End namespace adept

#endif
