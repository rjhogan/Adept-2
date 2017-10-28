/* settings.cpp -- View/change the overall Adept settings

    Copyright (C) 2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <sstream>
#include <cstring>

#include <adept/base.h>
#include <adept/settings.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_OPENBLAS_CBLAS_HEADER
#include <cblas.h>
#endif

namespace adept {

  // -------------------------------------------------------------------
  // Get compile-time settings
  // -------------------------------------------------------------------

  // Return the version of Adept at compile time
  std::string
  version()
  {
    return ADEPT_VERSION_STR;
  }

  // Return the compiler used to compile the Adept library (e.g. "g++
  // [4.3.2]" or "Microsoft Visual C++ [1800]")
  std::string
  compiler_version()
  {
#ifdef CXX
    std::string cv = CXX; // Defined in config.h
#elif defined(_MSC_VER)
    std::string cv = "Microsoft Visual C++";
#else
    std::string cv = "unknown";
#endif

#ifdef __GNUC__

#define STRINGIFY3(A,B,C) STRINGIFY(A) "." STRINGIFY(B) "." STRINGIFY(C)
#define STRINGIFY(A) #A
    cv += " [" STRINGIFY3(__GNUC__,__GNUC_MINOR__,__GNUC_PATCHLEVEL__) "]";
#undef STRINGIFY
#undef STRINGIFY3

#elif defined(_MSC_VER)

#define STRINGIFY1(A) STRINGIFY(A)
#define STRINGIFY(A) #A
    cv += " [" STRINGIFY1(_MSC_VER) "]";
#undef STRINGIFY
#undef STRINGIFY1

#endif
    return cv;
  }

  // Return the compiler flags used when compiling the Adept library
  // (e.g. "-Wall -g -O3")
  std::string
  compiler_flags()
  {
#ifdef CXXFLAGS
    return CXXFLAGS; // Defined in config.h
#else
    return "unknown";
#endif
  }

  // Return a multi-line string listing numerous aspects of the way
  // Adept has been configured.
  std::string
  configuration()
  {
    std::stringstream s;
    s << "Adept version " << adept::version() << ":\n";
    s << "  Compiled with " << adept::compiler_version() << "\n";
    s << "  Compiler flags \"" << adept::compiler_flags() << "\"\n";
#ifdef BLAS_LIBS
    if (std::strlen(BLAS_LIBS) > 2) {
      const char* blas_libs = BLAS_LIBS + 2;
      s << "  BLAS support from " << blas_libs << " library\n";
    }
    else {
      s << "  BLAS support from built-in library\n";
    }
#endif
#ifdef HAVE_OPENBLAS_CBLAS_HEADER
    s << "  Number of BLAS threads may be specified up to maximum of "
      << max_blas_threads() << "\n";
#endif
    s << "  Jacobians processed in blocks of size " 
      << ADEPT_MULTIPASS_SIZE << "\n";
    return s.str();
  }


  // -------------------------------------------------------------------
  // Get/set number of threads for array operations
  // -------------------------------------------------------------------

  // Get the maximum number of threads available for BLAS operations
  int
  max_blas_threads()
  {
#ifdef HAVE_OPENBLAS_CBLAS_HEADER
    return openblas_get_num_threads();
#else
    return 1;
#endif
  }

  // Set the maximum number of threads available for BLAS operations
  // (zero means use the maximum sensible number on the current
  // system), and return the number actually set. Note that OpenBLAS
  // uses pthreads and the Jacobian calculation uses OpenMP - this can
  // lead to inefficient behaviour so if you are computing Jacobians
  // then you may get better performance by setting the number of
  // array threads to one.
  int
  set_max_blas_threads(int n)
  {
#ifdef HAVE_OPENBLAS_CBLAS_HEADER
    openblas_set_num_threads(n);
    return openblas_get_num_threads();
#else
    return 1;
#endif
  }

  // Was the library compiled with matrix multiplication support (from
  // BLAS)?
  bool
  have_matrix_multiplication() {
#ifdef HAVE_BLAS
    return true;
#else
    return false;
#endif
  }

  // Was the library compiled with linear algebra support (e.g. inv
  // and solve from LAPACK)
  bool
  have_linear_algebra() {
#ifdef HAVE_LAPACK
    return true;
#else
    return false;
#endif
  }

} // End namespace adept
