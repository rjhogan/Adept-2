# Configure autoconf for the Adept library

### GENERAL CONFIGURATION ###

AC_PREREQ([2.61])
AC_INIT([adept], [2.0.5], [r.j.hogan@ecmwf.int], [adept], [http://www.met.reading.ac.uk/clouds/adept/])
AC_LANG([C++])
AC_CONFIG_SRCDIR([adept/Stack.cpp])
AC_CONFIG_HEADERS([config.h config_platform_independent.h])
AM_INIT_AUTOMAKE([-Wall -Werror])
AC_CONFIG_MACRO_DIR([m4])

# Checks for programs
AC_PROG_CXX
AC_PROG_F77
AC_PROG_MAKE_SET
m4_ifdef([AM_PROG_AR],[AM_PROG_AR])
AC_PROG_LIBTOOL

# Check for system features
AC_CHECK_HEADERS([sys/time.h])
AC_CHECK_FUNCS([gettimeofday pow sqrt])

# Check for OpenMP
AC_OPENMP
AC_SUBST(AM_CXXFLAGS,"$OPENMP_CXXFLAGS")

#### LIBRARIES NEEDED BY ADEPT ###

# Check for BLAS and LAPACK
# First we need this since the libraries are Fortran called from C++
AC_F77_LIBRARY_LDFLAGS
# The following tests for both BLAS and LAPACK
AX_LAPACK

# Dependencies dictate the following order of libraries
LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS"
# FLIBS should be included in LDADD or LIBADD in the relevant
# Makefile.am

# If the BLAS library is OpenBLAS then we need to give the user the
# option to change the number of threads, since OpenBLAS's pthreads
# can clash with Adept's use of OpenMP, leading to suboptimal
# performance.
ac_have_openblas_cblas_header=no

if test "$ax_blas_ok" = yes
then
	if test "x$BLAS_LIBS" = "x-lopenblas"
	then	
		AC_MSG_CHECKING([whether cblas.h is from OpenBLAS])
		AC_TRY_LINK([#include <cblas.h>],
		[openblas_set_num_threads(1)],
		[ac_have_openblas_cblas_header=yes
		AC_MSG_RESULT(yes)
		AC_DEFINE([HAVE_OPENBLAS_CBLAS_HEADER],1,[Is the clbas.h header file from OpenBLAS?])],
		AC_MSG_RESULT(no))
	fi
fi

### LIBRARIES THAT MAY BE USED BY TEST PROGRAMS ###

# Checks for GNU Scientific Library
AC_CHECK_LIB([gsl],[gsl_multimin_fdfminimizer_alloc],[AC_MSG_NOTICE([Note that GSL is not used by Adept, just by one of the test programs])])
AC_SUBST(USE_GSL, ["$ac_cv_lib_gsl_gsl_multimin_fdfminimizer_alloc"])

# Check for ADOL-C automatic differentiation library
AC_CHECK_HEADERS([adolc/adolc.h])
AC_CHECK_LIB([adolc],[tapestats])

# Check for SACADO automatic differentiation library
ac_have_sacado=no
save_LIBS=$LIBS
LIBS="$LIBS -lsacado -lteuchos"
AC_MSG_CHECKING([whether Sacado is installed])
AC_TRY_LINK([#include <Sacado.hpp>],
[Sacado::ELRFad::DFad<double> v = 1.0],
[ac_have_sacado=yes
AC_MSG_RESULT(yes)
AC_DEFINE([HAVE_SACADO],1,[Is the Sacado library working?])],
[LIBS=$save_LIBS
AC_MSG_RESULT(no)])

# Check for CppAD automatic differentiation library
AC_CHECK_HEADERS([cppad/cppad.hpp])
if test "$ac_cv_header_cppad_cppad_hpp" = yes
then
   AC_DEFINE([NDEBUG],1,[If CppAD is being used by the benchmarking program then it is much faster with debugging disabled])
fi


### CREATE MAKEFILES AND CONFIG HEADER ###

AC_CONFIG_FILES([Makefile makefile_include adept/Makefile include/Makefile benchmark/Makefile])

AC_DEFINE_UNQUOTED([CXX],["$CXX"],[C++ compiler])
AC_DEFINE_UNQUOTED([CXXFLAGS],["$CXXFLAGS"],[Flags passed to C++ compiler])
AC_DEFINE_UNQUOTED([BLAS_LIBS],["$BLAS_LIBS"],[BLAS library option])

AH_BOTTOM([/* Use ADOLC only if both the library and the header files are available */
#if defined( HAVE_LIBADOLC ) && defined( HAVE_ADOLC_ADOLC_H )
#define HAVE_ADOLC 1
#endif])
AH_BOTTOM([/* Use CPPAD if the header files are available */
#if defined( HAVE_CPPAD_CPPAD_HPP )
#define HAVE_CPPAD 1
#endif])

AC_OUTPUT


### REPORT CONFIGURATION TO THE USER ###

AC_MSG_NOTICE([********************* Summary **************************************])
AC_MSG_NOTICE([  CXX = $CXX ])
AC_MSG_NOTICE([  CPPFLAGS = $CPPFLAGS])
AC_MSG_NOTICE([  CXXFLAGS = $CXXFLAGS $OPENMP_CXXFLAGS])
AC_MSG_NOTICE([  LDFLAGS =  $LDFLAGS])
AC_MSG_NOTICE([  LIBS = $LIBS])
AC_MSG_NOTICE([Typing "make; make install" will install Adept header files in $includedir])
AC_MSG_NOTICE([and the static and shared libraries as $libdir/libadept.*, where])
AC_MSG_NOTICE([prefix=$prefix])
AC_MSG_NOTICE([********************* Libraries used by Adept **********************])
ac_warn_given=no
if test "$ax_blas_ok" = yes
then
	AC_MSG_NOTICE([BLAS (Basic Linear Algebra Subprograms) will be used: BLAS_LIBS = $BLAS_LIBS])
	if test "$ac_have_openblas_cblas_header" = yes
	then
	   AC_MSG_NOTICE([  Number of BLAS threads may be controlled at run time])
	fi
else
	AC_MSG_NOTICE([BLAS (Basic Linear Algebra Subprograms) will not be used: MATRIX MULTIPLICATION IS UNAVAILABLE])
	ac_warn_given=yes
fi
if test "$ax_lapack_ok" = yes
then
	AC_MSG_NOTICE([LAPACK (Linear Algebra Package) will be used: LAPACK_LIBS = $LAPACK_LIBS])
else
	AC_MSG_NOTICE([LAPACK (Linear Algebra Package) will not be used: LINEAR ALGEBRA ROUTINES ARE UNAVAILABLE])
	ac_warn_given=yes
fi

AC_MSG_NOTICE([********************* Libraries used by test programs **************])

if test "$ac_cv_lib_gsl_gsl_multimin_fdfminimizer_alloc" = no
then
	AC_MSG_NOTICE([GNU Scientific Library (GSL) not found; Adept will compile all the])
	AC_MSG_NOTICE([example programs except test/test_gsl_interface.])
	ac_warn_given=yes
else
	AC_MSG_NOTICE([GNU Scientific Library (GSL) found; Adept will compile all the])
	AC_MSG_NOTICE([example programs.])
fi

AC_MSG_NOTICE([********************* Benchmark program ****************************])
AC_MSG_NOTICE([The benchmarking program, "benchmark/advection_benchmark", will be])
AC_MSG_NOTICE([compiled with support for these automatic differentiation libraries:])
AC_MSG_NOTICE([   Adept: yes])

if test "$ac_cv_lib_adolc_tapestats" = yes -a "$ac_cv_header_adolc_adolc_h" = yes
then
   	AC_MSG_NOTICE([   ADOLC: yes])
else
	AC_MSG_NOTICE([   ADOLC: no])
	ac_warn_given=yes
fi

if test "$ac_cv_header_cppad_cppad_hpp" = yes
then
   	AC_MSG_NOTICE([   CppAD: yes])
else
	AC_MSG_NOTICE([   CppAD: no])
	ac_warn_given=yes
fi

if test "$ac_have_sacado" = no
then
	AC_MSG_NOTICE([   Sacado: no])
	ac_warn_given=yes
else
	AC_MSG_NOTICE([   Sacado: yes])
fi

AC_MSG_NOTICE([********************* Top tips *************************************])
AC_MSG_NOTICE([To use a higher than default optimization level, call this configure])
AC_MSG_NOTICE([script with something like: ./configure "CXXFLAGS=-g -O3"])
AC_MSG_NOTICE([If you have libraries in non-standard locations, specify their location])
AC_MSG_NOTICE([by calling this script with something like:])
AC_MSG_NOTICE([  ./configure CPPFLAGS=-I/local/include LDFLAGS="-L/local/lib -Wl,-rpath,/local/lib"])
AC_MSG_NOTICE([The rpath argument is especially useful for locating the BLAS and LAPACK])
AC_MSG_NOTICE([libraries if they are in non-standard locations, so that executables])
AC_MSG_NOTICE([built with Adept do not need to use the LD_LIBRARY_PATH environment])
AC_MSG_NOTICE([variable to specify their locations at run-time.])
AC_MSG_NOTICE([********************************************************************])

