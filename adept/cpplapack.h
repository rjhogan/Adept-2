/* cpplapack.h -- C++ interface to LAPACK

    Copyright (C) 2015-2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptCppLapack_H
#define AdeptCppLapack_H 1                       

#include <vector>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_LAPACK

extern "C" {
  // External LAPACK Fortran functions
  void sgetrf_(const int* m, const int* n, float*  a, const int* lda, int* ipiv, int* info);
  void dgetrf_(const int* m, const int* n, double* a, const int* lda, int* ipiv, int* info);
  void sgetri_(const int* n, float* a, const int* lda, const int* ipiv, 
	       float* work, const int* lwork, int* info);
  void dgetri_(const int* n, double* a, const int* lda, const int* ipiv, 
	       double* work, const int* lwork, int* info);
  void ssytrf_(const char* uplo, const int* n, float* a, const int* lda, int* ipiv,
	       float* work, const int* lwork, int* info);
  void dsytrf_(const char* uplo, const int* n, double* a, const int* lda, int* ipiv,
	       double* work, const int* lwork, int* info);
  void ssytri_(const char* uplo, const int* n, float* a, const int* lda, 
	       const int* ipiv, float* work, int* info);
  void dsytri_(const char* uplo, const int* n, double* a, const int* lda, 
	       const int* ipiv, double* work, int* info);
  void ssysv_(const char* uplo, const int* n, const int* nrhs, float* a, const int* lda, 
	      int* ipiv, float* b, const int* ldb, float* work, const int* lwork, int* info);
  void dsysv_(const char* uplo, const int* n, const int* nrhs, double* a, const int* lda, 
	      int* ipiv, double* b, const int* ldb, double* work, const int* lwork, int* info);
  void sgesv_(const int* n, const int* nrhs, float* a, const int* lda, 
	      int* ipiv, float* b, const int* ldb, int* info);
  void dgesv_(const int* n, const int* nrhs, double* a, const int* lda, 
	      int* ipiv, double* b, const int* ldb, int* info);
}

namespace adept {

  // Overloaded functions provide both single &
  // double precision versions, and prevents the huge lapacke.h having
  // to be included in all user code
  namespace internal {
    typedef int lapack_int;
    // Factorize a general matrix
    inline
    int cpplapack_getrf(int n, float* a,  int lda, int* ipiv) {
      int info;
      sgetrf_(&n, &n, a, &lda, ipiv, &info);
      return info;
    }
    inline
    int cpplapack_getrf(int n, double* a, int lda, int* ipiv) {
      int info;
      dgetrf_(&n, &n, a, &lda, ipiv, &info);
      return info;
    }

    // Invert a general matrix
    inline
    int cpplapack_getri(int n, float* a,  int lda, const int* ipiv) {
      int info;
      float work_query;
      int lwork = -1;
      // Find out how much work memory required
      sgetri_(&n, a, &lda, ipiv, &work_query, &lwork, &info);
      lwork = static_cast<int>(work_query);
      std::vector<float> work(static_cast<size_t>(lwork));
      // Do full calculation
      sgetri_(&n, a, &lda, ipiv, &work[0], &lwork, &info);
      return info;
    }
    inline
    int cpplapack_getri(int n, double* a,  int lda, const int* ipiv) {
      int info;
      double work_query;
      int lwork = -1;
      // Find out how much work memory required
      dgetri_(&n, a, &lda, ipiv, &work_query, &lwork, &info);
      lwork = static_cast<int>(work_query);
      std::vector<double> work(static_cast<size_t>(lwork));
      // Do full calculation
      dgetri_(&n, a, &lda, ipiv, &work[0], &lwork, &info);
      return info;
    }

    // Factorize a symmetric matrix
    inline
    int cpplapack_sytrf(char uplo, int n, float* a, int lda, int* ipiv) {
      int info;
      float work_query;
      int lwork = -1;
      // Find out how much work memory required
      ssytrf_(&uplo, &n, a, &lda, ipiv, &work_query, &lwork, &info);
      lwork = static_cast<int>(work_query);
      std::vector<float> work(static_cast<size_t>(lwork));
      // Do full calculation
      ssytrf_(&uplo, &n, a, &lda, ipiv, &work[0], &lwork, &info);
      return info;
    }
    inline
    int cpplapack_sytrf(char uplo, int n, double* a, int lda, int* ipiv) {
      int info;
      double work_query;
      int lwork = -1;
      // Find out how much work memory required
      dsytrf_(&uplo, &n, a, &lda, ipiv, &work_query, &lwork, &info);
      lwork = static_cast<int>(work_query);
      std::vector<double> work(static_cast<size_t>(lwork));
      // Do full calculation
      dsytrf_(&uplo, &n, a, &lda, ipiv, &work[0], &lwork, &info);
      return info;
    }

    // Invert a symmetric matrix
    inline
    int cpplapack_sytri(char uplo, int n, float* a, int lda, const int* ipiv) {
      int info;
      std::vector<float> work(n);
      ssytri_(&uplo, &n, a, &lda, ipiv, &work[0], &info);
      return info;
    }
    inline
    int cpplapack_sytri(char uplo, int n, double* a, int lda, const int* ipiv) {
      int info;
      std::vector<double> work(n);
      dsytri_(&uplo, &n, a, &lda, ipiv, &work[0], &info);
      return info;
    }

    // Solve system of linear equations with general matrix
    inline
    int cpplapack_gesv(int n, int nrhs, float* a, int lda,
		       int* ipiv, float* b, int ldb) {
      int info;
      sgesv_(&n, &nrhs, a, &lda, ipiv, b, &lda, &info);
      return info;
    }
    inline
    int cpplapack_gesv(int n, int nrhs, double* a, int lda,
		       int* ipiv, double* b, int ldb) {
      int info;
      dgesv_(&n, &nrhs, a, &lda, ipiv, b, &lda, &info);
      return info;
    }

    // Solve system of linear equations with symmetric matrix
    inline
    int cpplapack_sysv(char uplo, int n, int nrhs, float* a, int lda, int* ipiv,
		       float* b, int ldb) {
      int info;
      float work_query;
      int lwork = -1;
      // Find out how much work memory required
      ssysv_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &work_query, &lwork, &info);
      lwork = static_cast<int>(work_query);
      std::vector<float> work(static_cast<size_t>(lwork));
      // Do full calculation
      ssysv_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &work[0], &lwork, &info);
      return info;
    }
    inline
    int cpplapack_sysv(char uplo, int n, int nrhs, double* a, int lda, int* ipiv,
		       double* b, int ldb) {
      int info;
      double work_query;
      int lwork = -1;
      // Find out how much work memory required
      dsysv_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &work_query, &lwork, &info);
      lwork = static_cast<int>(work_query);
      std::vector<double> work(static_cast<size_t>(lwork));
      // Do full calculation
      dsysv_(&uplo, &n, &nrhs, a, &lda, ipiv, b, &ldb, &work[0], &lwork, &info);
      return info;
    }

  }
}

#endif

#endif
