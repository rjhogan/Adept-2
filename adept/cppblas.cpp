/* cppblas.cpp -- C++ interface to BLAS functions

    Copyright (C) 2015-2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

   This file provides a C++ interface to selected Level-2 and -3 BLAS
   functions in which the precision of the arguments (float versus
   double) is inferred via overloading

*/

#include <adept/exception.h>
#include <adept/cppblas.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_BLAS

extern "C" {
  void sgemm_(const char* TransA, const char* TransB, const int* M,
	      const int* N, const int* K, const float* alpha,
	      const float* A, const int* lda, const float* B, const int* ldb,
	      const float* beta, const float* C, const int* ldc);
  void dgemm_(const char* TransA, const char* TransB, const int* M,
	      const int* N, const int* K, const double* alpha,
	      const double* A, const int* lda, const double* B, const int* ldb,
	      const double* beta, const double* C, const int* ldc);
  void sgemv_(const char* TransA, const int* M, const int* N, const float* alpha,
	      const float* A, const int* lda, const float* X, const int* incX,
	      const float* beta, const float* Y, const int* incY);
  void dgemv_(const char* TransA, const int* M, const int* N, const double* alpha,
	      const double* A, const int* lda, const double* X, const int* incX,
	      const double* beta, const double* Y, const int* incY);
  void ssymm_(const char* side, const char* uplo, const int* M, const int* N,
	      const float* alpha, const float* A, const int* lda, const float* B,
	      const int* ldb, const float* beta, float* C, const int* ldc);
  void dsymm_(const char* side, const char* uplo, const int* M, const int* N,
	      const double* alpha, const double* A, const int* lda, const double* B,
	      const int* ldb, const double* beta, double* C, const int* ldc);
  void ssymv_(const char* uplo, const int* N, const float* alpha, const float* A, 
	      const int* lda, const float* X, const int* incX, const float* beta, 
	      const float* Y, const int* incY);
  void dsymv_(const char* uplo, const int* N, const double* alpha, const double* A, 
	      const int* lda, const double* X, const int* incX, const double* beta, 
	      const double* Y, const int* incY);
  void sgbmv_(const char* TransA, const int* M, const int* N, const int* kl, 
	      const int* ku, const float* alpha, const float* A, const int* lda,
	      const float* X, const int* incX, const float* beta, 
	      const float* Y, const int* incY);
  void dgbmv_(const char* TransA, const int* M, const int* N, const int* kl, 
	      const int* ku, const double* alpha, const double* A, const int* lda,
	      const double* X, const int* incX, const double* beta, 
	      const double* Y, const int* incY);
};

namespace adept {

  namespace internal {
    
    // Matrix-matrix multiplication for general dense matrices
#define ADEPT_DEFINE_GEMM(T, FUNC, FUNC_COMPLEX)		\
    void cppblas_gemm(BLAS_ORDER Order,				\
		      BLAS_TRANSPOSE TransA,			\
		      BLAS_TRANSPOSE TransB,			\
		      int M, int N,				\
		      int K, T alpha, const T *A,		\
		      int lda, const T *B, int ldb,		\
		      T beta, T *C, int ldc) {			\
      if (Order == BlasColMajor) {				\
        FUNC(&TransA, &TransB, &M, &N, &K, &alpha, A, &lda,	\
	     B, &ldb, &beta, C, &ldc);				\
      }								\
      else {							\
        FUNC(&TransB, &TransA, &N, &M, &K, &alpha, B, &ldb,	\
	     A, &lda, &beta, C, &ldc);				\
      }								\
    }
    ADEPT_DEFINE_GEMM(double, dgemm_, zgemm_);
    ADEPT_DEFINE_GEMM(float,  sgemm_, cgemm_);
#undef ADEPT_DEFINE_GEMM
    
    // Matrix-vector multiplication for a general dense matrix
#define ADEPT_DEFINE_GEMV(T, FUNC, FUNC_COMPLEX)		\
    void cppblas_gemv(const BLAS_ORDER Order,			\
		      const BLAS_TRANSPOSE TransA,		\
		      const int M, const int N,			\
		      const T alpha, const T *A, const int lda,	\
		      const T *X, const int incX, const T beta,	\
		      T *Y, const int incY) {			\
      if (Order == BlasColMajor) {				\
        FUNC(&TransA, &M, &N, &alpha, A, &lda, X, &incX, 	\
	     &beta, Y, &incY);					\
      }								\
      else {							\
        BLAS_TRANSPOSE TransNew					\
	  = TransA == BlasTrans ? BlasNoTrans : BlasTrans;	\
        FUNC(&TransNew, &N, &M, &alpha, A, &lda, X, &incX, 	\
	     &beta, Y, &incY);					\
      }								\
    }
    ADEPT_DEFINE_GEMV(double, dgemv_, zgemv_);
    ADEPT_DEFINE_GEMV(float,  sgemv_, cgemv_);
#undef ADEPT_DEFINE_GEMV
    
    // Matrix-matrix multiplication where matrix A is symmetric
    // FIX! CHECK ROW MAJOR VERSION IS RIGHT			
#define ADEPT_DEFINE_SYMM(T, FUNC, FUNC_COMPLEX)			\
    void cppblas_symm(const BLAS_ORDER Order,				\
		      const BLAS_SIDE Side,				\
		      const BLAS_UPLO Uplo,				\
		      const int M, const int N,				\
		      const T alpha, const T *A, const int lda,		\
		      const T *B, const int ldb, const T beta,		\
		      T *C, const int ldc) {				\
      if (Order == BlasColMajor) {					\
        FUNC(&Side, &Uplo, &M, &N, &alpha, A, &lda,			\
	     B, &ldb, &beta, C, &ldc);					\
      }									\
      else {								\
	BLAS_SIDE SideNew = Side == BlasLeft  ? BlasRight : BlasLeft;	\
	BLAS_UPLO UploNew = Uplo == BlasUpper ? BlasLower : BlasUpper;  \
        FUNC(&SideNew, &UploNew, &N, &M, &alpha, A, &lda,		\
	     B, &ldb, &beta, C, &ldc);					\
      }									\
    }
    ADEPT_DEFINE_SYMM(double, dsymm_, zsymm_);
    ADEPT_DEFINE_SYMM(float,  ssymm_, csymm_);
#undef ADEPT_DEFINE_SYMM
    
    // Matrix-vector multiplication where the matrix is symmetric
#define ADEPT_DEFINE_SYMV(T, FUNC, FUNC_COMPLEX)			\
    void cppblas_symv(const BLAS_ORDER Order,				\
		      const BLAS_UPLO Uplo,				\
		      const int N, const T alpha, const T *A,		\
		      const int lda, const T *X, const int incX,	\
		      const T beta, T *Y, const int incY) {		\
      if (Order == BlasColMajor) {					\
        FUNC(&Uplo, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY);	\
      }									\
      else {								\
        BLAS_UPLO UploNew = Uplo == BlasUpper ? BlasLower : BlasUpper;  \
        FUNC(&UploNew, &N, &alpha, A, &lda, X, &incX, &beta, Y, &incY);	\
      }									\
    }
    ADEPT_DEFINE_SYMV(double, dsymv_, zsymv_);
    ADEPT_DEFINE_SYMV(float,  ssymv_, csymv_);
#undef ADEPT_DEFINE_SYMV
    
    // Matrix-vector multiplication for a general band matrix
#define ADEPT_DEFINE_GBMV(T, FUNC, FUNC_COMPLEX)		\
    void cppblas_gbmv(const BLAS_ORDER Order,			\
		      const BLAS_TRANSPOSE TransA,		\
		      const int M, const int N,			\
		      const int KL, const int KU, const T alpha,\
		      const T *A, const int lda, const T *X,	\
		      const int incX, const T beta, T *Y,	\
		      const int incY) {				\
      if (Order == BlasColMajor) {				\
        FUNC(&TransA, &M, &N, &KL, &KU, &alpha, A, &lda,	\
	     X, &incX, &beta, Y, &incY);			\
      }								\
      else {							\
	BLAS_TRANSPOSE TransNew					\
	  = TransA == BlasTrans ? BlasNoTrans : BlasTrans;	\
	FUNC(&TransNew, &N, &M, &KU, &KL, &alpha, A, &lda,	\
	     X, &incX, &beta, Y, &incY);			\
      }								\
    }
    ADEPT_DEFINE_GBMV(double, dgbmv_, zgbmv_);
    ADEPT_DEFINE_GBMV(float,  sgbmv_, cgbmv_);
#undef ADEPT_DEFINE_GBMV
  
  } // End namespace internal
  
} // End namespace adept
  

#else // Don't have BLAS


namespace adept {

  namespace internal {
    
    // Matrix-matrix multiplication for general dense matrices
#define ADEPT_DEFINE_GEMM(T, FUNC, FUNC_COMPLEX)		\
    void cppblas_gemm(BLAS_ORDER Order,				\
		      BLAS_TRANSPOSE TransA,			\
		      BLAS_TRANSPOSE TransB,			\
		      int M, int N,				\
		      int K, T alpha, const T *A,		\
		      int lda, const T *B, int ldb,		\
		      T beta, T *C, int ldc) {			\
      throw feature_not_available("Cannot perform matrix-matrix multiplication because compiled without BLAS"); \
    }
    ADEPT_DEFINE_GEMM(double, dgemm_, zgemm_);
    ADEPT_DEFINE_GEMM(float,  sgemm_, cgemm_);
#undef ADEPT_DEFINE_GEMM
    
    // Matrix-vector multiplication for a general dense matrix
#define ADEPT_DEFINE_GEMV(T, FUNC, FUNC_COMPLEX)		\
    void cppblas_gemv(const BLAS_ORDER Order,			\
		      const BLAS_TRANSPOSE TransA,		\
		      const int M, const int N,			\
		      const T alpha, const T *A, const int lda,	\
		      const T *X, const int incX, const T beta,	\
		      T *Y, const int incY) {			\
      throw feature_not_available("Cannot perform matrix-vector multiplication because compiled without BLAS"); \
    }
    ADEPT_DEFINE_GEMV(double, dgemv_, zgemv_);
    ADEPT_DEFINE_GEMV(float,  sgemv_, cgemv_);
#undef ADEPT_DEFINE_GEMV
    
    // Matrix-matrix multiplication where matrix A is symmetric
    // FIX! CHECK ROW MAJOR VERSION IS RIGHT			
#define ADEPT_DEFINE_SYMM(T, FUNC, FUNC_COMPLEX)			\
    void cppblas_symm(const BLAS_ORDER Order,				\
		      const BLAS_SIDE Side,				\
		      const BLAS_UPLO Uplo,				\
		      const int M, const int N,				\
		      const T alpha, const T *A, const int lda,		\
		      const T *B, const int ldb, const T beta,		\
		      T *C, const int ldc) {				\
      throw feature_not_available("Cannot perform symmetric matrix-matrix multiplication because compiled without BLAS"); \
    }
    ADEPT_DEFINE_SYMM(double, dsymm_, zsymm_);
    ADEPT_DEFINE_SYMM(float,  ssymm_, csymm_);
#undef ADEPT_DEFINE_SYMM
    
    // Matrix-vector multiplication where the matrix is symmetric
#define ADEPT_DEFINE_SYMV(T, FUNC, FUNC_COMPLEX)			\
    void cppblas_symv(const BLAS_ORDER Order,				\
		      const BLAS_UPLO Uplo,				\
		      const int N, const T alpha, const T *A,		\
		      const int lda, const T *X, const int incX,	\
		      const T beta, T *Y, const int incY) {		\
      throw feature_not_available("Cannot perform symmetric matrix-vector multiplication because compiled without BLAS"); \
    }
    ADEPT_DEFINE_SYMV(double, dsymv_, zsymv_);
    ADEPT_DEFINE_SYMV(float,  ssymv_, csymv_);
#undef ADEPT_DEFINE_SYMV
    
    // Matrix-vector multiplication for a general band matrix
#define ADEPT_DEFINE_GBMV(T, FUNC, FUNC_COMPLEX)		\
    void cppblas_gbmv(const BLAS_ORDER Order,			\
		      const BLAS_TRANSPOSE TransA,		\
		      const int M, const int N,			\
		      const int KL, const int KU, const T alpha,\
		      const T *A, const int lda, const T *X,	\
		      const int incX, const T beta, T *Y,	\
		      const int incY) {				\
      throw feature_not_available("Cannot perform band matrix-vector multiplication because compiled without BLAS"); \
    }
    ADEPT_DEFINE_GBMV(double, dgbmv_, zgbmv_);
    ADEPT_DEFINE_GBMV(float,  sgbmv_, cgbmv_);
#undef ADEPT_DEFINE_GBMV

  }
}

#endif
