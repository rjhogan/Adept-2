/* cppblas.h -- C++ interface to BLAS functions

    Copyright (C) 2015-2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

   This file provides a C++ interface to selected Level-2 and -3 BLAS
   functions in which the precision of the arguments (float versus
   double) is inferred via overloading

*/

#ifndef AdeptCppBlas_H
#define AdeptCppBlas_H 1

namespace adept {

  namespace internal {

    typedef bool BLAS_ORDER;
    typedef char BLAS_TRANSPOSE;
    typedef char BLAS_UPLO;
    typedef char BLAS_SIDE;

    static const BLAS_ORDER     BlasRowMajor  = false;
    static const BLAS_ORDER     BlasColMajor  = true;
    static const BLAS_TRANSPOSE BlasNoTrans   = 'N';
    static const BLAS_TRANSPOSE BlasTrans     = 'T';
    static const BLAS_TRANSPOSE BlasConjTrans = 'C';
    static const BLAS_UPLO      BlasUpper     = 'U';
    static const BLAS_UPLO      BlasLower     = 'L';
    static const BLAS_SIDE      BlasLeft      = 'L';
    static const BLAS_SIDE      BlasRight     = 'R';

    // Matrix-matrix multiplication for general dense matrices
#define ADEPT_DEFINE_GEMM(T)					\
    void cppblas_gemm(const BLAS_ORDER Order,			\
		      const BLAS_TRANSPOSE TransA,		\
		      const BLAS_TRANSPOSE TransB,		\
		      const int M, const int N,			\
		      const int K, const T alpha, const T *A,	\
		      const int lda, const T *B, const int ldb,	\
		      const T beta, T *C, const int ldc);
    ADEPT_DEFINE_GEMM(double);
    ADEPT_DEFINE_GEMM(float);
#undef ADEPT_DEFINE_GEMM
    
    // Matrix-vector multiplication for a general dense matrix
#define ADEPT_DEFINE_GEMV(T)						\
    void cppblas_gemv(const BLAS_ORDER order,			\
		      const BLAS_TRANSPOSE TransA,		\
		      const int M, const int N,				\
		      const T alpha, const T *A, const int lda,		\
		      const T *X, const int incX, const T beta,		\
		      T *Y, const int incY);
    ADEPT_DEFINE_GEMV(double);
    ADEPT_DEFINE_GEMV(float);
#undef ADEPT_DEFINE_GEMV
    
    // Matrix-matrix multiplication where matrix A is symmetric
#define ADEPT_DEFINE_SYMM(T)						\
    void cppblas_symm(const BLAS_ORDER Order,			\
		      const BLAS_SIDE Side,			\
		      const BLAS_UPLO Uplo,			\
		      const int M, const int N,				\
		      const T alpha, const T *A, const int lda,		\
		      const T *B, const int ldb, const T beta,		\
		      T *C, const int ldc);
    ADEPT_DEFINE_SYMM(double);
    ADEPT_DEFINE_SYMM(float);
#undef ADEPT_DEFINE_SYMM
    
    // Matrix-vector multiplication where the matrix is symmetric
#define ADEPT_DEFINE_SYMV(T)						\
    void cppblas_symv(const BLAS_ORDER order,			\
		      const BLAS_UPLO Uplo,			\
		      const int N, const T alpha, const T *A,		\
		      const int lda, const T *X, const int incX,	\
		      const T beta, T *Y, const int incY);
    ADEPT_DEFINE_SYMV(double);
    ADEPT_DEFINE_SYMV(float);
#undef ADEPT_DEFINE_SYMV
    
    // Matrix-vector multiplication for a general band matrix
#define ADEPT_DEFINE_GBMV(T)						\
    void cppblas_gbmv(const BLAS_ORDER order,			\
		      const BLAS_TRANSPOSE TransA,		\
		      const int M, const int N,				\
		      const int KL, const int KU, const T alpha,	\
		      const T *A, const int lda, const T *X,		\
		      const int incX, const T beta, T *Y,		\
		      const int incY);
    ADEPT_DEFINE_GBMV(double);
    ADEPT_DEFINE_GBMV(float);
#undef ADEPT_DEFINE_GBMV

  } // End namespace internal

} // End namespace adept


#endif
