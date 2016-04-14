/* solve.cpp -- Solve systems of linear equations using LAPACK

    Copyright (C) 2015-2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/
                             

#include <vector>


#include <adept/solve.h>
#include <adept/Array.h>
#include <adept/SpecialMatrix.h>

// If ADEPT_SOURCE_H is defined then we are in a header file generated
// from all the source files, so cpplapack.h will already have been
// included
#ifndef AdeptSource_H
#include "cpplapack.h"
#endif

#ifdef HAVE_LAPACK

namespace adept {

  // -------------------------------------------------------------------
  // Solve Ax = b for general square matrix A
  // -------------------------------------------------------------------
  template <typename T>
  Array<1,T,false> 
  solve(const Array<2,T,false>& A, const Array<1,T,false>& b) {
    Array<2,T,false> A_;
    Array<1,T,false> b_;

    // LAPACKE is more efficient with column-major input
    // if (A.is_row_contiguous()) {
      A_.resize_column_major(A.dimensions());
      A_ = A;
    // }
    // else {
    //   A_.link(A);
    // }

    // if (b_.offset(0) != 0) {
      b_ = b;
    // }
    // else {
    //   b_.link(b);
    // }

    std::vector<lapack_int> ipiv(A_.dimension(0));

    //    lapack_int status = LAPACKE_dgesv(LAPACK_COL_MAJOR, A_.dimension(0), 1,
    //				      A_.data(), A_.offset(1), &ipiv[0],
    //				      b_.data(), b_.dimension(0));
    lapack_int status = cpplapack_gesv(A_.dimension(0), 1,
				       A_.data(), A_.offset(1), &ipiv[0],
				       b_.data(), b_.dimension(0));

    if (status != 0) {
      std::stringstream s;
      s << "Failed to solve general system of equations: LAPACK ?gesv returned code " << status;
      throw(matrix_ill_conditioned(s.str() ADEPT_EXCEPTION_LOCATION));
    }
    return b_;    
  }

  // -------------------------------------------------------------------
  // Solve AX = B for general square matrix A and rectangular matrix B
  // -------------------------------------------------------------------
  template <typename T>
  Array<2,T,false> 
  solve(const Array<2,T,false>& A, const Array<2,T,false>& B) {
    Array<2,T,false> A_;
    Array<2,T,false> B_;
    
    // LAPACKE is more efficient with column-major input
    // if (A.is_row_contiguous()) {
      A_.resize_column_major(A.dimensions());
      A_ = A;
    // }
    // else {
    //   A_.link(A);
    // }

    // if (B.is_row_contiguous()) {
      B_.resize_column_major(B.dimensions());
      B_ = B;
    // }
    // else {
    //   B_.link(B);
    // }

    std::vector<lapack_int> ipiv(A_.dimension(0));

    //    lapack_int status = LAPACKE_dgesv(LAPACK_COL_MAJOR, A_.dimension(0), B.dimension(1),
    //				      A_.data(), A_.offset(1), &ipiv[0],
    //				      B_.data(), B_.offset(1));
    lapack_int status = cpplapack_gesv(A_.dimension(0), B.dimension(1),
				       A_.data(), A_.offset(1), &ipiv[0],
				       B_.data(), B_.offset(1));
    if (status != 0) {
      std::stringstream s;
      s << "Failed to solve general system of equations for matrix RHS: LAPACK ?gesv returned code " << status;
      throw(matrix_ill_conditioned(s.str() ADEPT_EXCEPTION_LOCATION));
    }
    return B_;    
  }


  // -------------------------------------------------------------------
  // Solve Ax = b for symmetric square matrix A
  // -------------------------------------------------------------------
  template <typename T, SymmMatrixOrientation Orient>
  Array<1,T,false>
  solve(const SpecialMatrix<T,SymmEngine<Orient>,false>& A,
	const Array<1,T,false>& b) {
    SpecialMatrix<T,SymmEngine<Orient>,false> A_;
    Array<1,T,false> b_;

    // Not sure why the original code copies A...
    A_.resize(A.dimension());
    A_ = A;
    // A_.link(A);

    // if (b.offset(0) != 1) {
      b_ = b;
    // }
    // else {
    //   b_.link(b);
    // }

    // Treat symmetric matrix as column-major
    char uplo;
    if (Orient == ROW_LOWER_COL_UPPER) {
      uplo = 'U';
    }
    else {
      uplo = 'L';
    }

    std::vector<lapack_int> ipiv(A_.dimension());

    //    lapack_int status = LAPACKE_dsysv(LAPACK_COL_MAJOR, uplo, A_.dimension(0), 1,
    //				      A_.data(), A_.offset(), &ipiv[0],
    //				      b_.data(), b_.dimension(0));
    lapack_int status = cpplapack_sysv(uplo, A_.dimension(0), 1,
				       A_.data(), A_.offset(), &ipiv[0],
				       b_.data(), b_.dimension(0));

    if (status != 0) {
      //      std::stringstream s;
      //      s << "Failed to solve symmetric system of equations: LAPACK ?sysv returned code " << status;
      //      throw(matrix_ill_conditioned(s.str() ADEPT_EXCEPTION_LOCATION));
      std::cerr << "Warning: LAPACK solve symmetric system failed (?sysv): trying general (?gesv)\n";
      return solve(Array<2,T,false>(A_),b_);
    }
    return b_;    
  }


  // -------------------------------------------------------------------
  // Solve AX = B for symmetric square matrix A
  // -------------------------------------------------------------------
  template <typename T, SymmMatrixOrientation Orient>
  Array<2,T,false>
  solve(const SpecialMatrix<T,SymmEngine<Orient>,false>& A,
	const Array<2,T,false>& B) {
    SpecialMatrix<T,SymmEngine<Orient>,false> A_;
    Array<2,T,false> B_;

    A_.resize(A.dimension());
    A_ = A;
    // A_.link(A);

    // if (B.is_row_contiguous()) {
      B_.resize_column_major(B.dimensions());
      B_ = B;
    // }
    // else {
    //   B_.link(B);
    // }

    // Treat symmetric matrix as column-major
    char uplo;
    if (Orient == ROW_LOWER_COL_UPPER) {
      uplo = 'U';
    }
    else {
      uplo = 'L';
    }

    std::vector<lapack_int> ipiv(A_.dimension());

    //    lapack_int status = LAPACKE_dsysv(LAPACK_COL_MAJOR, uplo, A_.dimension(0), B.dimension(1),
    //				      A_.data(), A_.offset(), &ipiv[0],
    //				      B_.data(), B_.offset(1));
    lapack_int status = cpplapack_sysv(uplo, A_.dimension(0), B.dimension(1),
				       A_.data(), A_.offset(), &ipiv[0],
				       B_.data(), B_.offset(1));

    if (status != 0) {
      std::stringstream s;
      s << "Failed to solve symmetric system of equations with matrix RHS: LAPACK ?sysv returned code " << status;
      throw(matrix_ill_conditioned(s.str() ADEPT_EXCEPTION_LOCATION));
    }
    return B_;
  }

}

#else

namespace adept {
  
  // -------------------------------------------------------------------
  // Solve Ax = b for general square matrix A
  // -------------------------------------------------------------------
  template <typename T>
  Array<1,T,false> 
  solve(const Array<2,T,false>& A, const Array<1,T,false>& b) {
    throw feature_not_available("Cannot solve linear equations because compiled without LAPACK");
  }

  // -------------------------------------------------------------------
  // Solve AX = B for general square matrix A and rectangular matrix B
  // -------------------------------------------------------------------
  template <typename T>
  Array<2,T,false> 
  solve(const Array<2,T,false>& A, const Array<2,T,false>& B) {
    throw feature_not_available("Cannot solve linear equations because compiled without LAPACK");
  }

  // -------------------------------------------------------------------
  // Solve Ax = b for symmetric square matrix A
  // -------------------------------------------------------------------
  template <typename T, SymmMatrixOrientation Orient>
  Array<1,T,false>
  solve(const SpecialMatrix<T,SymmEngine<Orient>,false>& A,
	const Array<1,T,false>& b) {
    throw feature_not_available("Cannot solve linear equations because compiled without LAPACK");
  }

  // -------------------------------------------------------------------
  // Solve AX = B for symmetric square matrix A
  // -------------------------------------------------------------------
  template <typename T, SymmMatrixOrientation Orient>
  Array<2,T,false>
  solve(const SpecialMatrix<T,SymmEngine<Orient>,false>& A,
	const Array<2,T,false>& B) {
    throw feature_not_available("Cannot solve linear equations because compiled without LAPACK");
  }

}

#endif


namespace adept {

  // -------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------
#define ADEPT_EXPLICIT_SOLVE(TYPE,RRANK)				\
  template Array<RRANK,TYPE,false>					\
  solve(const Array<2,TYPE,false>& A, const Array<RRANK,TYPE,false>& b); \
  template Array<RRANK,TYPE,false>					\
  solve(const SpecialMatrix<TYPE,SymmEngine<ROW_LOWER_COL_UPPER>,false>& A, \
	const Array<RRANK,TYPE,false>& b);					\
  template Array<RRANK,TYPE,false>					\
  solve(const SpecialMatrix<TYPE,SymmEngine<ROW_UPPER_COL_LOWER>,false>& A, \
	const Array<RRANK,TYPE,false>& b);

  ADEPT_EXPLICIT_SOLVE(float,1)
  ADEPT_EXPLICIT_SOLVE(float,2)
  ADEPT_EXPLICIT_SOLVE(double,1)
  ADEPT_EXPLICIT_SOLVE(double,2)
#undef ADEPT_EXPLICIT_SOLVE

}

