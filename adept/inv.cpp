/* inv.cpp -- Invert matrices

    Copyright (C) 2015-2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/
                             
#include <vector>

#include <adept/Array.h>
#include <adept/SpecialMatrix.h>

#ifndef AdeptSource_H
#include "cpplapack.h"
#endif

#ifdef HAVE_LAPACK

namespace adept {

  // -------------------------------------------------------------------
  // Invert general square matrix A
  // -------------------------------------------------------------------
  template <typename Type>
  Array<2,Type,false> 
  inv(const Array<2,Type,false>& A) {
    using internal::cpplapack_getrf;
    using internal::cpplapack_getri;

    if (A.dimension(0) != A.dimension(1)) {
      throw invalid_operation("Only square matrices can be inverted"
			      ADEPT_EXCEPTION_LOCATION);
    }

    Array<2,Type,false> A_;

    // LAPACKE is more efficient with column-major input
    A_.resize_column_major(A.dimensions());
    A_ = A;

    std::vector<lapack_int> ipiv(A_.dimension(0));

    //    lapack_int status = LAPACKE_dgetrf(LAPACK_COL_MAJOR, A_.dimension(0), A_.dimension(1),
    //				       A_.data(), A_.offset(1), &ipiv[0]);

    lapack_int status = cpplapack_getrf(A_.dimension(0),
					A_.data(), A_.offset(1), &ipiv[0]);
    if (status != 0) {
      std::stringstream s;
      s << "Failed to factorize matrix: LAPACK ?getrf returned code " << status;
      throw(matrix_ill_conditioned(s.str() ADEPT_EXCEPTION_LOCATION));
    }

    //    status = LAPACKE_dgetri(LAPACK_COL_MAJOR, A_.dimension(0),
    //			    A_.data(), A_.offset(1), &ipiv[0]);
    status = cpplapack_getri(A_.dimension(0),
			     A_.data(), A_.offset(1), &ipiv[0]);

    if (status != 0) {
      std::stringstream s;
      s << "Failed to invert matrix: LAPACK ?getri returned code " << status;
      throw(matrix_ill_conditioned(s.str() ADEPT_EXCEPTION_LOCATION));
    }
    return A_;
  }



  // -------------------------------------------------------------------
  // Invert symmetric matrix A
  // -------------------------------------------------------------------
  template <typename Type, SymmMatrixOrientation Orient>
  SpecialMatrix<Type,SymmEngine<Orient>,false> 
  inv(const SpecialMatrix<Type,SymmEngine<Orient>,false>& A) {
    using internal::cpplapack_sytrf;
    using internal::cpplapack_sytri;

    SpecialMatrix<Type,SymmEngine<Orient>,false> A_;

    A_.resize(A.dimension());
    A_ = A;

    // Treat symmetric matrix as column-major
    char uplo;
    if (Orient == ROW_LOWER_COL_UPPER) {
      uplo = 'U';
    }
    else {
      uplo = 'L';
    }

    std::vector<lapack_int> ipiv(A_.dimension(0));

    //    lapack_int status = LAPACKE_dsytrf(LAPACK_COL_MAJOR, uplo, A_.dimension(),
    //				       A_.data(), A_.offset(), &ipiv[0]);
    lapack_int status = cpplapack_sytrf(uplo, A_.dimension(),
					A_.data(), A_.offset(), &ipiv[0]);
    if (status != 0) {
      std::stringstream s;
      s << "Failed to factorize symmetric matrix: LAPACK ?sytrf returned code " << status;
      throw(matrix_ill_conditioned(s.str() ADEPT_EXCEPTION_LOCATION));
    }

    //    status = LAPACKE_dsytri(LAPACK_COL_MAJOR, uplo, A_.dimension(),
    //			    A_.data(), A_.offset(), &ipiv[0]);
    status = cpplapack_sytri(uplo, A_.dimension(),
			     A_.data(), A_.offset(), &ipiv[0]);
    if (status != 0) {
      std::stringstream s;
      s << "Failed to invert symmetric matrix: LAPACK ?sytri returned code " << status;
      throw(matrix_ill_conditioned(s.str() ADEPT_EXCEPTION_LOCATION));
    }
    return A_;
  }

}

#else // LAPACK not available
    
namespace adept {

  // -------------------------------------------------------------------
  // Invert general square matrix A
  // -------------------------------------------------------------------
  template <typename Type>
  Array<2,Type,false> 
  inv(const Array<2,Type,false>& A) {
    throw feature_not_available("Cannot invert matrix because compiled without LAPACK");
  }

  // -------------------------------------------------------------------
  // Invert symmetric matrix A
  // -------------------------------------------------------------------
  template <typename Type, SymmMatrixOrientation Orient>
  SpecialMatrix<Type,SymmEngine<Orient>,false> 
  inv(const SpecialMatrix<Type,SymmEngine<Orient>,false>& A) {
    throw feature_not_available("Cannot invert matrix because compiled without LAPACK");
  }
  
}

#endif

namespace adept {
  // -------------------------------------------------------------------
  // Explicit instantiations
  // -------------------------------------------------------------------
#define ADEPT_EXPLICIT_INV(TYPE)					\
  template Array<2,TYPE,false>						\
  inv(const Array<2,TYPE,false>& A);					\
  template SpecialMatrix<TYPE,SymmEngine<ROW_LOWER_COL_UPPER>,false>	\
  inv(const SpecialMatrix<TYPE,SymmEngine<ROW_LOWER_COL_UPPER>,false>&); \
  template SpecialMatrix<TYPE,SymmEngine<ROW_UPPER_COL_LOWER>,false>	\
  inv(const SpecialMatrix<TYPE,SymmEngine<ROW_UPPER_COL_LOWER>,false>&)

  ADEPT_EXPLICIT_INV(float);
  ADEPT_EXPLICIT_INV(double);

#undef ADEPT_EXPLICIT_INV
  
}


