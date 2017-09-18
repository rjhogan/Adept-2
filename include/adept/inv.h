/* inv.h -- Invert matrices

    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/
                             
#ifndef AdeptInv_H
#define AdeptInv_H 1

#include <vector>

#include <adept/Array.h>
#include <adept/SpecialMatrix.h>

namespace adept {

  // -------------------------------------------------------------------
  // Invert general square matrix A
  // -------------------------------------------------------------------
  template <typename Type>
  Array<2,Type,false> 
  inv(const Array<2,Type,false>& A);

  // -------------------------------------------------------------------
  // Invert symmetric matrix A
  // -------------------------------------------------------------------
  template <typename Type, SymmMatrixOrientation Orient>
  SpecialMatrix<Type,SymmEngine<Orient>,false> 
  inv(const SpecialMatrix<Type,SymmEngine<Orient>,false>& A);
 
  // -------------------------------------------------------------------
  // Invert arbitrary expression
  // -------------------------------------------------------------------
  template <typename Type, class E>
  typename internal::enable_if<E::rank==2 && E::is_active==false
			       && internal::matrix_op_defined<Type>::value,
			       Array<2,Type,false> >::type
  inv(const Expression<Type,E>& A) {
    Array<2,Type,false> array = A.cast();
    return inv(array);
  };
 
}

#endif
