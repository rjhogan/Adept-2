/* solve.h -- Solve systems of linear equations

    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/
                             
#ifndef AdeptSolve_H
#define AdeptSolve_H 1

#include <vector>

#include <adept/Array.h>
#include <adept/SpecialMatrix.h>

namespace adept {

  // -------------------------------------------------------------------
  // Solve Ax = b for general square matrix A
  // -------------------------------------------------------------------
  template <typename T>
  Array<1,T,false> 
  solve(const Array<2,T,false>& A, const Array<1,T,false>& b);

  // -------------------------------------------------------------------
  // Solve AX = B for general square matrix A and rectangular matrix B
  // -------------------------------------------------------------------
  template <typename T>
  Array<2,T,false> 
  solve(const Array<2,T,false>& A, const Array<2,T,false>& B);

  // -------------------------------------------------------------------
  // Solve Ax = b for symmetric square matrix A
  // -------------------------------------------------------------------
  template <typename T, SymmMatrixOrientation Orient>
  Array<1,T,false>
  solve(const SpecialMatrix<T,SymmEngine<Orient>,false>& A,
	const Array<1,T,false>& b);

  // -------------------------------------------------------------------
  // Solve AX = B for symmetric square matrix A
  // -------------------------------------------------------------------
  template <typename T, SymmMatrixOrientation Orient>
  Array<2,T,false>
  solve(const SpecialMatrix<T,SymmEngine<Orient>,false>& A,
	const Array<2,T,false>& B);

  // -------------------------------------------------------------------
  // Solve AX = B for symmetric square matrices A and B
  // -------------------------------------------------------------------
  // Simply copy B into a general dense matrix
  template <typename T, SymmMatrixOrientation LOrient,
    SymmMatrixOrientation ROrient>
  inline
  Array<2,T,false>
  solve(const SpecialMatrix<T,SymmEngine<LOrient>,false>& A,
	const SpecialMatrix<T,SymmEngine<ROrient>,false>& B) {
    Array<2,T,false> B_array = B;
    return solve(A,B_array);
  }

  // -------------------------------------------------------------------
  // Solve Ax = b for general expressions
  // -------------------------------------------------------------------
  template <typename LType, class L, typename RType, class R>
  typename internal::enable_if<L::rank==2 && R::rank==1
			       && !L::is_active && !R::is_active
			       && internal::matrix_op_defined<LType>::value
			       && internal::matrix_op_defined<RType>::value,
			       Array<1,typename internal::promote<LType,RType>::type,false> >::type
  solve(const Expression<LType,L>& l, const Expression<RType,R>& r) {
    typedef typename internal::promote<LType,RType>::type PType;
    Array<2,PType,false> left = l.cast();
    Array<1,PType,false> right = r.cast();
    return solve(left,right);
  }

  // -------------------------------------------------------------------
  // Solve AX = B for general expressions
  // -------------------------------------------------------------------
  template <typename LType, class L, typename RType, class R>
  typename internal::enable_if<L::rank==2 && R::rank==2
			       && !L::is_active && !R::is_active
			       && internal::matrix_op_defined<LType>::value
			       && internal::matrix_op_defined<RType>::value,
			       Array<2,typename internal::promote<LType,RType>::type,false> >::type
  solve(const Expression<LType,L>& l, const Expression<RType,R>& r) {
    typedef typename internal::promote<LType,RType>::type PType;
    Array<2,PType,false> left = l.cast();
    Array<2,PType,false> right = r.cast();
    return solve(left,right);
  } 
}

#endif
