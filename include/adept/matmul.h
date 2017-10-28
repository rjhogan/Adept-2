/* matmul.h -- Matrix multiplication capability

    Copyright (C) 2015-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/
                             
#ifndef AdeptMatmul_H
#define AdeptMatmul_H

#include <cmath>

#include <adept/Array.h>
#include <adept/SpecialMatrix.h>
#include <adept/cppblas.h>

namespace adept {

  namespace internal {

    // ---------------------------------------------------------------------
    // Helper functions for checking dimensions
    // ---------------------------------------------------------------------
    template <class L, class R>
    inline
    void
    check_inner_dimensions(const L& left, const R& right) {
      if (left.empty() || right.empty()) {
	throw empty_array("Attempt to perform matrix multiplication with empty array(s)"
			  ADEPT_EXCEPTION_LOCATION);
      }
      if (left.dimension(1) != right.dimension(0)) {
	throw inner_dimension_mismatch("Inner dimension mismatch in array multiplication"
				       ADEPT_EXCEPTION_LOCATION);
      }
    }

    template <class R>
    inline
    void
    check_inner_dimensions_sqr(Index left_dim, const R& right) {
      if (left_dim == 0 || right.empty()) {
	throw empty_array("Attempt to perform matrix multiplication with empty array(s)"
			  ADEPT_EXCEPTION_LOCATION);
      }
      if (left_dim != right.dimension(0)) {
	throw inner_dimension_mismatch("Inner dimension mismatch in array multiplication"
				       ADEPT_EXCEPTION_LOCATION);
      }
    }

    // ---------------------------------------------------------------------
    // Underlying functions
    // ---------------------------------------------------------------------

    // Dense matrix-vector multiplication
    template <typename T, bool LIsActive, bool RIsActive>
    inline
    Array<1,T,(LIsActive||RIsActive)>
    matmul_(const Array<2,T,LIsActive>& left, const Array<1,T,RIsActive>& right) {      
      static const bool is_active = LIsActive || RIsActive;

      check_inner_dimensions(left, right);

      Array<1,T,is_active> ans(left.dimension(0));

      Index stride;
      BLAS_ORDER order;
      if (!left.is_row_contiguous() && !left.is_column_contiguous()) {
	// Matrix is strided in both directions so needs to be copied
	// first
	Array<2,T,LIsActive> left_;
	left_ = left;
	return matmul_(left_, right);
      }
      else if (left.is_row_contiguous()) {
	order = BlasRowMajor;
	stride = left.offset(0);
      }
      else {
	order = BlasColMajor;
	stride = left.offset(1);
      }
      cppblas_gemv(order, BlasNoTrans, left.dimension(0), left.dimension(1), 
		   1.0, left.const_data(), stride, 
		   right.const_data(), right.offset(0), 
		   0.0, ans.data(), ans.offset(0));
      if (is_active
#ifdef ADEPT_RECORDING_PAUSABLE
	  && ADEPT_ACTIVE_STACK->is_recording()
#endif
	  ) {

	uIndex left_index = left.gradient_index();
	uIndex right_index = right.gradient_index();
	uIndex ans_index = ans.gradient_index();
	Index n = right.dimension(0);
	const ExpressionSize<2>& left_offset = left.offset();
	const ExpressionSize<1>& right_offset = right.offset();
	for (Index i = 0; i < ans.dimension(0); ++i) {
	  if (LIsActive) {
	    active_stack()->push_derivative_dependence(left_index+i*left_offset[0], 
						       right.const_data(), n, left_offset[1], right_offset[0]);
	  }
	  if (RIsActive) {
	    active_stack()->push_derivative_dependence(right_index, 
						       left.const_data()+i*left_offset[0], 
						       n, right_offset[0], left_offset[1]);
	  }
	  active_stack()->push_lhs(ans_index + i*ans.offset(0));
	}
      }

      return ans;
    }


    // Dense matrix-matrix multiplication
    template <typename T, bool LIsActive, bool RIsActive>
    inline
    Array<2,T,(LIsActive||RIsActive)>
    matmul_(const Array<2,T,LIsActive>& left, const Array<2,T,RIsActive>& right) {
      static const bool is_active = LIsActive || RIsActive;

      check_inner_dimensions(left, right);

      if (!left.is_row_contiguous() && !left.is_column_contiguous()) {
	Array<2,T,LIsActive> left_;
	left_ = left;
	if (!right.is_row_contiguous() && !right.is_column_contiguous()) {
	  Array<2,T,RIsActive> right_;
	  right_ = right;
	  return matmul_(left_, right_);
	}
	else {
	  return matmul_(left_, right);
	}
      }
      else if (!right.is_row_contiguous() && !right.is_column_contiguous()) {
	Array<2,T,RIsActive> right_;
	right_ = right;
	return matmul_(left, right_);
      }
      else {
	Index left_stride, right_stride, ans_stride;
	BLAS_TRANSPOSE left_trans, right_trans;
	BLAS_ORDER order;
	Array<2,T,is_active> ans(left.dimension(0),right.dimension(1));

	if (ans.is_row_contiguous()) {
	  order = BlasRowMajor;
	  ans_stride = ans.offset(0);
	}
	else {
	  order = BlasColMajor;
	  ans_stride = ans.offset(1);
	}
	if (left.is_row_contiguous()) {
	  left_trans = order == BlasRowMajor ? BlasNoTrans : BlasTrans;
	  left_stride = left.offset(0);
	}
	else {
	  left_trans = order == BlasColMajor ? BlasNoTrans : BlasTrans;
	  left_stride = left.offset(1);
	}
	if (right.is_row_contiguous()) {
	  right_trans = order == BlasRowMajor ? BlasNoTrans : BlasTrans;
	  right_stride = right.offset(0);
	}
	else {
	  right_trans = order == BlasColMajor ? BlasNoTrans : BlasTrans;
	  right_stride = right.offset(1);
	}
	cppblas_gemm(order, left_trans, right_trans,
		    left.dimension(0), right.dimension(1), left.dimension(1),
		    1.0, left.const_data(), left_stride,
		    right.const_data(), right_stride,
		    0.0, ans.data(), ans_stride);
	if ( (LIsActive || RIsActive)
#ifdef ADEPT_RECORDING_PAUSABLE
	    && ADEPT_ACTIVE_STACK->is_recording()
#endif
	    ) {
	  uIndex left_index = left.gradient_index();
	  uIndex right_index = right.gradient_index();
	  uIndex ans_index = ans.gradient_index();
	  Index n = right.dimension(0);
	  const ExpressionSize<2>& left_offset = left.offset();
	  const ExpressionSize<2>& right_offset = right.offset();

	  for (Index i = 0; i < ans.dimension(0); ++i) {
	    for (Index j = 0; j < ans.dimension(1); ++j) {
	      if (LIsActive) {
		active_stack()->push_derivative_dependence(left_index+i*left_offset[0], 
			   right.const_data()+j*right_offset[1], n, 
			   left_offset[1], right_offset[0]);
	      }
	      if (RIsActive) {
		active_stack()->push_derivative_dependence(right_index+j*right_offset[1], 
			   left.const_data()+i*left_offset[0], n, 
			   right_offset[0], left_offset[1]);
	      }
	      active_stack()->push_lhs(ans_index + i*ans.offset(0) + j*ans.offset(1));
	    }
	  }

	}
	return ans;
      }
    }

    // Symmetric matrix-vector multiplication
    template <bool LIsActive, typename T, bool RIsActive>
    inline
    Array<1,T,(LIsActive||RIsActive)>
    matmul_symmetric(const T* left_ptr, SymmMatrixOrientation left_orient, Index left_dim,
		     Index left_offset, uIndex left_gradient_index,
		     const Array<1,T,RIsActive>& right) {

      check_inner_dimensions_sqr(left_dim, right);

      if (LIsActive || RIsActive) {
	throw(invalid_operation("Cannot yet do matmul(SymmMatrix,Vector) when either are active"));
      }
      BLAS_UPLO uplo;
      if (left_orient == ROW_LOWER_COL_UPPER) {
	uplo = BlasLower;
      }
      else {
	uplo = BlasUpper;
      }
      Array<1,T,LIsActive||RIsActive> ans(right.dimension(0));
      cppblas_symv(BlasRowMajor, uplo, right.dimension(0), 
		   1.0, left_ptr, left_offset, 
		   right.const_data(), right.offset(0), 
		   0.0, ans.data(), ans.offset(0));
      return ans;
    }

    // Symmetric matrix-matrix multiplication
    template <bool LIsActive, typename T, bool RIsActive>
    inline
    Array<2,T,(LIsActive||RIsActive)>
    matmul_symmetric(const T* left_ptr, SymmMatrixOrientation left_orient, Index left_dim,
		     Index left_offset, uIndex left_gradient_index,
		     const Array<2,T,RIsActive>& right) {

      check_inner_dimensions_sqr(left_dim, right);

      if (LIsActive || RIsActive) {
	throw(invalid_operation("Cannot yet do matmul(SymmMatrix,Matrix) when either are active"));
      }
      if (!right.is_row_contiguous() && !right.is_column_contiguous()) {
	Array<2,T,RIsActive> right_;
	right_ = right;
	return matmul_symmetric<LIsActive>(left_ptr, left_orient, left_dim, left_offset,
					   left_gradient_index, right_);
      }
      else {
	BLAS_ORDER order;
	BLAS_UPLO uplo;
	Index right_stride, ans_stride;
	Array<2,T,LIsActive||RIsActive> ans;

	if (right.is_row_contiguous()) {
	  order = BlasRowMajor;
	  uplo = left_orient == ROW_LOWER_COL_UPPER ? BlasLower : BlasUpper;
	  right_stride = right.offset(0);
	  ans.resize_row_major(right.dimensions());
	  ans_stride = ans.offset(0);
	}
	else {
	  order = BlasColMajor;
	  uplo = left_orient == ROW_LOWER_COL_UPPER ? BlasUpper : BlasLower;
	  right_stride = right.offset(1);
	  ans.resize_column_major(right.dimensions());
	  ans_stride = ans.offset(1);
	}

	cppblas_symm(order, BlasLeft, uplo,  right.dimension(0), right.dimension(1),
		     1.0, left_ptr, left_offset, 
		     right.const_data(), right_stride, 0.0,
		     ans.data(), ans_stride);
	return ans;
      }
    }


    // Band matrix-vector multiplication
    template <bool LIsActive, typename T, bool RIsActive>
    inline
    Array<1,T,(LIsActive||RIsActive)>
    matmul_band(const T* left_ptr, MatrixStorageOrder left_order, 
		Index LDiags, Index UDiags, Index left_dim, Index left_offset,
		uIndex left_gradient_index, const Array<1,T,RIsActive>& right) {
      check_inner_dimensions_sqr(left_dim, right);

      if (LIsActive) {
	throw(invalid_operation("Cannot yet do matmul(BandMatrix,Vector) for active BandMatrix"));
      }

      BLAS_ORDER order;
      // BLAS declares the start pointer to be in the "missing data"
      // zone, so we need to subtract from the address of the top-left
      // corner of the matrix
      const T* left_start;
      if (left_order == ROW_MAJOR) {
	order = BlasRowMajor;
	left_start = left_ptr-UDiags;
      }
      else {
	order = BlasColMajor;
	left_start = left_ptr-LDiags;
      }
      Array<1,T,(LIsActive||RIsActive)> ans(right.dimension(0));
      cppblas_gbmv(order, BlasNoTrans, left_dim, left_dim, LDiags, UDiags,
		   1.0, left_start, left_offset+1,
		   right.const_data(), right.offset(0), 
		   0.0, ans.data(), ans.offset(0));
      if (RIsActive) {
	uIndex right_index = right.gradient_index();
	uIndex ans_index = ans.gradient_index();

	if (left_order == ROW_MAJOR) {
	  for (Index i = 0; i < ans.dimension(0); ++i) {
	    // Using info from BandEngine<ROW_MAJOR>::get_row_range in
	    // SpecialMatrix.h
	    Index j_start = i<LDiags ? 0 : i-LDiags;
	    Index j_end_plus_1 = i+UDiags+1>left_dim ? left_dim : i+UDiags+1;
	    Index n = j_end_plus_1 - j_start;
	    Index index_start = i*left_offset + j_start;
	    Index index_stride = 1;
	    active_stack()->push_derivative_dependence(right_index + j_start, 
						       left_ptr+index_start,
						       n, right.offset(0), index_stride);
	    active_stack()->push_lhs(ans_index + i*ans.offset(0));
	  }
	}
	else {
	  for (Index i = 0; i < ans.dimension(0); ++i) {
	    // Using info from BandEngine<COL_MAJOR>::get_row_range in
	    // SpecialMatrix.h
	    Index j_start = i<LDiags ? 0 : i-LDiags;
	    Index j_end_plus_1 = i+UDiags+1>left_dim ? left_dim : i+UDiags+1;
	    Index n = j_end_plus_1 - j_start;
	    Index index_start = i + j_start*left_offset;
	    Index index_stride = left_offset;
	    active_stack()->push_derivative_dependence(right_index + j_start, 
						       left_ptr+index_start,
						       n, right.offset(0), index_stride);
	    active_stack()->push_lhs(ans_index + i*ans.offset(0));
	  }
	}
      }
      return ans;
    }


    // Matrix-matrix multiplication with a band matrix on the left,
    // achieved by repeated matrix-vector multiplications
    template <bool LIsActive, typename T, bool RIsActive>
    inline
    Array<2,T,(LIsActive||RIsActive)>
    matmul_band(const T* left_ptr, MatrixStorageOrder left_order, 
		Index LDiags, Index UDiags, Index left_dim, Index left_offset,
		uIndex left_gradient_index, const Array<2,T,RIsActive>& right) {
      check_inner_dimensions_sqr(left_dim, right);
      if (LIsActive || RIsActive) {
	throw(invalid_operation("Cannot yet do matmul(BandMatrix,Matrix) when either are active"));
      }
      BLAS_ORDER order;
      // BLAS declares the start pointer to be in the "missing data"
      // zone, so we need to subtract from the address of the top-left
      // corner of the matrix
      const T* left_start;
      if (left_order == ROW_MAJOR) {
	order = BlasRowMajor;
	left_start = left_ptr-UDiags;
      }
      else {
	order = BlasColMajor;
	left_start = left_ptr-LDiags;
      }
      Array<2,T,(LIsActive||RIsActive)> ans(right.dimension(0),right.dimension(1));
      for (Index i = 0; i < right.dimension(1); ++i) {
	cppblas_gbmv(order, BlasNoTrans, left_dim, left_dim, LDiags, UDiags,
		     1.0, left_start, left_offset+1,
		     right.const_data()+i*right.offset(1), right.offset(0), 
		     0.0, ans.data()+i*ans.offset(1), ans.offset(0));
      }
      return ans;
    }
    

    // ---------------------------------------------------------------------
    // Versions of matmul_ implemented in terms of the underlying functions
    // ---------------------------------------------------------------------

    // Dense vector-matrix multiplication is evaluated by swapping and
    // transposing the arguments
    template <typename T, bool LIsActive, bool RIsActive>
    inline
    Array<1,T,(LIsActive||RIsActive)>
    matmul_(const Array<1,T,LIsActive>& left,
	    const Array<2,T,RIsActive>& right) {
      return matmul_(right.T(), left);
    }

    // Symmetric matrix-vector and matrix-matrix multiplication
    template <typename T, SymmMatrixOrientation LOrient, bool LIsActive, bool RIsActive, int RRank>
    inline
    Array<RRank,T,(LIsActive||RIsActive)>
    matmul_(const SpecialMatrix<T,internal::SymmEngine<LOrient>,LIsActive>& left,
	    const Array<RRank,T,RIsActive>& right) {
      return matmul_symmetric<LIsActive>(left.const_data(), LOrient, left.dimension(0),
					 left.offset(), left.gradient_index(), right);
    }

    // Vector multiplied by symmetric matrix: swap and transpose the arguments
    template <typename T, bool LIsActive, SymmMatrixOrientation ROrient, bool RIsActive>
    inline
    Array<1,T,(LIsActive||RIsActive)>
    matmul_(const Array<1,T,LIsActive>& left,
	    const SpecialMatrix<T,internal::SymmEngine<ROrient>,RIsActive>& right) {
      return matmul_symmetric<RIsActive>(right.const_data(), ROrient, 
					 right.dimension(0), right.offset(),
					 right.gradient_index(), left);
    }

    // Dense matrix multiplied by symmetric matrix: swap and transpose
    // the arguments, then transpose the result
    template <typename T, bool LIsActive, SymmMatrixOrientation ROrient, bool RIsActive>
    inline
    Array<2,T,(LIsActive||RIsActive)>
    matmul_(const Array<2,T,LIsActive>& left,
	    const SpecialMatrix<T,internal::SymmEngine<ROrient>,RIsActive>& right) {
      return matmul_symmetric<RIsActive>(right.const_data(), ROrient,
					 right.dimension(0), right.offset(),
					 right.gradient_index(), left.T()).T();
    }

    // Band matrix-vector and matrix-matrix multiplication
    template <typename T, MatrixStorageOrder LOrder, Index LDiags, Index UDiags, 
	      bool LIsActive, bool RIsActive, int RRank>
    inline
    Array<RRank,T,(LIsActive||RIsActive)>
    matmul_(const SpecialMatrix<T,internal::BandEngine<LOrder,LDiags,UDiags>,LIsActive>& left,
	    const Array<RRank,T,RIsActive>& right) {
      return matmul_band<LIsActive>(left.const_data(), LOrder, LDiags, UDiags,
				    left.dimension(0), left.offset(), left.gradient_index(), right);
    }

    // Vector multiplied by band matrix: swap and transpose the arguments
    template <typename T, bool LIsActive, MatrixStorageOrder ROrder, Index LDiags, Index UDiags,
	      bool RIsActive>
    inline
    Array<1,T,(LIsActive||RIsActive)>
    matmul_(const Array<1,T,LIsActive>& left,
	    const SpecialMatrix<T,internal::BandEngine<ROrder,LDiags,UDiags>,RIsActive>& right) {
      static const MatrixStorageOrder new_r_order = ROrder == ROW_MAJOR ? COL_MAJOR : ROW_MAJOR;
      return matmul_band<RIsActive>(right.const_data(), new_r_order, UDiags, LDiags,
				    right.dimension(0), right.offset(), right.gradient_index(), left);
    }

    // Dense matrix multiplied by band matrix: swap and transpose the
    // arguments, then transpose the result
    template <typename T, bool LIsActive, MatrixStorageOrder ROrder, Index LDiags, Index UDiags,
	      bool RIsActive>
    inline
    Array<2,T,(LIsActive||RIsActive)>
    matmul_(const Array<2,T,LIsActive>& left,
	    const SpecialMatrix<T,internal::BandEngine<ROrder,LDiags,UDiags>,RIsActive>& right) {
      static const MatrixStorageOrder new_r_order = ROrder == ROW_MAJOR ? COL_MAJOR : ROW_MAJOR;
      return matmul_band<RIsActive>(right.const_data(), new_r_order, UDiags, LDiags,
				    right.dimension(0), right.offset(), right.gradient_index(), left.T()).T();
    }


    // ---------------------------------------------------------------------
    // promote_array: helper function to change type of array and
    // convert expressions to arrays
    // ---------------------------------------------------------------------

    // If the argument is not an l-value then convert it to a dense
    // array of the same rank
    template <typename NewType, typename OldType, class A>
    inline
    typename internal::enable_if<!A::is_lvalue,Array<A::rank,NewType,A::is_active> >::type
    promote_array(const Expression<OldType,A>& arg) {
      return Array<A::rank,NewType,A::is_active>(arg);
    }

    // If the argument is a dense array then convert it to the new
    // type; this will only involve a copy of the raw data if the type
    // is changed, otherwise the new array will simply link to the old
    // one
    template <typename NewType, int Rank, typename OldType, bool IsActive>
    inline
    Array<Rank,NewType,IsActive>
    promote_array(const Array<Rank,OldType,IsActive>& arg) {
      return Array<Rank,NewType,IsActive>(const_cast<Array<Rank,OldType,IsActive>&>(arg));
    }

#ifdef ADEPT_ONLY_DIFFERENTIATE_DENSE_MATRIX_MULTIPLICATION
    // If the argument is an active special matrix then it must be
    // copied to a dense "Array" because differentiation of the
    // various types of special matrix (symmetric, band, upper, lower
    // etc) is not yet implemented.
    template <typename NewType, typename OldType, class Engine>
    inline
    Array<2,NewType,true>
    promote_array(const SpecialMatrix<OldType,Engine,true>& arg) {
      return Array<2,NewType,true>(
	   const_cast<SpecialMatrix<OldType,Engine,true>&>(arg));
    }

    // If the argument is an inactive symmetric or band matrix then
    // convert the element type; this will only involve a copy of the
    // raw data if the type is changed, otherwise the new array will
    // simply link to the old
    template <typename NewType, typename OldType, SymmMatrixOrientation Orient>
    inline
    SpecialMatrix<NewType,internal::SymmEngine<Orient>,false>
    promote_array(const SpecialMatrix<OldType,internal::SymmEngine<Orient>,false>& arg) {
      return SpecialMatrix<NewType,internal::SymmEngine<Orient>,false>(
	 const_cast<SpecialMatrix<OldType,internal::SymmEngine<Orient>,false>&>(arg));
    }
    template <typename NewType, typename OldType, 
      MatrixStorageOrder Order, Index LDiags, Index UDiags>
    inline
    SpecialMatrix<NewType,internal::BandEngine<Order,LDiags,UDiags>,false>
    promote_array(const SpecialMatrix<OldType,internal::BandEngine<Order,LDiags,UDiags>,false>& arg) {
      return SpecialMatrix<NewType,internal::BandEngine<Order,LDiags,UDiags>,false>(
	 const_cast<SpecialMatrix<OldType,internal::BandEngine<Order,LDiags,UDiags>,false>&>(arg));
    } 

    // For other special matrices (square and triangular), specific
    // matrix multiplication functions have not yet been added, so we
    // have to convert to a dense array first
    template <typename NewType, typename OldType, class Engine>
    inline
    Array<2,NewType,false>
    promote_array(const SpecialMatrix<OldType,Engine,false>& arg) {
      return Array<2,NewType,false>(
	 const_cast<SpecialMatrix<OldType,Engine,false>&>(arg));
    } 

#else
    // The following assumes that the Adept library knows how to
    // differentiate special matrices: currently it doesn't so this
    // path is likely to throw a run-time exception.
    template <typename NewType, typename OldType, class Engine, bool IsActive>
    inline
    SpecialMatrix<NewType,Engine,IsActive>
    promote_array(const SpecialMatrix<OldType,Engine,IsActive>& arg) {
      return SpecialMatrix<NewType,Engine,IsActive>(
		     const_cast<SpecialMatrix<OldType,Engine,IsActive>&>(arg));
    }
#endif

    // If the argument is a fixed array of a different type then copy it
    template <typename NewType, typename OldType, bool IsActive, Index J0,
	      Index J1, Index J2, Index J3, Index J4, Index J5, Index J6>
    inline
    typename enable_if<!is_same<NewType,OldType>::value,
		       Array<fixed_array<J0,J1,J2,J3,J4,J5,J6>::rank,
			     NewType,IsActive> >::type
    promote_array(const FixedArray<OldType,IsActive,J0,J1,J2,J3,J4,J5,J6>& arg) {
      return Array<fixed_array<J0,J1,J2,J3,J4,J5,J6>::rank, 
	NewType,IsActive>(const_cast<FixedArray<OldType,IsActive,J0,J1,J2,J3,J4,J5,J6>&>(arg));
    }

    // If the argument is a fixed array of the same type then link to it
    template <typename NewType, typename OldType, bool IsActive, Index J0, 
	      Index J1, Index J2, Index J3, Index J4, Index J5, Index J6>
    inline
    typename enable_if<is_same<NewType,OldType>::value,
		       Array<fixed_array<J0,J1,J2,J3,J4,J5,J6>::rank,
			     NewType,IsActive> >::type
    promote_array(const FixedArray<OldType,IsActive,J0,J1,J2,J3,J4,J5,J6>& arg) {
      return Array<fixed_array<J0,J1,J2,J3,J4,J5,J6>::rank,NewType,IsActive>
	(const_cast<FixedArray<OldType,IsActive,J0,J1,J2,J3,J4,J5,J6>&>(arg).data(), 0,
	 arg.dimensions(), arg.offset(), arg.gradient_index());
    }

  } // End namespace internal

  // ---------------------------------------------------------------------
  // matmul function: replicates Fortran-90 equivalent
  // ---------------------------------------------------------------------

  // If either argument is not an lvalue (i.e. is an array expression
  // rather than an array) then convert it into a dense array
  template <typename LType, class L, typename RType, class R>
  inline
  typename internal::enable_if<(L::rank == 1 || L::rank == 2) && (R::rank == 1 || R::rank == 2)
                      && (L::rank+R::rank > 2),
    Array<L::rank+R::rank-2,typename promote<LType,RType>::type,
    L::is_active||R::is_active> >::type
  matmul(const Expression<LType,L>& left, const Expression<RType,R>& right) {
    typedef typename promote<typename L::type,typename R::type>::type type;
    return internal::matmul_(internal::promote_array<type>(left.cast()),
			     internal::promote_array<type>(right.cast()));
  }
  

  // ---------------------------------------------------------------------
  // Implement "**" pseudo-operator for matrix multiplication
  // ---------------------------------------------------------------------

  // In order for A**B to lead to matrix multiplication, *B will
  // return a MatmulRHS object, and A*[a MatmulRHS object] will send
  // the two arguments to the matmul function

  namespace internal {

    // The MatmulRHS class simply contains a reference to an array
    template <class A>
    struct MatmulRHS {
      // The following are not used but enable
      // expr_cast<MatmulRHS>::... to work
      static const int  rank      = A::rank;
      static const bool is_active = A::is_active;
      static const int  n_arrays  = 0;
      static const bool n_active  = 0;
      static const bool is_lvalue = false;
      static const bool is_vectorizable = false;
      static const int  n_scratch = 0;
      // The following are necessary in order that other binary
      // operator* functions can compile, even if they are rejected
      // for a particular multiplication
      typedef typename A::type type;
      typedef bool _adept_expression_flag;
      // Constructor simply saves a reference to the expression
      // argument
      MatmulRHS(const A& a) : array(a) { }
      const A& array;
    };
  }

  // Dereference operator returns a MatmulRHS object
  template <typename Type, class A>
  inline
  typename internal::enable_if<(A::rank == 1 || A::rank == 2),
			       internal::MatmulRHS<A> >::type
  operator*(const Expression<Type,A>& a) {
    return internal::MatmulRHS<A>(a.cast());
  }

  // Multiply operator with a MatmulRHS object on the right-hand-side
  // will call the matmul function
  template <typename LType, class L, class R>
  inline
  Array<L::rank+R::rank-2,typename promote<LType,typename R::type>::type,
	(L::is_active||R::is_active)>
  operator*(const Expression<LType,L>& left, const internal::MatmulRHS<R>& right) {
    return matmul(left.cast(),right.array.cast());
  }


} // End namespace adept

#endif
