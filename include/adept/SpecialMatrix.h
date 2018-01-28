/* SpecialMatrix.h -- Active or inactive symmetric and band-diagonal matrices

    Copyright (C) 2015-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   The SpecialMatrix is the basis for a wide range of matrix types
   such as SquareMatrix, DiagonalMatrix, TridiagonalMatrix,
   SymmetricMatrix etc.

*/

#ifndef AdeptSpecialMatrix_H
#define AdeptSpecialMatrix_H 1

#include <iostream>
#include <sstream>
#include <limits>

#include <adept/base.h>
#include <adept/Storage.h>
#include <adept/Expression.h>
#include <adept/RangeIndex.h>
#include <adept/ActiveReference.h>
#include <adept/Array.h>
#include <adept/FixedArray.h>

namespace adept {
  using namespace adept::internal;

  namespace internal {

    // -------------------------------------------------------------------
    // SpecialMatrix Engine helper classes
    // -------------------------------------------------------------------
    enum SymmMatrixOrientation {
      ROW_LOWER_COL_UPPER=0, ROW_UPPER_COL_LOWER=1
    };

    // -------------------------------------------------------------------
    // Conventional matrix storage engine
    // -------------------------------------------------------------------

    // The SpecialMatrix class is assisted by data-free policy classes
    // that define the behaviour of different matrix types. The first
    // most basic one is for square matrices. Comments are provided
    // for the first one only to explain the meaning of each
    // function. The default here is ROW_MAJOR; the alternative
    // COL_MAJOR is provided as a specialization of this class.
    template <MatrixStorageOrder Order>
    struct SquareEngine {
      // The number of variables to store for a SpecialMatrix when it
      // is on the right-hand-side of an expression for its location
      static const int my_n_arrays = 1;
      // Used by SpecialMatrix::expression_string() to describe the
      // matrix type
      const char* name() const { return "SquareMatrix"; }
      // Used by SpecialMatrix::info_string() to describe the matrix
      // type
      std::string long_name() const { return "SquareMatrix<ROW_MAJOR>"; }
      // The offset to use (the spacing in memory of elements along
      // the slowest varying dimension) for "packed" data, i.e. when
      // this matrix is created by the SpecialMatrix::resize function
      // rather than being a submatrix to something larger.
      Index pack_offset(Index dim) const { return dim; }
      // Provide the memory index to the element at row i, column j
      Index index(Index i, Index j, Index offset) const {
	return i*offset + j;
      }
      // When traversing along a row, this is the separation in memory
      // of each element
      template <int MyArrayNum, int NArrays>
      Index row_offset(Index offset, const ExpressionSize<NArrays>& loc) const {
	return 1; 
      }
      // This function is used when a SpecialMatrix is used on the
      // left-hand-side of an expression. For row i, return the range
      // of columns containing unique elements in j_start and
      // j_end_plus_1, the memory location of the element
      // corresponding to j_start in index_start, and the separation
      // in memory of consecutive elements in this range
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = 0;
	j_end_plus_1 = dim;
	index_start = i*offset;
	index_stride = 1;
      }
      // Return value at row i, column j as an rvalue, first in the
      // case of an inactive array...
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type>::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		 Index gradient_index, const Type* data) const {
	return data[index(i,j,offset)]; 
      }
      // ...now in the case of an active array.
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,Active<Type> >::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, const Type* data) const {
	return Active<Type>(data[index(i,j,offset)]);
      }
      // Return value at row i, column j as an lvalue, first in the
      // case of an inactive array...
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type&>::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	return data[index(i,j,offset)]; 
      }
      // ...now in the case of an active array.
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,ActiveReference<Type> >::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	Index ind = index(i,j,offset);
	return ActiveReference<Type>(data[ind], gradient_index+ind);
      }
      // Return the number of elements stored for a SpecialMatrix of
      // size dim x dim.  This is used both by SpecialMatrix::resize
      // to know how much memory to allocate, and by
      // SpecialMatrix::is_aliased to know the memory range spanned by
      // the object.
      Index data_size(Index dim, Index offset) const {
	return (dim-1)*offset+dim;
      }
      // Memory offset of start of a superdiagonal (offdiag > 0)
      Index upper_offset(Index dim, Index offset, Index offdiag) const {
	return offdiag;
      }
      // Memory offset of start of a subdiagonal (offdiag < 0)
      Index lower_offset(Index dim, Index offset, Index offdiag) const {
	return -offdiag*offset;
      }
      // Check super- and sub-diagonals are in range, otherwise throw
      // an exception (errors only thrown for band matrices)
      void check_upper_diag(Index offdiag) const { }
      void check_lower_diag(Index offdiag) const { }
      // The type returned by the transpose .T() member function
      typedef SquareEngine<COL_MAJOR> transpose_engine;
      // Extra info to store when traversing a SpecialMatrix on the
      // right-hand-side of an expression
      template <int MyArrayNum, int NArrays>
      void set_extras(Index i, Index offset,
		      ExpressionSize<NArrays>& index) const { }
      // Return the value at the specified location in memory
      template <int MyArrayNum, int NArrays, typename Type>
      Type value_at_location(const Type* data, 
			     const ExpressionSize<NArrays>& loc) const {
	return data[loc[MyArrayNum]];
      }
      // Push an element of an active SpecialMatrix onto the stack
      template <int MyArrayNum, int NArrays, typename Type>
      void push_rhs(Stack& stack, Type multiplier, Index gradient_index,
		    const ExpressionSize<NArrays>& loc) const {
	stack.push_rhs(multiplier, gradient_index + loc[MyArrayNum]);
      }
    };

    // The engine for the SquareMatrix type using column-major
    // storage; note that this inherits from the row-major version in
    // order that functions that don't need to be changed can be
    // imported using "using".
    template <>
    struct SquareEngine<COL_MAJOR> : public SquareEngine<ROW_MAJOR> {
      static const int my_n_arrays = 1;
      const char* name() const { return "SquareMatrix"; }
      std::string long_name() const { return "SquareMatrix<COL_MAJOR>"; }
      Index pack_offset(Index dim) const { return dim; }
      Index index(Index i, Index j, Index offset) const {
	return i + j*offset;
      }
      template <int MyArrayNum, int NArrays>
      Index row_offset(Index offset, const ExpressionSize<NArrays>& loc) const {
	return offset; 
      }
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = 0;
	j_end_plus_1 = dim;
	index_start = i;
	index_stride = offset;
      }

      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type>::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		 Index gradient_index, const Type* data) const {
	return data[index(i,j,offset)]; 
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,Active<Type> >::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, const Type* data) const {
	return Active<Type>(data[index(i,j,offset)]);
      }
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type&>::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	return data[index(i,j,offset)]; 
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,ActiveReference<Type> >::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	Index ind = index(i,j,offset);
	return ActiveReference<Type>(data[ind], gradient_index+ind);
      }
      Index upper_offset(Index dim, Index offset, Index offdiag) const {
	return offdiag*offset;
      }
      Index lower_offset(Index dim, Index offset, Index offdiag) const {
	return -offdiag;
      }
      typedef SquareEngine<ROW_MAJOR> transpose_engine;
      using SquareEngine<ROW_MAJOR>::data_size;
      using SquareEngine<ROW_MAJOR>::check_upper_diag;
      using SquareEngine<ROW_MAJOR>::check_lower_diag;
      using SquareEngine<ROW_MAJOR>::set_extras;
      using SquareEngine<ROW_MAJOR>::value_at_location;
      using SquareEngine<ROW_MAJOR>::push_rhs;
    };

    // -------------------------------------------------------------------
    // Band matrix storage engine
    // -------------------------------------------------------------------

    // A band matrix uses the BLAS packed storage to store LDiags
    // subdiagonals and UDiags superdiagonals; the default version
    // uses row-major storage
    template <Index LDiags, Index UDiags>
    struct BandEngineHelper {
      const char* name() const { return "BandMatrix"; }
    };
    template <>
    struct BandEngineHelper<0,0> {
      const char* name() const { return "DiagMatrix"; }
    };
    template <>
    struct BandEngineHelper<1,1> {
      const char* name() const { return "TridiagMatrix"; }
    };
    template <>
    struct BandEngineHelper<2,2> {
      const char* name() const { return "PentadiagMatrix"; }
    };

    template <MatrixStorageOrder Order, Index LDiags, Index UDiags>
    struct BandEngine {
      static const int my_n_arrays = 3;
      static const Index diagonals = 1+LDiags+UDiags;
      const char* name() const { return BandEngineHelper<LDiags,UDiags>().name(); }
      std::string long_name() const { 
	std::stringstream s;
	s << "BandMatrix<ROW_MAJOR,LDiags=" << LDiags
	  << ",UDiags=" << UDiags << ">";
	return s.str();
      }
      Index pack_offset(Index dim) const { return diagonals-1; }
      Index index(Index i, Index j, Index offset) const {
	//	return LDiags + i*offset + j;
	return i*offset + j;
      }
      template <int MyArrayNum, int NArrays>
      Index row_offset(Index offset, const ExpressionSize<NArrays>& loc) const {
	return 1; 
      }
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = i<LDiags ? 0 : i-LDiags;
	j_end_plus_1 = i+UDiags+1>dim ? dim : i+UDiags+1;
	index_start = i*offset + j_start;
	index_stride = 1;
      }
      typedef BandEngine<COL_MAJOR,UDiags,LDiags> transpose_engine;
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type>::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		 Index gradient_index, const Type* data) const {
	Index off = j-i;
	Type val;
	if (off > UDiags || off < (-LDiags)) {
	  val = 0;
	}
	else {
	  val = data[index(i,j,offset)]; 
	}
	return val;
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,Active<Type> >::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, const Type* data) const {
	Index off = j-i;
	if (off > UDiags || off < (-LDiags)) {
	  return Active<Type>(0.0);
	}
	else {
	  return Active<Type>(data[index(i,j,offset)]);
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type&>::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	Index off = j-i;
	if (off > UDiags || off < (-LDiags)) {
	  throw index_out_of_bounds("Attempt to get lvalue to off-diagonal in BandMatrix"
				    ADEPT_EXCEPTION_LOCATION);
	}
	else {
	  return data[index(i,j,offset)]; 
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,ActiveReference<Type> >::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	Index off = j-i;
	if (off > UDiags || off < (-LDiags)) {
	  throw index_out_of_bounds("Attempt to get lvalue to off-diagonal in BandMatrix"
				    ADEPT_EXCEPTION_LOCATION);
	}
	else {
	  Index ind = index(i,j,offset);
	  return ActiveReference<Type>(data[ind], gradient_index+ind);
	}
      }
      Index data_size(Index dim, Index offset) const {
	return (dim-1)*(offset+1) + 1;// + dim; // - UDiags;
      }

      Index upper_offset(Index dim, Index offset, Index offdiag) const {
	return offdiag;
      }
      Index lower_offset(Index dim, Index offset, Index offdiag) const {
	return -offdiag*offset;
      }
      void check_upper_diag(Index offdiag) const {
	if (offdiag > UDiags) {
	  throw index_out_of_bounds("Attempt to get lvalue diagonal to off-diagonal in BandMatrix"
				    ADEPT_EXCEPTION_LOCATION);	  
	}
      }
      void check_lower_diag(Index offdiag) const { 
	if (-offdiag > LDiags) {
	  throw index_out_of_bounds("Attempt to get lvalue diagonal to off-diagonal in BandMatrix"
				    ADEPT_EXCEPTION_LOCATION);
	}
      }
      template <int MyArrayNum, int NArrays>
      void set_extras(Index i, Index offset,
		      ExpressionSize<NArrays>& index) const {
	index[MyArrayNum+1] = i*(offset+1) - LDiags;
	index[MyArrayNum+2] = index[MyArrayNum+1] + diagonals;
      }
      template <int MyArrayNum, int NArrays, typename Type>
      Type value_at_location(const Type* data, 
			     const ExpressionSize<NArrays>& loc) const {
	if (loc[MyArrayNum] >= loc[MyArrayNum+1]
	    && loc[MyArrayNum] < loc[MyArrayNum+2]) {
	  return data[loc[MyArrayNum]];
	}
	else {
	  return 0;
	}
      }
      template <int MyArrayNum, int NArrays, typename Type>
      void push_rhs(Stack& stack, Type multiplier, Index gradient_index,
		    const ExpressionSize<NArrays>& loc) const {
	if (loc[MyArrayNum] >= loc[MyArrayNum+1]
	    && loc[MyArrayNum] < loc[MyArrayNum+2]) {
	  stack.push_rhs(multiplier, gradient_index + loc[MyArrayNum]);
	}
      }
    };

    // The column-major version inherits from the row-major version in
    // order that some functionality can be imported
    template <Index LDiags, Index UDiags>
    struct BandEngine<COL_MAJOR, LDiags, UDiags>
      : public BandEngine<ROW_MAJOR, LDiags, UDiags> {
      static const int my_n_arrays = 3;
      static const Index diagonals = 1+LDiags+UDiags;
      const char* name() const { return BandEngineHelper<LDiags,UDiags>().name(); }
      std::string long_name() const { 
	std::stringstream s;
	s << "BandMatrix<COL_MAJOR,LDiags=" << LDiags
	  << ",UDiags=" << UDiags << ">";
	return s.str();
      }
      using BandEngine<ROW_MAJOR,LDiags,UDiags>::pack_offset;
      Index index(Index i, Index j, Index offset) const {
	//	return UDiags + i + j*offset;
	return i + j*offset;
      }
      template <int MyArrayNum, int NArrays>
      Index row_offset(Index offset, const ExpressionSize<NArrays>& loc) const {
	return offset;
      }
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = i<LDiags ? 0 : i-LDiags;
	j_end_plus_1 = i+UDiags+1>dim ? dim : i+UDiags+1;
	index_start = i + j_start*offset;
	index_stride = offset;
      }
      typedef BandEngine<ROW_MAJOR,UDiags,LDiags> transpose_engine;
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type>::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		 Index gradient_index, const Type* data) const {
	Index off = j-i;
	Type val;
	if (off > UDiags || off < (-LDiags)) {
	  val = 0;
	}
	else {
	  val = data[index(i,j,offset)]; 
	}
	return val;
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,Active<Type> >::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, const Type* data) const {
	Index off = j-i;
	if (off > UDiags || off < (-LDiags)) {
	  return Active<Type>(0.0);
	}
	else {
	  return Active<Type>(data[index(i,j,offset)]);
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,ActiveReference<Type> >::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	Index off = j-i;
	if (off > UDiags || off < (-LDiags)) {
	  throw index_out_of_bounds("Attempt to get lvalue to off-diagonal in BandMatrix"
				    ADEPT_EXCEPTION_LOCATION);
	}
	else {
	  return data[index(i,j,offset)]; 
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,ActiveReference<Type> >::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	Index off = j-i;
	if (off > UDiags || off < (-LDiags)) {
	  throw index_out_of_bounds("Attempt to get lvalue to off-diagonal in BandMatrix"
				    ADEPT_EXCEPTION_LOCATION);
	}
	else {
	  Index ind = index(i,j,offset);
	  return ActiveReference<Type>(data[ind], gradient_index+ind);
	}
      }
      using BandEngine<ROW_MAJOR,LDiags,UDiags>::data_size;

      Index upper_offset(Index dim, Index offset, Index offdiag) const {
	//	return LDiags + offdiag*offset;
	return offdiag*offset;
      }
      Index lower_offset(Index dim, Index offset, Index offdiag) const {
	//	return LDiags - offdiag;
	return -offdiag;
      }
      template <int MyArrayNum, int NArrays>
      void set_extras(Index i, Index offset,
		      ExpressionSize<NArrays>& index) const {
	index[MyArrayNum+1] = (i-LDiags)*(offset+1) + LDiags;
	index[MyArrayNum+2] = index[MyArrayNum+1] + (diagonals-1)*offset+1;
      }
      using BandEngine<ROW_MAJOR,LDiags,UDiags>::check_upper_diag;
      using BandEngine<ROW_MAJOR,LDiags,UDiags>::check_lower_diag;
      using BandEngine<ROW_MAJOR,LDiags,UDiags>::value_at_location;
      using BandEngine<ROW_MAJOR,LDiags,UDiags>::push_rhs;
    };

    // -------------------------------------------------------------------
    // Symmetric matrix storage engine
    // -------------------------------------------------------------------

    // A symmetric matrix - the default version (template parameter
    // ROW_LOWER_COL_UPPER) should be considered to use row-major
    // storage with the data held on the lower triangle of the
    // matrix. This is equivalent to column-major upper-triangle
    // storage for most uses, except that when this kind of symmetric
    // matrix is used on the left-hand-side of a statement, it will
    // only read the lower triangle of the right-hand-side of the
    // statement (assuming the upper triangle to be a symmetric copy).
    template <SymmMatrixOrientation Orient>
    struct SymmEngine : public SquareEngine<ROW_MAJOR> {
      static const int my_n_arrays = 2;
      const char* name() const { return "SymmMatrix"; }
      std::string long_name() const {
	return "SymmMatrix<ROW_LOWER_COL_UPPER>";
      }
      Index index(Index i, Index j, Index offset) const {
	return i >= j ? i*offset + j : i + j*offset;
      }
      template <int MyArrayNum, int NArrays>
      Index row_offset(Index offset, const ExpressionSize<NArrays>& loc) const {
	return loc[MyArrayNum] < loc[MyArrayNum+1] ? 1 : offset; 
      }
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = 0;
	j_end_plus_1 = i+1;
	index_start = i*offset;
	index_stride = 1;
      }
      typedef SymmEngine<ROW_LOWER_COL_UPPER> transpose_engine;
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type>::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		 Index gradient_index, const Type* data) const {
	return data[index(i,j,offset)]; 
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,Active<Type> >::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, const Type* data) const {
	return Active<Type>(data[index(i,j,offset)]);
      }
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type&>::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	return data[index(i,j,offset)]; 
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,ActiveReference<Type> >::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	Index ind = index(i,j,offset);
	return ActiveReference<Type>(data[ind], gradient_index+ind);
      }
      template <int MyArrayNum, int NArrays>
      void set_extras(Index i, Index offset,
		      ExpressionSize<NArrays>& index) const {
	index[MyArrayNum+1] = i*(offset+1);
      }
      Index upper_offset(Index dim, Index offset, Index offdiag) const {
	return offdiag*offset;
      }
      Index lower_offset(Index dim, Index offset, Index offdiag) const {
	return -offdiag*offset;
      }

      using SquareEngine<ROW_MAJOR>::pack_offset;
      using SquareEngine<ROW_MAJOR>::data_size;
      using SquareEngine<ROW_MAJOR>::check_upper_diag;
      using SquareEngine<ROW_MAJOR>::check_lower_diag;
      using SquareEngine<ROW_MAJOR>::value_at_location;
      using SquareEngine<ROW_MAJOR>::push_rhs;
    };

    // A symmetric matrix whose storage can be considered to be
    // row-major with the data stored on the upper triangle. This is
    // equivalent to column-major lower-triangular storage, except
    // that when this kind of symmetric matrix is on the LHS of a
    // statement, it will only read the upper triangle of the RHS of
    // the statement.
    template <>
    struct SymmEngine<ROW_UPPER_COL_LOWER> : public SquareEngine<ROW_MAJOR> {
      static const int my_n_arrays = 2;
      const char* name() const { return "SymmMatrix"; }
      std::string long_name() const { 
	return "SymmMatrix<ROW_UPPER_COL_LOWER>";
      }
      Index pack_offset(Index dim) const { return dim; }
      Index index(Index i, Index j, Index offset) const {
	return i <= j ? i*offset + j : i + j*offset;
      }
      template <int MyArrayNum, int NArrays>
      Index row_offset(Index offset, const ExpressionSize<NArrays>& loc) const {
	return loc[MyArrayNum] < loc[MyArrayNum+1] ? offset : 1; 
      }
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = i;
	j_end_plus_1 = dim;
	index_start = i*(1+offset);
	index_stride = 1;
      }
      typedef SymmEngine<ROW_UPPER_COL_LOWER> transpose_engine;
      Index upper_offset(Index dim, Index offset, Index offdiag) const {
	return offdiag;
      }
      Index lower_offset(Index dim, Index offset, Index offdiag) const {
	return -offdiag;
      }
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type>::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		 Index gradient_index, const Type* data) const {
	return data[index(i,j,offset)]; 
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,Active<Type> >::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, const Type* data) const {
	return Active<Type>(data[index(i,j,offset)]);
      }
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type&>::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	return data[index(i,j,offset)]; 
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,ActiveReference<Type> >::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	Index ind = index(i,j,offset);
	return ActiveReference<Type>(data[ind], gradient_index+ind);
      }
      template <int MyArrayNum, int NArrays>
      void set_extras(Index i, Index offset,
		      ExpressionSize<NArrays>& index) const {
	index[MyArrayNum+1] = i*(offset+1);
      }

      using SquareEngine<ROW_MAJOR>::data_size;
      using SquareEngine<ROW_MAJOR>::check_upper_diag;
      using SquareEngine<ROW_MAJOR>::check_lower_diag;
      using SquareEngine<ROW_MAJOR>::value_at_location;
      using SquareEngine<ROW_MAJOR>::push_rhs;
    };

    /*
    // -------------------------------------------------------------------
    // Symmetric band matrix storage engine
    // -------------------------------------------------------------------
    */

    // -------------------------------------------------------------------
    // Triangular matrix storage engines
    // -------------------------------------------------------------------

    // Forward declaration
    template <MatrixStorageOrder Order> struct UpperEngine;

    // Base class for common functions for row-major and column-major
    // storage
    template <MatrixStorageOrder Order>
    struct LowerBase : public SquareEngine<Order> {
      static const int my_n_arrays = 2;

      using SquareEngine<Order>::pack_offset;
      using SquareEngine<Order>::data_size;
      using SquareEngine<Order>::index;
      using SquareEngine<Order>::row_offset;
      using SquareEngine<Order>::check_lower_diag;
      using SquareEngine<Order>::upper_offset;
      using SquareEngine<Order>::lower_offset;

      const char* name() const { return "LowerMatrix"; }
      template <int MyArrayNum, int NArrays>
      void set_extras(Index i, Index offset,
		      ExpressionSize<NArrays>& index) const {
	index[MyArrayNum+1] = i*(offset+1);
      }
      void check_upper_diag(Index offdiag) const {
	if (offdiag > 0) {
	  throw index_out_of_bounds("Attempt to get lvalue to an upper diagonal of a lower-triangular matrix"
				    ADEPT_EXCEPTION_LOCATION);	  
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type>::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		 Index gradient_index, const Type* data) const {
	if (i >= j) {
	  return data[index(i,j,offset)]; 
	}
	else {
	  return 0;
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,Active<Type> >::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, const Type* data) const {
	if (i >= j) {
	  return Active<Type>(data[index(i,j,offset)]);
	}
	else {
	  return Active<Type>(0.0);
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type&>::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	if (i >= j) {
	  return data[index(i,j,offset)]; 
	}
	else {
	  throw index_out_of_bounds("Attempt to get lvalue to upper part of lower-triangular matrix"
				    ADEPT_EXCEPTION_LOCATION);
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,ActiveReference<Type> >::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	if (i >= j) {
	  Index ind = index(i,j,offset);
	  return ActiveReference<Type>(data[ind], gradient_index+ind);
	}
	else {
	  throw index_out_of_bounds("Attempt to get lvalue to upper part of lower-triangular matrix"
				    ADEPT_EXCEPTION_LOCATION);
	  
	}
      }
      template <int MyArrayNum, int NArrays, typename Type>
      Type value_at_location(const Type* data, 
			     const ExpressionSize<NArrays>& loc) const {
	if (loc[MyArrayNum] <= loc[MyArrayNum+1]) {
	  return data[loc[MyArrayNum]];
	}
	else {
	  return 0;
	}
      }
      template <int MyArrayNum, int NArrays, typename Type>
      void push_rhs(Stack& stack, Type multiplier, Index gradient_index,
		    const ExpressionSize<NArrays>& loc) const {
	if (loc[MyArrayNum] <= loc[MyArrayNum+1]) {
	  stack.push_rhs(multiplier, gradient_index + loc[MyArrayNum]);
	}
      }
    };

    // Lower-triangular matrix using row-major storage
    template <MatrixStorageOrder Order>
    struct LowerEngine : public LowerBase<ROW_MAJOR> {
      std::string long_name() const {
	return "LowerMatrix<ROW_MAJOR>";
      }
      typedef UpperEngine<COL_MAJOR> transpose_engine;
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = 0;
	j_end_plus_1 = i+1;
	index_start = i*offset;
	index_stride = 1;
      }
    };

    // Lower-triangular matrix using column-major storage
    template <>
    struct LowerEngine<COL_MAJOR> : public LowerBase<COL_MAJOR> {
      std::string long_name() const {
	return "LowerMatrix<COL_MAJOR>";
      }
      typedef UpperEngine<ROW_MAJOR> transpose_engine;
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = 0;
	j_end_plus_1 = i+1;
	index_start = i;
	index_stride = offset;
      }
    };

    // Base class for common functions for row-major and column-major
    // storage
    template <MatrixStorageOrder Order>
    struct UpperBase : public SquareEngine<Order> {
      static const int my_n_arrays = 2;

      using SquareEngine<Order>::pack_offset;
      using SquareEngine<Order>::data_size;
      using SquareEngine<Order>::index;
      using SquareEngine<Order>::row_offset;
      using SquareEngine<Order>::check_lower_diag;
      using SquareEngine<Order>::upper_offset;
      using SquareEngine<Order>::lower_offset;

      const char* name() const { return "UpperMatrix"; }
      template <int MyArrayNum, int NArrays>
      void set_extras(Index i, Index offset,
		      ExpressionSize<NArrays>& index) const {
	index[MyArrayNum+1] = i*(offset+1);
      }
      void check_lower_diag(Index offdiag) const {
	if (offdiag < 0) {
	  throw index_out_of_bounds("Attempt to get lvalue to a lower diagonal of an upper-triangular matrix"
				    ADEPT_EXCEPTION_LOCATION);	  
	}
      }

      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type>::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		 Index gradient_index, const Type* data) const {
	if (i <= j) {
	  return data[index(i,j,offset)]; 
	}
	else {
	  return 0;
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,Active<Type> >::type
      get_scalar(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, const Type* data) const {
	if (i <= j) {
	  return Active<Type>(data[index(i,j,offset)]);
	}
	else {
	  return Active<Type>(0.0);
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<!IsActive,Type&>::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	if (i <= j) {
	  return data[index(i,j,offset)]; 
	}
	else {
	  throw index_out_of_bounds("Attempt to get lvalue to lower part of upper-triangular matrix"
				    ADEPT_EXCEPTION_LOCATION);
	}
      }
      template <bool IsActive, typename Type>
      typename enable_if<IsActive,ActiveReference<Type> >::type
      get_reference(Index i, Index j, Index dim, Index offset, 
		    Index gradient_index, Type* data) {
	if (i <= j) {
	  Index ind = index(i,j,offset);
	  return ActiveReference<Type>(data[ind], gradient_index+ind);
	}
	else {
	  throw index_out_of_bounds("Attempt to get lvalue to lower part of upper-triangular matrix"
				    ADEPT_EXCEPTION_LOCATION);
	  
	}
      }
      template <int MyArrayNum, int NArrays, typename Type>
      Type value_at_location(const Type* data, 
			     const ExpressionSize<NArrays>& loc) const {
	if (loc[MyArrayNum] >= loc[MyArrayNum+1]) {
	  return data[loc[MyArrayNum]];
	}
	else {
	  return 0;
	}
      }
      template <int MyArrayNum, int NArrays, typename Type>
      void push_rhs(Stack& stack, Type multiplier, Index gradient_index,
		    const ExpressionSize<NArrays>& loc) const {
	if (loc[MyArrayNum] >= loc[MyArrayNum+1]) {
	  stack.push_rhs(multiplier, gradient_index + loc[MyArrayNum]);
	}
      }
    };

    // Upper-triangular matrix using row-major storage
    template <MatrixStorageOrder Order>
    struct UpperEngine : public UpperBase<ROW_MAJOR> {
      typedef LowerEngine<COL_MAJOR> transpose_engine;

      std::string long_name() const {
	return "UpperMatrix<ROW_MAJOR>";
      }
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = i;
	j_end_plus_1 = dim;
	index_start = i*(offset+1);
	index_stride = 1;
      }
    };

    // Upper-triangular matrix using column-major storage
    template <>
    struct UpperEngine<COL_MAJOR> : public UpperBase<COL_MAJOR> {
      typedef LowerEngine<ROW_MAJOR> transpose_engine;
      std::string long_name() const {
	return "UpperMatrix<COL_MAJOR>";
      }
      void get_row_range(Index i, Index dim, Index offset,
			 Index& j_start, Index& j_end_plus_1,
			 Index& index_start, Index& index_stride) const {
	j_start = i;
	j_end_plus_1 = dim;
	index_start = i*(offset+1);
	index_stride = offset;
      }
    };

  } // End namespace internal

  // -------------------------------------------------------------------
  // Definition of SpecialMatrix class
  // -------------------------------------------------------------------
  template <typename Type = Real, class Engine = internal::SquareEngine<ROW_MAJOR>,
    bool IsActive = false>
  class SpecialMatrix 
    : public Expression<Type,SpecialMatrix<Type,Engine,IsActive> >,
      protected Engine,
      protected internal::GradientIndex<IsActive> {
  public:
    // -------------------------------------------------------------------
    // SpecialMatrix: 1. Static Definitions
    // -------------------------------------------------------------------

    // Static definitions to enable the properties of this type of
    // expression to be discerned at compile time
    static const bool is_active  = IsActive;
    static const bool is_lvalue  = true;
    static const int  rank       = 2;
    static const int  n_active   = IsActive * (1 + is_complex<Type>::value);
    static const int  n_scratch  = 0;
    static const int  n_arrays   = Engine::my_n_arrays;
    static const bool is_vectorizable = false;

    // -------------------------------------------------------------------
    // SpecialMatrix: 2. Constructors
    // -------------------------------------------------------------------
    
    // Initialize an empty array
    SpecialMatrix() : data_(0), storage_(0), dimension_(0)
    { ADEPT_STATIC_ASSERT(!(std::numeric_limits<Type>::is_integer
			    && IsActive), CANNOT_CREATE_ACTIVE_ARRAY_OF_INTEGERS); }

    // Initialize an array with specified size
    SpecialMatrix(const ExpressionSize<2>& dims) : storage_(0)
    { resize(dims[0], dims[1]); }
    SpecialMatrix(Index m0) : storage_(0) { resize(m0); }
    SpecialMatrix(Index m0, Index m1) : storage_(0) { resize(m0,m1); }

    // A way to directly create arrays, needed when subsetting
    // other arrays
    SpecialMatrix(Type* data, Storage<Type>* s, Index dim, Index offset)
      : data_(data), storage_(s), dimension_(dim), offset_(offset) {
      if (storage_) {
	storage_->add_link(); 
	GradientIndex<IsActive>::set(data_, storage_);
      }
      else {
	// It is an error if an active object gets here since it will
	// not have a valid gradient index
	GradientIndex<IsActive>::assert_inactive();
      }
    }
    // Similar to the above, but with the gradient index supplied explicitly,
    // needed when an active FixedArray is being sliced
    SpecialMatrix(const Type* data0, Index data_offset, Index dim, Index offset,
		  Index gradient_index0)
      : GradientIndex<IsActive>(gradient_index0, data_offset),
	data_(const_cast<Type*>(data0)+data_offset), storage_(0), dimension_(dim), offset_(offset) { }


    // Initialize an array pointing at existing data: the fact that
    // storage_ is a null pointer is used to convey the information
    // that it is not necessary to deallocate the data when this array
    // is destructed
    SpecialMatrix(Type* data, Index dim)
      : data_(data), storage_(0), dimension_(dim), 
	offset_(Engine::pack_offset(dim)) {
      ADEPT_STATIC_ASSERT(!IsActive, CANNOT_CONSTRUCT_ACTIVE_SQUARE_ARRAY_WITHOUT_GRADIENT_INDEX);
    }

    // Copy constructor: links to the source data rather than copying
    // it.  This is needed because we want a function returning an
    // SpecialMatrix not to make a deep copy, but rather to perform a
    // (computationally cheaper) shallow copy; when the SpecialMatrix within
    // the function is destructed, it will remove its link to the
    // data, and the responsibility for deallocating the data will
    // then pass to the SpecialMatrix in the calling function.
    SpecialMatrix(SpecialMatrix& rhs) 
      : GradientIndex<IsActive>(rhs.gradient_index()),
        data_(rhs.data()), storage_(rhs.storage()), 
	dimension_(rhs.dimension()), offset_(rhs.offset()) 
    { if (storage_) storage_->add_link(); }

    // Copy constructor with const argument does exactly the same
    // thing
    SpecialMatrix(const SpecialMatrix& rhs) 
      : GradientIndex<IsActive>(rhs.gradient_index()),
        dimension_(rhs.dimension()), offset_(rhs.offset())
    { link_(const_cast<SpecialMatrix&>(rhs)); }
  private:
    void link_(SpecialMatrix& rhs) {
      data_ = const_cast<Type*>(rhs.data()); 
      storage_ = const_cast<Storage<Type>*>(rhs.storage());
      if (storage_) storage_->add_link();
    }

  public:
    // Initialize with an expression on the right hand side by
    // evaluating the expression, requiring the ranks to be equal.
    // Note that this constructor enables expressions to be used as
    // arguments to functions that expect an array - to prevent this
    // implicit conversion, use the "explicit" keyword.
    template<typename EType, class E>
    explicit
    SpecialMatrix(const Expression<EType, E>& rhs,
	  typename enable_if<E::rank == 2,int>::type = 0)
      : data_(0), storage_(0), dimension_(0)
    { *this = rhs; }

    // Destructor: if the data are stored in a Storage object then we
    // tell it that one fewer object is linking to it; if the number
    // of links to it drops to zero, it will destruct itself and
    // deallocate the memory.
    ~SpecialMatrix()
    { if (storage_) storage_->remove_link(); }

    // -------------------------------------------------------------------
    // SpecialMatrix: 3. Assignment operators
    // -------------------------------------------------------------------

    // Assignment to another matrix: copy the data...
    // Ideally we would like this to fall back to the operator=(const
    // Expression&) function, but if we don't define a copy assignment
    // operator then C++ will generate a default one :-(
    SpecialMatrix& operator=(const SpecialMatrix& rhs) {
      *this = static_cast<const Expression<Type,SpecialMatrix>&> (rhs);
      return *this;
    }

    // Assignment to an array expression of the same rank
    template <typename EType, class E>
    typename enable_if<E::rank == 2, SpecialMatrix&>::type
    operator=(const Expression<EType,E>& rhs) {
#ifndef ADEPT_NO_DIMENSION_CHECKING
      ExpressionSize<2> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (empty()) {
	resize(dims[0], dims[1]);
      }
      else if (!compatible(dims, dimensions())) {
	std::string str = "Expr";
	str += dims.str() + " object assigned to " + expression_string_();
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
#else
      if (empty()) {
	ExpressionSize<2> dims;
	if (!rhs.get_dimensions(dims)) {
	  std::string str = "Array size mismatch in "
	    + rhs.expression_string() + ".";
	  throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
	}
	resize(dims[0], dims[1]);
      }
#endif
      if (!empty()) {
#ifndef ADEPT_NO_ALIAS_CHECKING
	// Check for aliasing first
	Type const * ptr_begin;
	Type const * ptr_end;
	data_range(ptr_begin, ptr_end);
	if (rhs.is_aliased(ptr_begin, ptr_end)) {
	  SpecialMatrix copy;
	  // It would be nice to wrap noalias around rhs, but then
	  // this leads to infinite template recursion since the "="
	  // operator calls the current function but with a modified
	  // expression type. perhaps a better way would be to make
	  // copy.assign_no_alias(rhs) work.
	  copy = rhs;
	  assign_expression_<IsActive, E::is_active>(copy);
	}
	else {
#endif
	  // Select active/passive version by delegating to a
	  // protected function
	  assign_expression_<IsActive, E::is_active>(rhs);
#ifndef ADEPT_NO_ALIAS_CHECKING
	}
#endif
      }
      return *this;
    }
    
    // Assignment to an array expression of the same rank in which the
    // activeness of the right-hand-side is ignored
    template <typename EType, class E>
    typename enable_if<E::rank == 2, SpecialMatrix&>::type
    assign_inactive(const Expression<EType,E>& rhs) {
      ExpressionSize<2> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (empty()) {
	resize(dims[0], dims[1]);
      }
      else if (!compatible(dims, dimensions())) {
	std::string str = "Expr";
	str += dims.str() + " object assigned to " + expression_string_();
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }

      if (!empty()) {
	// Check for aliasing first
	Type const * ptr_begin;
	Type const * ptr_end;
	data_range(ptr_begin, ptr_end);
	if (rhs.is_aliased(ptr_begin, ptr_end)) {
	  std::cout << "ALIASED!\n";
	  SpecialMatrix copy;
	  copy.assign_inactive(rhs);
	  //	  *this = copy;
	  assign_expression_<IsActive, false>(copy);
	}
	else {
	  assign_expression_<IsActive, false>(rhs);
	}
      }
      return *this;
    }

    // Assignment to a single value copies to every element
    template <typename RType>
    typename enable_if<is_not_expression<RType>::value, SpecialMatrix&>::type
    operator=(RType rhs) {
      if (!empty()) {
	assign_inactive_scalar<IsActive>(rhs);
      }
      return *this;
    }

    // Assign active scalar expression to an active array by first
    // converting the RHS to an active scalar
    template <typename EType, class E>
    typename enable_if<E::rank == 0 && IsActive && !E::is_lvalue,
      SpecialMatrix&>::type
      operator=(const Expression<EType,E>& rhs) {
      Active<EType> x = rhs;
      *this = x;
      return *this;
    }

  
    // An active array being assigned to an active scalar
    template <typename PType>
    typename enable_if<!internal::is_active<PType>::value && IsActive, SpecialMatrix&>::type
    operator=(const Active<PType>& rhs) {
      // If not recording we call the inactive version instead
#ifdef ADEPT_RECORDING_PAUSABLE
      if (! ADEPT_ACTIVE_STACK->is_recording()) {
	assign_inactive_scalar<false>(rhs.scalar_value());
	return *this;
      }
#endif
      Type val = rhs.scalar_value();
      Index j_start, j_end_plus_1, index, index_stride;
      for (Index i = 0 ; i < dimension_; ++i) {
	Engine::get_row_range(i, dimension_, offset_, 
			      j_start, j_end_plus_1, index, index_stride);
	for (Index j = j_start; j < j_end_plus_1; ++j, index += index_stride) {
	  data_[index] = val;
	  ADEPT_ACTIVE_STACK->push_rhs(1.0, rhs.gradient_index());
	  ADEPT_ACTIVE_STACK->push_lhs(gradient_index()+index);	  
	}
      }
      return *this;
    }




    // All the compound assignment operators are unpacked, i.e. a+=b
    // becomes a=a+b; first for an Expression on the rhs.  We use
    // "noalias" sine there is no need for the entirety of the
    // right-hand-side of the expression to be copied before
    // evaluation.
    template<typename EType, class E>
    SpecialMatrix& operator+=(const Expression<EType,E>& rhs) {
      return *this = (noalias(*this) + rhs);
    }
    template<typename EType, class E>
    SpecialMatrix& operator-=(const Expression<EType,E>& rhs) {
      return *this = (noalias(*this) - rhs);
    }
    template<typename EType, class E>
    SpecialMatrix& operator*=(const Expression<EType,E>& rhs) {
      return *this = (noalias(*this) * rhs);
    }
    template<typename EType, class E>
    SpecialMatrix& operator/=(const Expression<EType,E>& rhs) {
      return *this = (noalias(*this) / rhs);
    }

    // And likewise for a passive scalar on the rhs
    template <typename PType>
    typename enable_if<is_not_expression<PType>::value, SpecialMatrix&>::type
    operator+=(const PType& rhs) {
      return *this = (noalias(*this) + rhs);
    }
    template <typename PType>
    typename enable_if<is_not_expression<PType>::value, SpecialMatrix&>::type
    operator-=(const PType& rhs) {
      return *this = (noalias(*this) - rhs);
    }
    template <typename PType>
    typename enable_if<is_not_expression<PType>::value, SpecialMatrix&>::type
    operator*=(const PType& rhs) {
      return *this = (noalias(*this) * rhs);
    }
    template <typename PType>
    typename enable_if<is_not_expression<PType>::value, SpecialMatrix&>::type
    operator/=(const PType& rhs) {
      return *this = (noalias(*this) / rhs);
    }

  
    // -------------------------------------------------------------------
    // SpecialMatrix: 4. Access functions, particularly operator()
    // -------------------------------------------------------------------
  
    // Get l-value of the element at the specified coordinates
    typename active_reference<Type,IsActive>::type
    get_lvalue(const ExpressionSize<2>& i) {
      return get_lvalue_<IsActive>(Engine::index(i[0],i[1],offset_));
    }
    
  protected:
    template <bool MyIsActive>
    typename enable_if<MyIsActive, ActiveReference<Type> >::type
    get_lvalue_(const Index& loc) {
      return ActiveReference<Type>(data_[loc], gradient_index()+loc);
    }
    template <bool MyIsActive>
    typename enable_if<!MyIsActive, Type&>::type
    get_lvalue_(const Index& loc) {
      return data_[loc];
    }

  public:
    // Access individual elements of the array.  Each argument must be
    // of integer type, or a rank-0 expression of integer type (such
    // as "end" or "end-3"). Inactive arrays return a reference to the
    // element, while active arrays return an ActiveReference<Type>
    // object.
    template <typename I0, typename I1>
    typename enable_if<all_scalar_ints<2,I0,I1>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1) {
      return Engine::template 
	get_reference<IsActive>(get_index_with_len(i0,dimension_),
				get_index_with_len(i1,dimension_),
				dimension_, offset_, 
				gradient_index(), data_);
    }
    template <typename I0, typename I1>
    typename enable_if<all_scalar_ints<2,I0,I1>::value,
		       typename active_scalar<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1) const {
      return Engine::template get_scalar<IsActive>(get_index_with_len(i0,dimension_),
						   get_index_with_len(i1,dimension_),
						   dimension_, offset_, 
						   gradient_index(), data_);
    }
    
    /*
    // If one or more of the indices is not guaranteed to be monotonic
    // at compile time then we must return an IndexedSpecialMatrix, now done
    // for all possible numbers of arguments
  
    template <typename I0, typename I1>
    typename enable_if<is_indexed<Rank,I0,I1>::value
                       && !is_ranged<Rank,I0,I1>::value,
		       IndexedSpecialMatrix<is_indexed<Rank,I0,I1>::count,
				    Type,IsActive,SpecialMatrix,I0,I1> >::type
    operator()(const I0& i0, const I1& i1) {
      static const int new_rank = is_indexed<Rank,I0,I1>::count;
      return IndexedSpecialMatrix<new_rank,Type,IsActive,SpecialMatrix,I0,I1>(*this, i0, i1);
    }
    */

    // diag_vector(offdiag), where A is a 2D square band matrix (including
    // DiagMatrix, TridiagMatrix etc), returns a 1D array pointing to
    // the "offdiag"-th diagonal of the original data, Can be used as an
    // lvalue.
    Array<1,Type,IsActive>
    diag_vector(Index offdiag = 0) {
      if (offdiag >= 0) {
	Engine::check_upper_diag(offdiag);
	ExpressionSize<1> dim(dimension_ - offdiag);
	ExpressionSize<1> offset(offset_+1);
	return Array<1,Type,IsActive>(data_
	      +Engine::upper_offset(dimension_,offset_,offdiag),
				    storage_, dim, offset);
      }
      else {
	Engine::check_lower_diag(offdiag);
	ExpressionSize<1> dim(dimension_ + offdiag);
	ExpressionSize<1> offset(offset_+1);
	return Array<1,Type,IsActive>(data_
	      +Engine::lower_offset(dimension_,offset_,offdiag),
				      storage_, dim, offset);
      }
    }

    // Extract a square sub-matrix on the diagonal
    SpecialMatrix
    submatrix_on_diagonal(Index istart, Index iend) {
      if (istart < 0 || istart > iend || iend >= dimension_) {
	throw index_out_of_bounds("Dimensions out of range in submatrix_on_diagonal"
				  ADEPT_EXCEPTION_LOCATION);
      }
      return SpecialMatrix(data_+(offset_+1)*istart, 
			  storage_, iend-istart+1, offset_);
    }

    // FIX - add an rvalue version returning const Array (?)

    // Transpose as an lvalue
    SpecialMatrix<Type, typename Engine::transpose_engine, IsActive>
    T() {
      return SpecialMatrix<Type, typename Engine::transpose_engine, 
	IsActive>(data_, storage_, dimension_, offset_);
    }

    // Return a SpecialMatrix that is a "soft" link to the data in the
    // present array; that is, it does not copy the Storage object and
    // increase the reference counter therein. This is useful in a
    // multi-threaded environment when multiple threads may wish to
    // subset the same array.
    SpecialMatrix soft_link() {
      return SpecialMatrix(data_,0,dimension_,offset_,gradient_index());
    }
    const SpecialMatrix soft_link() const {
      return SpecialMatrix(data_,0,dimension_,offset_,gradient_index());
    }
    

    // -------------------------------------------------------------------
    // SpecialMatrix: 5. Public member functions
    // -------------------------------------------------------------------
  
    // Link to an existing array of the same rank, type and activeness
    SpecialMatrix& link(SpecialMatrix& rhs) {
      if (!rhs.data()) {
	throw empty_array("Attempt to link to empty array"
			  ADEPT_EXCEPTION_LOCATION);
      }
      else {
	clear();
	data_ = rhs.data();
	storage_ = rhs.storage();
	dimension_ = rhs.dimension();
	offset_ = rhs.offset();
	if (storage_) {
	  storage_->add_link();
	}
      }
      return *this;
    }
   

#ifndef ADEPT_MOVE_SEMANTICS
    // A common pattern is to link to a subset of another
    // SpecialMatrix, e.g. vec1.link(vec2(range(2,4))), but the
    // problem is that the argument to link is a temporary so will not
    // bind to SpecialMatrix&. In C++98 we therefore need a function
    // taking const SpecialMatrix& and then cast away the const-ness. This has
    // the unfortunate side effect that a non-const SpecialMatrix can be
    // linked to a const SpecialMatrix.
    SpecialMatrix& link(const SpecialMatrix& rhs) { 
      return link(const_cast<SpecialMatrix&>(rhs)); 
    }
#else
    // But in C++11 we can solve this problem and only bind to
    // temporary non-const SpecialMatrix
    SpecialMatrix& link(SpecialMatrix&& rhs) {
      return link(const_cast<SpecialMatrix&>(rhs));
    }
#endif

    // Fortran-like link syntax A >>= B
    SpecialMatrix& operator>>=(SpecialMatrix& rhs)
    { return link(rhs); }
#ifndef ADEPT_MOVE_SEMANTICS
    SpecialMatrix& operator>>=(const SpecialMatrix& rhs)
    { return link(const_cast<SpecialMatrix&>(rhs)); }
#else
    SpecialMatrix& operator>>=(SpecialMatrix&& rhs)
    { return link(const_cast<SpecialMatrix&>(rhs)); }
#endif

    // STL-like size() returns total length of array
    Index size() const {
      return dimension_*dimension_;
    }

    // Return dimensions
    ExpressionSize<2> dimensions() const {
      return ExpressionSize<2>(dimension_,dimension_);
    }

    bool get_dimensions_(ExpressionSize<2>& dim) const {
      dim[0] = dim[1] = dimension_;
      return true;
    }

    // Return individual dimension
    Index dimension(int j = 0) const {
      return dimension_;
    }

    
    // Return individual offset
    Index offset() const {
      return offset_;
    }
    

  /*
    // Get dimensions for matrix operations, treating 1D arrays as
    // column vectors
    void get_matrix_dimensions(ExpressionSize<2>& dim) const {
      dim[0] = dim[1] = dimension_;
    }
  */

    /*
    // Return constant reference to offsets
    const ExpressionSize<Rank>& offset() const {
      return offset_;
    }
    const Index& last_offset() const { return offset_[Rank-1]; }
    */

    // Return true if the array is empty
    bool empty() const { return (dimension_ == 0); }

    // Return a string describing the array
    std::string info_string() const {
      std::stringstream str;
      str << Engine::long_name() << ", dim=" << dimension_ 
	  << ", offset=" << offset_ << ", data_location=" << data_;
      return str.str();
    }

    // Return a pointer to the start of the data
    Type* data() { return data_; }
    const Type* data() const { return data_; }
    const Type* const_data() const { return data_; }

    // Older style
    Type* data_pointer() { return data_; }
    const Type* data_pointer() const { return data_; }
    const Type* const_data_pointer() const { return data_; }

    // Return a pointer to the storage object
    Storage<Type>* storage() { return storage_; }

    // Reset the array to its original empty state, removing the link
    // to the data (which may deallocate the data if it was the only
    // link) and set the dimensions to zero
    void clear() {
      if (storage_) {
	storage_->remove_link();
	storage_ = 0;
      }
      data_ = 0;
      dimension_ = 0;
      offset_ = 0;
      GradientIndex<IsActive>::clear();
    }

    // Resize an array
    void resize(Index dim) {

      ADEPT_STATIC_ASSERT(!(std::numeric_limits<Type>::is_integer
	    && IsActive), CANNOT_CREATE_ACTIVE_ARRAY_OF_INTEGERS);

      if (storage_) {
	storage_->remove_link();
	storage_ = 0;
      }
      // Check requested dimensions
      if (dim < 0) {
	throw invalid_dimension("Negative array dimension requested"
				ADEPT_EXCEPTION_LOCATION);
      }
      else if (dim == 0) {
	clear();
      }
      else {
	dimension_ = dim;
	offset_ = Engine::pack_offset(dim);
	storage_ = new Storage<Type>(Engine::data_size(dimension_,offset_), IsActive);
	data_ = storage_->data();
	GradientIndex<IsActive>::set(data_, storage_);
      }
    }

    // Resize with an ExpressionSize object
    void resize(Index dim0, Index dim1) {
      if (dim0 != dim1) {
	throw invalid_dimension("Square matrix must have the same x and y dimensions"
				ADEPT_EXCEPTION_LOCATION);
      }
      resize(dim0);
    }

    bool is_aliased_(const Type* mem1, const Type* mem2) const {
      Type const * ptr_begin;
      Type const * ptr_end;
      data_range(ptr_begin, ptr_end);
      if (ptr_begin <= mem2 && ptr_end >= mem1) {
	return true;
      }
      else {
	return false;
      }
    }
  
    // Cannot traverse a full row just by incrementing an index by 1
    bool all_arrays_contiguous_() const { return false; }

    Type value_with_len_(const Index& j, const Index& len) const {
      ADEPT_STATIC_ASSERT(false, CANNOT_USE_VALUE_WITH_LEN_ON_ARRAY_OF_RANK_OTHER_THAN_1);
      return 0;
    }

    std::string expression_string_() const {
      std::stringstream a;
      a << Engine::name()
	<< "[" << dimension_ << "," << dimension_ << "]";
      return a.str();
    }

    // The same as operator=(inactive scalar) but does not put
    // anything on the stack
    template <typename RType>
    typename enable_if<is_not_expression<RType>::value, SpecialMatrix&>::type
    set_value(RType x) {
      if (!empty()) {
	assign_inactive_scalar<false>(x);
      }
      return *this;
    }
  
    // Is the array contiguous in memory?
    bool is_contiguous() const {
      return (offset_ == Engine::pack_offset(dimension_));
    }
  
    // Return the gradient index for the first element in the array,
    // or -1 if not active
    Index gradient_index() const {
      return GradientIndex<IsActive>::get();
    }

    /*
    std::ostream& print(std::ostream& os) const {
      if (empty()) {
	os << "(empty " << Engine::name() << ")";
      }
      else if (adept::internal::array_print_curly_brackets) {
	os << "\n";
	for (int i = 0; i < dimension_; ++i) {
	  if (i == 0) {
	    os << "{{";
	  }
	  else {
	    os << " {";
	  }
	  for (int j = 0; j < dimension_; ++j) {
	    os << (*this)(i,j);
	    if (j < dimension_-1) { os << ", "; }
	  }
	  os << "}";
	  if (i < dimension_-1) { 
	    os << ",\n"; 
	  }
	  else {
	    //	    os << "}\n"; 
	    os << "}"; 
	  }
	}
      }
      else {
	for (int i = 0; i < dimension_; ++i) {
	  for (int j = 0; j < dimension_; ++j) {
	    os << (*this)(i,j);
	    if (j < dimension_-1) { os << " "; }
	  }
	  os << "\n"; 
	}
      }
      return os;
    }
    */

    std::ostream& print(std::ostream& os) const {
      const Array<rank,Type,IsActive> x(*this);
      x.print(os);
      return os;
    }    

    std::ostream& print_raw(std::ostream& os) const {
      if (empty()) {
	os << "(empty " << Engine::name() << ")\n";
      }
      else {
	for (Index i = 0; i < Engine::data_size(dimension_,offset_); ++i) {
	  os << " " << data_[i];
	}
	os << "\n";
      }
      return os;
    }

    // Get pointers to the first and last data members in memory.  
    void data_range(Type const * &data_begin, Type const * &data_end) const {
      data_begin = data_;
      data_end = data_ + Engine::data_size(dimension_, offset_) - 1;
    }

    // The Stack::independent(x) and Stack::dependent(y) functions add
    // the gradient_index of objects x and y to std::vector<uIndex>
    // objects in Stack. Since x and y may be scalars or arrays, this
    // is best done by delegating to the Active or Array classes.
    template <typename IndexType>
    void push_gradient_indices(std::vector<IndexType>& vec) {
      ADEPT_STATIC_ASSERT(IsActive,
	  CANNOT_PUSH_GRADIENT_INDICES_FOR_INACTIVE_SPECIAL_MATRIX); 
      Index j_start, j_end_plus_1, index, index_stride;
      Index gradient_ind = gradient_index();
      vec.reserve(vec.size() + Engine::data_size(dimension_, offset_));
      for (Index i; i < dimension_; ++i) {
	Engine::get_row_range(i, dimension_, offset_, 
			      j_start, j_end_plus_1, index, index_stride);
	for (Index j = j_start; j < j_end_plus_1; ++j, index += index_stride) {
	  vec.push_back(gradient_ind + index);
	}
      }
    }

    // Return inactive array linked to original data
    SpecialMatrix<Type, Engine, false> inactive_link() {
      SpecialMatrix<Type, Engine, false> A;
      A.data_ = data_;
      A.storage_ = storage_;
      A.dimension_ = dimension_;
      A.offset_ = offset_;
      if (storage_) storage_->add_link();
      return A;
    }


    // -------------------------------------------------------------------
    // SpecialMatrix: 6. Member functions accessed by the Expression class
    // -------------------------------------------------------------------

    template <int MyArrayNum, int NArrays>
    void set_location_(const ExpressionSize<2>& i, 
		       ExpressionSize<NArrays>& index) const {
      index[MyArrayNum] = Engine::index(i[0],i[1],offset_);
      Engine::template set_extras<MyArrayNum>(i[0],offset_,index);
    }
    
    template <int MyArrayNum, int NArrays>
    Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
      return Engine::template value_at_location<MyArrayNum>(data_, loc);
    }

    Type& lvalue_at_location(const Index& loc) {
      return data_[loc];
    }

    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
    Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				  ScratchVector<NScratch>& scratch) const {
      return Engine::template value_at_location<MyArrayNum>(data_, loc);

    }

    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
    Type value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
      return Engine::template value_at_location<MyArrayNum>(data_, loc);
    }

    template <int MyArrayNum, int NArrays>
    void advance_location_(ExpressionSize<NArrays>& loc) const {
      loc[MyArrayNum] += Engine::template row_offset<MyArrayNum>(offset_, loc);
    }

    // If an expression leads to calc_gradient being called on an
    // active object, we push the multiplier and the gradient index on
    // to the operation stack (or 1.0 if no multiplier is specified
    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
    void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			const ScratchVector<NScratch>& scratch) const {
      Engine::template push_rhs<MyArrayNum>(stack, static_cast<Type>(1.0), 
					    gradient_index(), loc);
    }
    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, typename MyType>
    void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			const ScratchVector<NScratch>& scratch,
			const MyType& multiplier) const {
      Engine::template push_rhs<MyArrayNum>(stack, multiplier, gradient_index(), loc);
    }
  


    // -------------------------------------------------------------------
    // SpecialMatrix: 7. Protected member functions
    // -------------------------------------------------------------------
  protected:

    // When assigning a scalar to a whole array, there may be
    // advantage in specialist behaviour depending on the rank of the
    // array. This is a generic one that copies the number but treats
    // the present array as passive.
    template <bool LocalIsActive, typename X>
    typename enable_if<!LocalIsActive,void>::type
    assign_inactive_scalar(X x) {
      Index j_start, j_end_plus_1, index, index_stride;
      for (Index i = 0 ; i < dimension_; ++i) {
	Engine::get_row_range(i, dimension_, offset_, 
			      j_start, j_end_plus_1, index, index_stride);
	for (Index j = j_start; j < j_end_plus_1; ++j, index += index_stride) {
	  data_[index] = x;
	}
      }
    }

    // An active array being assigned the value of an inactive scalar
    template <bool LocalIsActive, typename X>
    typename enable_if<LocalIsActive,void>::type
    assign_inactive_scalar(X x) {
      // If not recording we call the inactive version instead
#ifdef ADEPT_RECORDING_PAUSABLE
      if (! ADEPT_ACTIVE_STACK->is_recording()) {
	assign_inactive_scalar<false, X>(x);
	return;
      }
#endif
      Index j_start, j_end_plus_1, index, index_stride;
      for (Index i = 0 ; i < dimension_; ++i) {
	Engine::get_row_range(i, dimension_, offset_, 
			      j_start, j_end_plus_1, index, index_stride);
	ADEPT_ACTIVE_STACK->push_lhs_range(gradient_index()+index, j_end_plus_1-j_start,
					   index_stride);
	for (Index j = j_start; j < j_end_plus_1; ++j, index += index_stride) {
	  data_[index] = x;
	}
      }
    }


    // When copying an expression to a whole array, there may be
    // advantage in specialist behaviour depending on the rank of the
    // array
    template<bool LocalIsActive, bool EIsActive, class E>
    typename enable_if<!LocalIsActive,void>::type
    assign_expression_(const E& rhs) {
      ADEPT_STATIC_ASSERT(!EIsActive, CANNOT_ASSIGN_ACTIVE_EXPRESSION_TO_INACTIVE_ARRAY);
      ExpressionSize<2> i(0);
      ExpressionSize<expr_cast<E>::n_arrays> ind(0);
      Index j_start, j_end_plus_1, index, index_stride;
      for ( ; i[0] < dimension_; ++i[0]) {
	Engine::get_row_range(i[0], dimension_, offset_, 
			      j_start, j_end_plus_1, index, index_stride);
	i[1] = j_start;
	rhs.set_location(i, ind);	
	for (i[1] = j_start; i[1] < j_end_plus_1;
	     ++i[1], index += index_stride) {
	  data_[index] = rhs.next_value(ind);
	}
      }
    }

    template<bool LocalIsActive, bool EIsActive, class E>
    typename enable_if<LocalIsActive,void>::type
    assign_expression_(const E& rhs) {
      // If recording has been paused then call the inactive version
#ifdef ADEPT_RECORDING_PAUSABLE
      if (!ADEPT_ACTIVE_STACK->is_recording()) {
	assign_expression_<false,false>(rhs);
	return;
      }
#endif
      ExpressionSize<2> i(0);
      ExpressionSize<expr_cast<E>::n_arrays> ind(0);
      ADEPT_ACTIVE_STACK->check_space(expr_cast<E>::n_active * size());
      Index j_start, j_end_plus_1, index, index_stride;
      for ( ; i[0] < dimension_; ++i[0]) {
	Engine::get_row_range(i[0], dimension_, offset_, 
			      j_start, j_end_plus_1, index, index_stride);
	i[1] = j_start;
	rhs.set_location(i, ind);	
	for (i[1] = j_start; i[1] < j_end_plus_1; ++i[1], index += index_stride) {
	  data_[index] = rhs.next_value_and_gradient(*ADEPT_ACTIVE_STACK, ind);
	  ADEPT_ACTIVE_STACK->push_lhs(gradient_index()+index);
	}
      }
    }


    // -------------------------------------------------------------------
    // SpecialMatrix: 8. Data
    // -------------------------------------------------------------------
  protected:
    Type* data_;                      // Pointer to values
    Storage<Type>* storage_;          // Pointer to Storage object
    Index dimension_;                 // Size of each dimension
    Index offset_;                    // Memory offset for
				      // slowest-varying dimension

  }; // End of SpecialMatrix class


  // -------------------------------------------------------------------
  // Helper functions
  // -------------------------------------------------------------------

  // Print array on a stream
  template <typename Type, class Engine, bool IsActive>
  inline
  std::ostream&
  operator<<(std::ostream& os, const SpecialMatrix<Type,Engine,IsActive>& A) {
    return A.print(os);
  }

  // Extract inactive part of array, working correctly depending on
  // whether argument is active or inactive
  template <typename Type, class Engine>
  inline
  SpecialMatrix<Type, Engine, false>&
  value(SpecialMatrix<Type, Engine, false>& expr) {
    return expr;
  }
  template <typename Type, class Engine>
  inline
  SpecialMatrix<Type, Engine, false>
  value(SpecialMatrix<Type, Engine, true>& expr) {
    return expr.inactive_link();
  }

  // Array::diag_matrix(), where Array is a 1D array, returns a
  // DiagMatrix containing the data as the diagonal pointing to the
  // original data, Can be used as an lvalue. Needs to be defined
  // after DiagMatrix.
  template <int Rank, typename Type, bool IsActive>
  inline
  SpecialMatrix<Type, internal::BandEngine<internal::ROW_MAJOR,0,0>, IsActive>
  Array<Rank,Type,IsActive>::diag_matrix() {
    return SpecialMatrix<Type, internal::BandEngine<internal::ROW_MAJOR,0,0>,
      IsActive> (data_, storage_, dimensions_[0], offset_[0]-1);
  }

  template <typename Type, bool IsActive, Index J0, Index J1, Index J2,
	    Index J3, Index J4, Index J5, Index J6>
  inline
  SpecialMatrix<Type, internal::BandEngine<internal::ROW_MAJOR,0,0>, IsActive>
  FixedArray<Type,IsActive,J0,J1,J2,J3,J4,J5,J6>::diag_matrix() {
    return SpecialMatrix<Type, internal::BandEngine<internal::ROW_MAJOR,0,0>, 
      IsActive> (data_, 0, dimension_<0>::value, offset_<0>::value-1,
		 GradientIndex<IsActive>::get());
  }

} // End namespace adept




#endif
