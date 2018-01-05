/* IndexedArray.h -- Support for indexed arrays

    Copyright (C) 2015-2018 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

  
   If an Array is indexed via A(i,j,...) then there are three possible
   return values: (1) a scalar, if all indices are scalar integers
   (including 0-rank expressions such as "end"); (2) an Array that
   links to a subset of the data in the original Array, if one or more
   of the indices is a RangeIndex object and all the rest are scalar
   integers; and (3) an IndexedArray object, if one or more of the
   indices is a vector of integers.  All of these return values can be
   used on the left-hand-side of an expression.

   This file treats the last case.  The code is quite complex because
   the rank of the IndexedArray may be reduced compared to the
   original Array, since dimensions indexed by scalar integers are
   removed in IndexedArray.

*/


#ifndef AdeptIndexedArray_H
#define AdeptIndexedArray_H 1

#include <vector>

#include <adept/Expression.h>

namespace adept {

  namespace internal {

    // ---------------------------------------------------------------------
    // Section 1. get_size_with_len
    // ---------------------------------------------------------------------
    // Return the size of an index to an individual dimension, with
    // specializations for the different types of index. The second
    // argument passes in the length of the dimension being indexed;
    // that way if any of the indices are expressions containing
    // "end", this will be replaced by that dimension length minus 1.

    // A scalar integer and rank-0 expression have a size of unity
    inline
    Index get_size_with_len(const Index& j, const Index&) { return 1; }

    template <typename T, class E>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer
		       && E::rank == 0, Index>::type
    get_size_with_len(const Expression<T,E>&, const Index& len) { return 1; }

    // Extract the length of an IntVector
    template <typename T, class E>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer
		       && E::rank == 1 && !is_range<E>::value, Index>::type
    get_size_with_len(const Expression<T,E>& e, const Index& len) { 
      ExpressionSize<1> s;
      e.get_dimensions(s);
      return s[0];
    }

    // Extract the length of a RangeIndex object, which might be
    // dependent on len if "end" is present
    template <typename T, class E>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer
		       && is_range<E>::value, Index>::type
    get_size_with_len(const Expression<T,E>& e, const Index& len) { 
      return e.cast().size_with_len_(len);
    }

    // Allow std::vector to be used to index Arrays
    template <typename T>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer, Index>::type
    get_size_with_len(const std::vector<T>& v, const Index&) { 
      return v.size();
    }


    // ---------------------------------------------------------------------
    // Section 2. get_value_with_len
    // ---------------------------------------------------------------------
    // Return the j'th value of index ind.

#ifndef ADEPT_BOUNDS_CHECKING
    // For scalar indices there is only one value to return - j ought
    // to be zero but we don't check this
    inline
    Index get_value_with_len(const Index& ind, const Index& j, const Index&)
    { return ind; }

    template <typename T, class E>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer
		       && (E::rank < 2), Index>::type
    get_value_with_len(const Expression<T,E>& ind, const Index& j, 
		       const Index& len) {
      return ind.value_with_len(j, len); 
    }

    template <typename T>
    inline
    Index get_value_with_len(const std::vector<T>& ind, const Index& j, 
			     const Index&) { 
      return ind[j];
    }
#else
    // For scalar indices there is only one value to return - j ought
    // to be zero but we don't check this
    inline
    Index get_value_with_len(const Index& ind, const Index& j, const Index& len)   { 
      if (j != 0) {
	throw index_out_of_bounds("Index to IndexedArray is out of bounds"
				  ADEPT_EXCEPTION_LOCATION);
      }
      else if (ind < 0 || ind >= len) {
	throw index_out_of_bounds("Scalar index out of bounds in IndexedArray"
				  ADEPT_EXCEPTION_LOCATION);
      }
      else {
	return ind; 
      }
    }

    template <typename T, class E>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer
		       && (E::rank < 2), Index>::type
    get_value_with_len(const Expression<T,E>& ind, const Index& j, 
		       const Index& len) {
      Index i = ind.value_with_len(j, len);
      if (i < 0 || i >= len) {
	abort();
	throw index_out_of_bounds("Index out of bounds in IndexedArray"
				  ADEPT_EXCEPTION_LOCATION);
      }
      else {
	return i;
      }
    }

    template <typename T>
    inline
    Index get_value_with_len(const std::vector<T>& ind, const Index& j, 
			     const Index& len) {
      Index i = ind[j];
      if (i < 0 || i >= len) {
	throw index_out_of_bounds("Index from std::vector out of bounds in IndexedArray"
				  ADEPT_EXCEPTION_LOCATION);
      }
      else {
	return i;
      }    
    }
#endif

    // ---------------------------------------------------------------------
    // Section 3. is_int_vector
    // ---------------------------------------------------------------------
    // is_int_vector<Type>::value is "true" if Type is a rank-1
    // integer expression (including RangeIndex objects), false
    // otherwise.

    template <typename T, class Enable = void>
    struct is_int_vector { };

    template <typename T>
    struct is_int_vector<T,
	 typename enable_if<is_not_expression<T>::value>::type>
    { static const bool value = false; };

    template <typename T>
    struct is_int_vector<T,
       typename enable_if<!is_not_expression<T>::value>::type>
    {
      static const bool value 
      = std::numeric_limits<typename T::type>::is_integer
	&& expr_cast<T>::rank == 1;
    };
    
    template <typename T>
    struct is_index {
      static const bool value = is_regular_index<T>::value 
	|| is_int_vector<T>::value;
      static const int count = value;
    };

    template <typename T>
    struct is_irregular_index {
      static const bool value = !is_range<T>::value 
	&& is_int_vector<T>::value;
      static const int count = value;
    };
    
    
    // ---------------------------------------------------------------------
    // Section 4. is_irregular_index
    // ---------------------------------------------------------------------

    // is_irregular_index<Rank,I0,I1,...>::value is "true" if indices
    // I0 to I[Rank-1] contains at least one integer vector that could
    // be irregularly spaced, and all the other are valid indices.
    // The ::count member gives the number of non-scalar indices,
    // which is the rank of the IndexedArray objects resulting from
    // indexing an Array of the specified Rank with indices I0 to
    // I[Rank-1].
    template <int Rank, typename I0, typename I1 = Index, 
	      typename I2 = Index, typename I3 = Index,
	      typename I4 = Index, typename I5 = Index,
	      typename I6 = Index>
    struct is_irreg_indexed {
      static const bool value
        = (   is_irregular_index<I0>::value || is_irregular_index<I1>::value
	   || is_irregular_index<I2>::value || is_irregular_index<I3>::value
	   || is_irregular_index<I4>::value || is_irregular_index<I5>::value
	   || is_irregular_index<I6>::value)
	&& (   is_index<I0>::value && is_index<I1>::value
	    && is_index<I2>::value && is_index<I3>::value
	    && is_index<I4>::value && is_index<I5>::value
	    && is_index<I6>::value);
      static const int count 
         = 7 - (  is_scalar_int<I0>::count + is_scalar_int<I1>::count
		+ is_scalar_int<I2>::count + is_scalar_int<I3>::count
		+ is_scalar_int<I4>::count + is_scalar_int<I5>::count
		+ is_scalar_int<I6>::count);
    };
    

    // ---------------------------------------------------------------------
    // Section 5. IndexedArray class
    // ---------------------------------------------------------------------
    // A class holding references to an Array to be indexed, plus
    // references to the objects corresponding to each of its
    // dimension being indexed.  IndexedArray objects are temporary,
    // generated by indexing an Array object "A" via A(i,j,...) within
    // an expression.  The indices themselves may be temporary results
    // of integer expressions, but by C++ rules they will not be
    // deleted until the full expression is complete.
    template <int Rank, typename Type, bool IsActive, 
	      class ArrayType, class I0, 
	      class I1 = Index, class I2 = Index, 
	      class I3 = Index, class I4 = Index, 
	      class I5 = Index, class I6 = Index>
    class IndexedArray : public Expression<Type, 
		   IndexedArray<Rank, Type, IsActive, ArrayType, 
				I0, I1, I2, I3, I4, I5, I6> > {
    public:
      // ---------------------------------------------------------------------
      // Section 5.1. IndexedArray: Static definitions
      // ---------------------------------------------------------------------
      static const int  rank       = Rank;
      static const int  n_scratch  = 1;
      static const int  n_active   = IsActive;

      // We require three indices to be stored to optimize the
      // calculation of the location: first the location of the start
      // of the row, second the index to i[Rank-1] (0, 1, 2...), and 
      // third the location passed to the Array
      static const int  n_arrays   = 3;
      static const bool is_active  = IsActive;

      // The rank of the array being indexed may be higher than the
      // result of the index due to singleton indices
      // (e.g. Matrix(IntVector,int) has rank 1 even though Matrix has
      // rank 2).
      static const int  a_rank      = ArrayType::rank;


      // ---------------------------------------------------------------------
      // Section 5.2. IndexedArray: Constructors
      // ---------------------------------------------------------------------
      // Make default constructor that the compiler might generate
      // itself unreachable
    private:
      IndexedArray() { }

    public:
      // The constructor sets all unused indices to an integer of zero
      IndexedArray(ArrayType& a, const I0& i0,
		   const I1& i1 = 0, const I2& i2 = 0,
		   const I3& i3 = 0, const I4& i4 = 0,
		   const I5& i5 = 0, const I6& i6 = 0)
	: a_(a), i0_(i0), i1_(i1), i2_(i2), i3_(i3), 
	  i4_(i4), i5_(i5), i6_(i6), a_dims_(a.dimensions())
      {
	// Compute the dimensions of the IndexedArray objects from the
	// lengths of the non-singleton indices to Array
	set_dimensions_<0,0>(); 

	// For stepping through memory efficiently in the inner loop,
	// we store the distance between elements in the fastest
	// varying dimension in Array
	last_offset_ = a.offset()[a_fastest_varying_dim];
      }

      // ---------------------------------------------------------------------
      // Section 5.3. IndexedArray: Functions facilitating Expression functionality
      // ---------------------------------------------------------------------
      bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	dim = dimensions_;
	return true;
      }
      
      std::string info_string() const {
	std::stringstream s;
	s << expression_string_() 
	  << ", array-dim=" << a_dims_ << ", dim=" << dimensions_
	  << ", last-offset_=" << last_offset_;
	return s.str();	
      }

      std::string expression_string_() const {
	std::string str;
	str = a_.expression_string() + "(";
	str += expr_string(i0_);
	if (a_rank > 1) {
	  str += std::string(",") + expr_string(i1_);
	  if (a_rank > 2) {
	    str += std::string(",") + expr_string(i2_);
	    if (a_rank > 3) {
	      str += std::string(",") + expr_string(i3_);
	      if (a_rank > 4) {
		str += std::string(",") + expr_string(i4_);
		if (a_rank > 5) {
		  str += std::string(",") + expr_string(i5_);
		  if (a_rank > 6) {
		    str += std::string(",") + expr_string(i6_);
		  }
		}
	      }
	    }
	  }
	}
	str += ")";
	return str;
      }
     
    protected:
      // Helper functions for expression_string()
      template <typename T, typename E>
      std::string expr_string(const Expression<T,E>& e) const {
	return e.expression_string();
      }
      template <typename T>
      typename enable_if<is_not_expression<T>::value, std::string>::type
      expr_string(const T& e) const {
	std::stringstream s;
	s << e;
	return s.str();
      }

    public:
      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return a_.is_aliased(mem1, mem2);
      }

      Type value_with_len_(const Index& i, const Index& len) const {
	// Treat as one dimensional
	return a_(get_value_with_len_<Rank-1>(i));
      }
      
      template <int MyArrayNum, int NArrays>
      void set_location_(const ExpressionSize<Rank>& coords,
			 ExpressionSize<NArrays>& loc) const {
	ExpressionSize<a_rank> a_coords;
	translate_coords_<0,0>(coords, a_coords);
	// Location of start of most rapidly varying dimension in
	// Array
	a_.template set_location_<MyArrayNum>(a_coords, loc);
	// Index to most rapidly varying dimension in IndexedArray
	loc[MyArrayNum+1] = coords[Rank-1];
	loc[MyArrayNum+2] = loc[MyArrayNum] + last_offset_
	  * get_value_with_len_<a_fastest_varying_dim>(loc[MyArrayNum+1]);
      }

      // Advance the location of each array in the expression
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	++loc[MyArrayNum+1];
	// Note that next_value calls advance_location even when it
	// has reached the end of a row, in which case finding the
	// location of an indexed array is an invalid operation since
	// it would require accessing the indexing array out of
	// bounds. Hence the "if" test here.
	if (loc[MyArrayNum+1] < dimensions_[Rank-1]) {
	  loc[MyArrayNum+2] = loc[MyArrayNum] + last_offset_
	    * get_value_with_len_<a_fastest_varying_dim>(loc[MyArrayNum+1]);
	}
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return a_.template value_at_location_<MyArrayNum+2>(loc);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	ADEPT_STATIC_ASSERT(ArrayType::n_scratch == 0,
			    ASSUMING_ARRAY_N_SCRATCH_IS_ZERO);
	return (scratch[MyScratchNum] 
		= a_.template value_at_location_<MyArrayNum+2>(loc));
      }
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	a_.template calc_gradient_<MyArrayNum+2,MyScratchNum+1>(stack, loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch,
		typename MyType>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
	a_.template calc_gradient_<MyArrayNum+2, MyScratchNum+1>(stack, loc, 
								 scratch, multiplier);
      }


      // ---------------------------------------------------------------------
      // Section 5.4. IndexedArray: Operators
      // ---------------------------------------------------------------------
      // Operators so that IndexedArray can appear on the
      // left-hand-side of a statement
      IndexedArray& operator=(const IndexedArray& src) {
	*this = static_cast<const Expression<Type,IndexedArray>&>(src);
	return *this;
      }

      // Assignment to a single value copies to every element
      template <typename RType>
      typename enable_if<is_not_expression<RType>::value, IndexedArray&>::type
      operator=(RType rhs) {
	if (!empty()) {
#ifdef ADEPT_RECORDING_PAUSABLE
	  if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
	    assign_inactive_scalar_<IsActive>(rhs);
#ifdef ADEPT_RECORDING_PAUSABLE
	  }
	  else {
	    assign_inactive_scalar_<false>(rhs);
	  }
#endif
	}
	return *this;
      }

    public:
      // Assignment to an array expression of the same rank
      template <typename EType, class E>
      typename enable_if<E::rank == Rank, IndexedArray&>::type
      operator=(const Expression<EType,E>& rhs) {
      // Definition moved to Array.h due to its dependence on the
      // Array class
	ExpressionSize<Rank> dims;
	if (!rhs.get_dimensions(dims)) {
	  std::string str = "Array size mismatch in "
	    + rhs.expression_string() + ".";
	  throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
	}
	else if (!compatible(dims, dimensions_)) {
	  std::string str = "Expr";
	  str += dims.str() + " object assigned to " + expression_string_();
	  throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
	}

	if (!empty()) {
#ifndef ADEPT_NO_ALIAS_CHECKING
	  // Check for aliasing first
	  Type const * ptr_begin;
	  Type const * ptr_end;
	  a_.data_range(ptr_begin, ptr_end);
	  if (rhs.is_aliased(ptr_begin, ptr_end)) {
	    Array<Rank,Type,IsActive> copy;
	    copy = noalias(rhs);
	    assign_expression_<IsActive, E::is_active>(copy);
	  }
	  else {
#endif
	    assign_expression_<IsActive, E::is_active>(rhs);
#ifndef ADEPT_NO_ALIAS_CHECKING
	  }
#endif
	}
	return *this;
      }


      // Assign active scalar expression to an active array by first
      // converting the RHS to an active scalar
      template <typename EType, class E>
      typename enable_if<E::rank == 0 && (Rank > 0)
	                 && IsActive && !E::is_lvalue,
	IndexedArray&>::type
      operator=(const Expression<EType,E>& rhs) {
	Active<EType> x = rhs;
	*this = x;
	return *this;
      }

      // Assign an active scalar to an active array
      template <typename PType>
      typename enable_if<!internal::is_active<PType>::value && IsActive, IndexedArray&>::type
      operator=(const Active<PType>& rhs) {
	ADEPT_STATIC_ASSERT(IsActive, ATTEMPT_TO_ASSIGN_ACTIVE_SCALAR_TO_INACTIVE_INDEXED_ARRAY);
	if (!empty()) {
#ifdef ADEPT_RECORDING_PAUSABLE
	  if (!ADEPT_ACTIVE_STACK->is_recording()) {
	    assign_inactive_scalar_<false>(rhs.scalar_value());
	    return *this;
	  }
#endif
	  
	  ExpressionSize<Rank> coords(0);
	  ExpressionSize<a_rank> a_coords(0);
	  ExpressionSize<1> a_loc(0);
	  Type val = rhs.scalar_value();
	  int dim;
	  static const int last = Rank-1;
	  do {
 	    coords[last] = 0;
	    // Convert between the coordinates of the IndexedArray
	    // object to the coordinates of the Array object
	    translate_coords_<0,0>(coords, a_coords);
	    a_.set_location(a_coords, a_loc);
	    // Innermost loop
	    for ( ; coords[last] < dimensions_[last]; ++coords[last]) {
	      Index index = a_loc[0]
		+ last_offset_
		* get_value_with_len_<a_fastest_varying_dim>(coords[last]);
	      a_.data()[index] = val;
	      ADEPT_ACTIVE_STACK->push_rhs(1.0, rhs.gradient_index());
	      ADEPT_ACTIVE_STACK->push_lhs(a_.gradient_index()+index);
	    }
	    advance_index(dim, coords);
	  } while (dim >= 0);
        }
        return *this;
      } 


#define ADEPT_DEFINE_OPERATOR(OPERATOR, OPSYMBOL)	\
    template <class RType>			\
    IndexedArray& OPERATOR(const RType& rhs) {	\
    return *this = noalias(*this) OPSYMBOL rhs;	\
    }
    ADEPT_DEFINE_OPERATOR(operator+=, +);
    ADEPT_DEFINE_OPERATOR(operator-=, -);
    ADEPT_DEFINE_OPERATOR(operator*=, *);
    ADEPT_DEFINE_OPERATOR(operator/=, /);
    //    ADEPT_DEFINE_OPERATOR(operator&=, &);
    //    ADEPT_DEFINE_OPERATOR(operator|=, |);
#undef ADEPT_DEFINE_OPERATOR

#ifdef ADEPT_CXX11_FEATURES

    // To enable assignment to an initializer list we take a simple
    // but inefficient strategy of creating a temporary Array and
    // assigning to that
    template <class IType>
    IndexedArray& operator=(std::initializer_list<IType> list) {
      ADEPT_STATIC_ASSERT(Rank==1,RANK_MISMATCH_IN_INITIALIZER_LIST);
      Array<Rank,Type,false> array = list;
      return (*this = array);
    }
    template <class IType>
    IndexedArray& operator=(std::initializer_list<
			    std::initializer_list<IType> > list) {
      ADEPT_STATIC_ASSERT(Rank==2,RANK_MISMATCH_IN_INITIALIZER_LIST);
      Array<Rank,Type,false> array = list;
      return (*this = array);
    }
    template <class IType>
    IndexedArray& operator=(std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<IType> > > list) {
      ADEPT_STATIC_ASSERT(Rank==3,RANK_MISMATCH_IN_INITIALIZER_LIST);
      Array<Rank,Type,false> array = list;
      return (*this = array);
    }
    template <class IType>
    IndexedArray& operator=(std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<IType> > > > list) {
      ADEPT_STATIC_ASSERT(Rank==4,RANK_MISMATCH_IN_INITIALIZER_LIST);
      Array<Rank,Type,false> array = list;
      return (*this = array);
    }
    template <class IType>
    IndexedArray& operator=(std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<IType> > > > > list) {
      ADEPT_STATIC_ASSERT(Rank==5,RANK_MISMATCH_IN_INITIALIZER_LIST);
      Array<Rank,Type,false> array = list;
      return (*this = array);
    }
    template <class IType>
    IndexedArray& operator=(std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<
			    std::initializer_list<IType> > > > > > list) {
      ADEPT_STATIC_ASSERT(Rank==6,RANK_MISMATCH_IN_INITIALIZER_LIST);
      Array<Rank,Type,false> array = list;
      return (*this = array);
    }

#endif


    protected:
      // ---------------------------------------------------------------------
      // Section 5.5. IndexedArray: Internal functions facilitating operator=
      // ---------------------------------------------------------------------

      // Two versions of assigning an inactive scalar to an indexed
      // array depending on whether the indexed array is active -
      // first the case when it is not
      template <bool LocalIsActive, typename X>
      typename enable_if<!LocalIsActive,void>::type
      assign_inactive_scalar_(X x) {
	ExpressionSize<Rank> coords(0);
	ExpressionSize<a_rank> a_coords(0);
	ExpressionSize<1> a_loc(0);
	int dim;
	static const int last = Rank-1;
	do {
	  coords[last] = 0;
	  // Convert between the coordinates of the IndexedArray
	  // object to the coordinates of the Array object
	  translate_coords_<0,0>(coords, a_coords);
	  a_.set_location(a_coords, a_loc);
	  // Innermost loop
	  for ( ; coords[last] < dimensions_[last]; ++coords[last]) {
	    a_.data()[a_loc[0]
		      + last_offset_
		      * get_value_with_len_<a_fastest_varying_dim>(coords[last])]
	      = x;
	  }
	  advance_index(dim, coords);
	} while (dim >= 0);
      }

      // Active version of assigning an inactive scalar
      template <bool LocalIsActive, typename X>
      typename enable_if<LocalIsActive,void>::type
      assign_inactive_scalar_(X x) {
	// If not recording we call the inactive version instead
#ifdef ADEPT_RECORDING_PAUSABLE
	if (!ADEPT_ACTIVE_STACK->is_recording()) {
	  assign_inactive_scalar_<false, X>(x);
	  return;
	}
#endif
	ExpressionSize<Rank> coords(0);
	ExpressionSize<a_rank> a_coords(0);
	ExpressionSize<1> a_loc(0);
	int dim;
	static const int last = Rank-1;
	do {
	  coords[last] = 0;
	  // Convert between the coordinates of the IndexedArray
	  // object to the coordinates of the Array object
	  translate_coords_<0,0>(coords, a_coords);
	  a_.set_location(a_coords, a_loc);
	  // Innermost loop
	  for ( ; coords[last] < dimensions_[last]; ++coords[last]) {
	    Index index = a_loc[0]
	      + last_offset_
	      * get_value_with_len_<a_fastest_varying_dim>(coords[last]);
	    a_.data()[index] = x;
	    ADEPT_ACTIVE_STACK->push_lhs(a_.gradient_index()+index);
	  }
	  advance_index(dim, coords);
	} while (dim >= 0);
      }
      

      // Assign expression has two versions, passive and active
      template<bool LeftIsActive, bool RightIsActive, class E>
      typename enable_if<!LeftIsActive,void>::type
      assign_expression_(const E& rhs) {
	ADEPT_STATIC_ASSERT(!RightIsActive, 
		    CANNOT_ASSIGN_ACTIVE_EXPRESSION_TO_INACTIVE_INDEXED_ARRAY);
	ExpressionSize<Rank> coords(0);
	ExpressionSize<a_rank> a_coords(0);
	ExpressionSize<expr_cast<E>::n_arrays> loc(0);
	ExpressionSize<1> a_loc(0);
	int dim;
	static const int last = Rank-1;
	do {
	  coords[last] = 0;
	  rhs.set_location(coords, loc);
	  // Convert between the coordinates of the IndexedArray
	  // object to the coordinates of the Array object
	  translate_coords_<0,0>(coords, a_coords);
	  a_.set_location(a_coords, a_loc);
	  // Innermost loop
	  for ( ; coords[last] < dimensions_[last]; ++coords[last]) {
	    a_.data()[a_loc[0]
		      + last_offset_
		      * get_value_with_len_<a_fastest_varying_dim>(coords[last])]
	      = rhs.next_value(loc);
	  }
	  advance_index(dim, coords);
	} while (dim >= 0);
      }

      // Active LHS, passive RHS
      template<bool LeftIsActive, bool RightIsActive, class E>
      typename enable_if<LeftIsActive && !RightIsActive,void>::type
      assign_expression_(const E& rhs) {
#ifdef ADEPT_RECORDING_PAUSABLE
	if (!ADEPT_ACTIVE_STACK->is_recording()) {
	  assign_expression_<false,false>(rhs);
	  return;
	}
#endif
	ExpressionSize<Rank> coords(0);
	ExpressionSize<a_rank> a_coords(0);
	ExpressionSize<expr_cast<E>::n_arrays> loc(0);
	ExpressionSize<1> a_loc(0);
	int dim;
	static const int last = Rank-1;
	do {
	  coords[last] = 0;
	  rhs.set_location(coords, loc);
	  // Convert between the coordinates of the IndexedArray
	  // object to the coordinates of the Array object
	  translate_coords_<0,0>(coords, a_coords);
	  a_.set_location(a_coords, a_loc);
	  // Innermost loop
	  for ( ; coords[last] < dimensions_[last]; ++coords[last]) {
	    Index index = a_loc[0]
		      + last_offset_
	      * get_value_with_len_<a_fastest_varying_dim>(coords[last]);
	    a_.data()[index] = rhs.next_value(loc);
	    ADEPT_ACTIVE_STACK->push_lhs(a_.gradient_index()+index);
	  }
	  advance_index(dim, coords);
	} while (dim >= 0);
      }

      // Active LHS, active RHS
      template<bool LeftIsActive, bool RightIsActive, class E>
      typename enable_if<LeftIsActive && RightIsActive,void>::type
      assign_expression_(const E& rhs) {
#ifdef ADEPT_RECORDING_PAUSABLE
	if (!ADEPT_ACTIVE_STACK->is_recording()) {
	  assign_expression_<false,false>(rhs);
	  return;
	}
#endif
	ExpressionSize<Rank> coords(0);
	ExpressionSize<a_rank> a_coords(0);
	ExpressionSize<expr_cast<E>::n_arrays> loc(0);
	ExpressionSize<1> a_loc(0);
	int dim;
	static const int last = Rank-1;

	ADEPT_ACTIVE_STACK->check_space(expr_cast<E>::n_active * dimensions_[0]);
	do {
	  coords[last] = 0;
	  rhs.set_location(coords, loc);
	  // Convert between the coordinates of the IndexedArray
	  // object to the coordinates of the Array object
	  translate_coords_<0,0>(coords, a_coords);
	  a_.set_location(a_coords, a_loc);
	  // Innermost loop
	  for ( ; coords[last] < dimensions_[last]; ++coords[last]) {
	    Index index = a_loc[0]
		      + last_offset_
	      * get_value_with_len_<a_fastest_varying_dim>(coords[last]);
	    a_.data()[index] = rhs.next_value_and_gradient(*ADEPT_ACTIVE_STACK, loc);
	    ADEPT_ACTIVE_STACK->push_lhs(a_.gradient_index()+index);
	  }
	  advance_index(dim, coords);
	} while (dim >= 0);
      }

      // Move to the start of the next row
      void advance_index(int& dim, ExpressionSize<Rank>& coords) const {
	dim = Rank-1;
	while (--dim >= 0) {
	  if (++coords[dim] >= dimensions_[dim]) {
	    coords[dim] = 0;
	  }
	  else {
	    break;
	  }
	}
      }


      bool empty() { return dimensions_[0] == 0; }
      
      // Declare I as it is used before it is defined
      template<int Dim> struct Ix;

      // Translate coordinates in terms of the IndexedArray object in
      // to coordinates to the Array object it wraps, accounting for
      // singleton dimensions in Array that are not included in the
      // dimensions that IndexedArray presents to external objects
      template <int InDim, int OutDim>
      typename enable_if<!is_scalar_int<typename Ix<OutDim>::type>::value
                         && (InDim < Rank-1), void>::type
      translate_coords_(const ExpressionSize<Rank>& in,
		       ExpressionSize<a_rank>& out) const {
	// Compute the index of the OutDim dimension of Array
	out[OutDim] = get_value_with_len(index_object_<OutDim>(),
					 in[InDim],a_dims_[OutDim]);
	// Move on to the next dimension
	translate_coords_<InDim+1,OutDim+1>(in, out);
      }

      template <int InDim, int OutDim>
      typename enable_if<(OutDim < a_rank)
	                 && is_scalar_int<typename Ix<OutDim>::type>::value,
			 void>::type
      translate_coords_(const ExpressionSize<Rank>& in,
		        ExpressionSize<a_rank>& out) const {
	// This is a singleton dimension so the 0th element is the
	// only element
	out[OutDim] = get_value_with_len(index_object_<OutDim>(),
					  0,a_dims_[OutDim]);
	// Move on to the next OutDim dimension of Array
	translate_coords_<InDim,OutDim+1>(in, out);
      }

      template <int InDim, int OutDim>
      typename enable_if<!is_scalar_int<typename Ix<OutDim>::type>::value
                         && InDim == Rank-1, void>::type
      translate_coords_(const ExpressionSize<Rank>& in,
		       ExpressionSize<a_rank>& out) const {
	// The final non-singleton dimension is set to zero, since it
	// will be incremented later by advance_location
	out[OutDim] = 0;
	// Do any further dimensions, which must be singletons
	translate_coords_<InDim+1,OutDim+1>(in, out);
      }

      // Run out of dimensions: do nothing
      template <int InDim, int OutDim>
      typename enable_if<InDim == Rank && OutDim == a_rank, void>::type
      translate_coords_(const ExpressionSize<Rank>& in,
		       ExpressionSize<a_rank>& out) const { }

      template <int Dim>
      Index get_value_with_len_(const Index& j) const {
	return get_value_with_len(index_object_<Dim>(), j, a_dims_[Dim]);
 	//return get_value_with_len(index_object_<Dim>(), j, dimensions_[Dim]);
     }


      // ---------------------------------------------------------------------
      // Section 5.6. IndexedArray: Helper functions for the constructor
      // ---------------------------------------------------------------------
      // Helper function for translating between the dimensions of the
      // Array object and that of the IndexedArray, the latter of
      // which has removed the singleton dimensions of the former
      template <int InDim, int OutDim>
      typename enable_if<(OutDim < a_rank)
	&& !is_scalar_int<typename Ix<OutDim>::type>::value,void>::type
      set_dimensions_() {
	dimensions_[InDim] = get_size_with_len(index_object_<OutDim>(),
					      a_dims_[OutDim]);
	set_dimensions_<InDim+1, OutDim+1>();
      }
      template <int InDim, int OutDim>
      typename enable_if<(OutDim < a_rank)
	&& is_scalar_int<typename Ix<OutDim>::type>::value,void>::type
      set_dimensions_() {
	set_dimensions_<InDim, OutDim+1>();
      }
      template <int InDim, int OutDim>
      typename enable_if<OutDim == a_rank,void>::type
      set_dimensions_() { }



      // ---------------------------------------------------------------------
      // Section 5.7. IndexedArray: Low-level helper sub-classes and functions
      // ---------------------------------------------------------------------

      // The individual indices are stored in objects of type I0 to
      // I[Rank-1].  The following sub-class "index_alias" enables the
      // definition of the sub-class I that is used such that
      // Ix<Dim>::type returns the type of index "Dim" at compile time.
      template <int Dim,class X0,class X1,class X2,class X3,class X4,
		class X5,class X6> struct index_alias { };

      template<class X0,class X1,class X2,class X3,class X4,class X5,class X6> 
      struct index_alias<0,X0,X1,X2,X3,X4,X5,X6> { typedef X0 type; };

      template<class X0,class X1,class X2,class X3,class X4,class X5,class X6> 
      struct index_alias<1,X0,X1,X2,X3,X4,X5,X6> { typedef X1 type; };

      template<class X0,class X1,class X2,class X3,class X4,class X5,class X6> 
      struct index_alias<2,X0,X1,X2,X3,X4,X5,X6> { typedef X2 type; };

      template<class X0,class X1,class X2,class X3,class X4,class X5,class X6> 
      struct index_alias<3,X0,X1,X2,X3,X4,X5,X6> { typedef X3 type; };

      template<class X0,class X1,class X2,class X3,class X4,class X5,class X6> 
      struct index_alias<4,X0,X1,X2,X3,X4,X5,X6> { typedef X4 type; };

      template<class X0,class X1,class X2,class X3,class X4,class X5,class X6> 
      struct index_alias<5,X0,X1,X2,X3,X4,X5,X6> { typedef X5 type; };

      template<class X0,class X1,class X2,class X3,class X4,class X5,class X6> 
      struct index_alias<6,X0,X1,X2,X3,X4,X5,X6> { typedef X6 type; };

      template<int Dim> struct Ix { 
	typedef typename index_alias<Dim,I0,I1,I2,I3,I4,I5,I6>::type type; 
      };

      // Similarly, the following enables us to return not just the
      // type but also a reference to the actual index object via
      // index_object_<Dim>()
      template <int Dim> typename enable_if<Dim == 0, const I0&>::type
      index_object_() const { return i0_; }
      template <int Dim> typename enable_if<Dim == 1, const I1&>::type
      index_object_() const { return i1_; }
      template <int Dim> typename enable_if<Dim == 2, const I2&>::type
      index_object_() const { return i2_; }
      template <int Dim> typename enable_if<Dim == 3, const I3&>::type
      index_object_() const { return i3_; }
      template <int Dim> typename enable_if<Dim == 4, const I4&>::type
      index_object_() const { return i4_; }
      template <int Dim> typename enable_if<Dim == 5, const I5&>::type
      index_object_() const { return i5_; }
      template <int Dim> typename enable_if<Dim == 6, const I6&>::type
      index_object_() const { return i6_; }

      // The following sub-class "fastest_varying" enables the
      // definition of "a_fastest_varying_dim" static constant integer
      // that contains the dimension of Array that varies fastest when
      // progessing through memory and is not a singleton.  This
      // corresponds to the dimension "Rank-1" of IndexedArray.
      template<int Dim, class X0,class X1,class X2,
	       class X3,class X4,class X5,class X6> 
      struct fastest_varying {
	static const int value
	  = is_scalar_int<typename index_alias<Dim,X0,X1,X2,X3,X4,X5,X6>::type>::value 
	  ? fastest_varying<Dim-1,X0,X1,X2,X3,X4,X5,X6>::value
	  : Dim;
      };
      template<class X0,class X1,class X2,class X3,class X4,class X5,class X6> 
      struct fastest_varying<0,X0,X1,X2,X3,X4,X5,X6> {
	static const int value = 0;
      };

      static const int a_fastest_varying_dim 
        = fastest_varying<6,I0,I1,I2,I3,I4,I5,I6>::value;

      // ---------------------------------------------------------------------
      // Section 5.8. IndexedArray: Data
      // ---------------------------------------------------------------------
      // Reference to the array being indexed
      ArrayType& a_;
      // Individual indices to up to seven dimensions
      const I0& i0_;
      const I1& i1_;
      const I2& i2_;
      const I3& i3_;
      const I4& i4_;
      const I5& i5_;
      const I6& i6_;
      // Dimensions of the array being indexed (cannot be a reference
      // because FixedArrays do not store their dimensions explicitly)
      ExpressionSize<ArrayType::rank> a_dims_;
      // Dimensions of the IndexedArray
      ExpressionSize<Rank> dimensions_;
      // Separation of elements of the array objects in the dimension
      // that varies fastests
      Index last_offset_;

    }; // End class IndexedArray

  } // End namespace internal
  
} // End namespace adept

#endif 
