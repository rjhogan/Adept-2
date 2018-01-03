/* RangeIndex.h -- Helper classes to enable indexing of arrays

    Copyright (C) 2015-2017 European Centre for Medium-Range Weather Forecasts

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

   This file defines the RangeIndex class and associated helper types
   that facilitate the second case.  A RangeIndex object expresses a
   sequence of regularly spaced integers, which may have a separation
   greater than 1 or a negative separation.  Since an Array need not
   be contiguous in memory, when an Array is indexed by one or more
   RangeIndex objects the result is also a valid Array.  RangeIndex
   objects are created by the range(begin,end) and
   stride(begin,end,stride) functions.

   This file also includes the EndIndex class to enable the use of
   "end" to express the final element of an array dimension being
   indexed (as in Matlab), and the AllIndex class to enable the use of
   "__" to express all elements of a dimension (as ":" in Fortran 90
   and Matlab).

*/


#ifndef AdeptRangeIndex_H
#define AdeptRangeIndex_H 1

#include <adept/Expression.h>

namespace adept {

  namespace internal {
    // ---------------------------------------------------------------------
    // Section 1. EndIndex: enable Matlab-like "end" indexing
    // ---------------------------------------------------------------------

    // When an integer Expression is used as the index to another
    // expression, make "end" (or "adept::end") be interpretted as the
    // index of the final element of the array dimension being
    // referred to. If an whole multi-dimensional array is referred to
    // by a single integer Expression, then "end" is resolved to the
    // len-1 ("len" being the length of the dimension being indexed).
    // "end" is actually an instantiation of the "EndIndex" class, a
    // rank-0 expression.
    struct EndIndex : public Expression<Index, EndIndex>
    {
      // Static definitions
      static const int  rank       = 0;
      static const bool is_active  = false;
      static const int  n_scratch  = 0;
      static const int  n_arrays   = 0;
      static const int  n_active   = 0;
      
      // Functions to implement Expression behaviour

      bool get_dimensions_(ExpressionSize<0>& dim) const
      { return true; }
      
      std::string expression_string_() const
      { return std::string("end"); }

      bool is_aliased_(const Index* mem1, const Index* mem2) const
      { return false; }

      Index value_with_len_(const Index& j, const Index& len) const
      { return len-1; }

      // Note that "end" can only be used as an index to an array or
      // expression: when used in any other context it will fail.
      template <int Rank>
      Index value_at_location_(const ExpressionSize<Rank>&) const
      { throw array_exception("Cannot determine to which object the \"end\" index refers to"
			      ADEPT_EXCEPTION_LOCATION); }
    };
    
    // ---------------------------------------------------------------------
    // Section 2. get_index_with_len
    // ---------------------------------------------------------------------
    // We want range(x,y) and stride(x,y,z) to work for integer
    // arguments or for 0-rank expressions (including "end" and
    // constructs such as "end - 1"), so define the following helper
    // function. For an integer first argument, "get_index_with_len"
    // just returns the first argument, but for 0-rank expressions of
    // integer type, the second argument "len" is passed in and if the
    // expression contains an "end" then this resolves to len-1.

#ifndef ADEPT_BOUNDS_CHECKING
    inline Index get_index_with_len(Index j, Index) { return j; }

    template <typename T, class E>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer
		       && E::rank == 0, Index>::type
    get_index_with_len(const Expression<T,E>& j, Index len) {
      return j.value_with_len(0, len);
    }
#else
    // Bounds-checking versions
    inline Index get_index_with_len(Index j, Index len) {
      if (j < 0 || j >= len) {
	throw index_out_of_bounds();
      }
      else {
	return j; 
      }
    }

    template <typename T, class E>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer
		       && E::rank == 0, Index>::type
    get_index_with_len(const Expression<T,E>& j, Index len) {
      Index ind = j.value_with_len(0, len);
      if (ind < 0 || ind >= len) {
	throw index_out_of_bounds("Array index (probably generated from a scalar expression containing \"end\") is out of bounds"
				  ADEPT_EXCEPTION_LOCATION);
      }
      else {
	return ind;
      }
    }
#endif

    // get_stride_with_len is just like get_index_with_len except that
    // there is no need to do bounds checking
    inline Index get_stride_with_len(Index j, Index) { return j; }

    template <typename T, class E>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer
		       && E::rank == 0, Index>::type
    get_stride_with_len(const Expression<T,E>& j, Index len) {
      return j.value_with_len(0, len);
    }

    // ---------------------------------------------------------------------
    // Section 3. get_value
    // ---------------------------------------------------------------------
    // If a RangeIndex object is not to be used as an index to an
    // array, we may wish to access its elements without consideration
    // of the length of a dimension.

    inline Index get_value(Index j) { return j; }

    template <typename T, class E>
    inline
    typename enable_if<std::numeric_limits<T>::is_integer
		       && E::rank == 0, Index>::type
    get_value(const Expression<T,E>& j) {
      return j.scalar_value();
    }

    // ---------------------------------------------------------------------
    // Section 3. RangeIndex class
    // ---------------------------------------------------------------------
    // A class to store a range of integers, optionally with a fixed
    // stride, for simple indexing of arrays. 
    template<class BeginType, class EndType, class StrideType>
    class RangeIndex
      : public Expression<Index, RangeIndex<BeginType, EndType, StrideType> >
    {
    public:
      static const int  rank       = 1;
      static const bool is_active  = false;
      static const int  n_scratch  = 0;
      static const int  n_arrays   = 1;
      static const int  n_active   = 0;
      
      // Construct with a specified stride
      RangeIndex(const BeginType& begin, const EndType& end, 
		 const StrideType& stride)
	: begin_(begin), end_(end), stride_(stride)
      { };

      // Construct without a specified stride: defaults to 1
      RangeIndex(const BeginType& begin, const EndType& end)
	: begin_(begin), end_(end), stride_(1)
      { };

      Index size() const 
      { return (end() - begin() + stride()) / stride(); }

      Index size_with_len_(const Index& len) const
      { return (end(len) - begin(len) + stride(len)) / stride(len); }

      bool get_dimensions_(ExpressionSize<1>& dim) const {
	dim[0] = size();
	return true;
      }
      std::string expression_string_() const {
	std::stringstream s;
	s << "(" << begin() << ":" << end();
	Index str = stride();
	if (str != 1) {
	  s << ":" << str;
	}
	s << ")";
	return s.str();
      }

      bool is_aliased_(const Index* mem1, const Index* mem2) const {
	return false;
      }

      bool all_arrays_contiguous_() const { return true; }

      // When this object is used as an index to another, the
      // following version of the function is called, in which the
      // "len" element is specified in order for the "end" index
      // specifier to work
      Index value_with_len_(const Index&j, const Index& len) const 
      { return begin(len) + stride(len)*j; }

      // Advance the location of each array in the expression
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	++loc[MyArrayNum];
      }

      template <int MyArrayNum, int NArrays>
      void set_location_(const ExpressionSize<1>& i, 
			 ExpressionSize<NArrays>& index) const { }

      // Give the value at a particular offset
      template <int MyArrayNum, int NArrays>
      Index value_at_location_(const ExpressionSize<NArrays>& j) const 
      { return begin() + stride()*j[MyArrayNum]; }

      // Access the beginning, end and stride, where the argument
      // gives the length of the dimension in case any of these is
      // expressed with respect to "end" (which resolves to length-1)
      Index begin()  const { return get_value(begin_);  }
      Index end()    const { return get_value(end_);    }
      Index stride() const { return get_value(stride_); }
      Index begin(Index len) const
      {	return get_index_with_len(begin_, len); }
      Index end(Index len) const 
      { return get_index_with_len(end_, len); }
      Index stride(Index len) const
      { return get_stride_with_len(stride_, len); }

    private:
      // Note that a copy rather than a reference to the Expression or
      // int is stored: this is because if range(i1, i2) is used as
      // the index to another object, then a temporary object will be
      // created that will be destroyed immediately after calling the
      // RangeIndex constructor (following ANSI C++ rules), so a
      // reference would then point to invalid data.
      // FIX!!!
      const BeginType begin_;
      const EndType end_;
      const StrideType stride_;
    };

    // ---------------------------------------------------------------------
    // Section 4. AllIndex class
    // ---------------------------------------------------------------------
    // A class to represent all elements along one dimension, for simple
    // indexing of arrays with "__" (equivalent to ":" in Fortran).
    class AllIndex : public Expression<Index, AllIndex>
    {
    public:
      static const int  rank      = 1;
      static const bool is_active = false;
      static const int  n_active  = 0;
      static const int  n_static_ = 0;
      static const int  n_arrays  = 0;

      // Unknown!
      //      bool get_dimensions_(ExpressionSize<1>& dim) const { return true; }      

      std::string expression_string_() const { return std::string("__"); }

      bool is_aliased_(const Index* mem1, const Index* mem2) const { return false; }

      Index size_with_len_(const Index& len) const
      { return len; }

      Index value_with_len_(const Index& j, const Index& len) const
      { return j; }

      Index value_at_location_(const ExpressionSize<1>& loc) const
      { return loc[0]; }
      
      Index begin(Index len = -1) const { return 0; }
      Index end(Index len) const { return len-1; }
      Index stride(Index len = -1) const { return 1; }
    };


    // is_range<T>::value is true if T is of type RangeIndex or
    // AllIndex
    template <typename T>
    struct is_range {
      static const bool value = false;
      static const int  count = 0;
    };
    template <>
    struct is_range<AllIndex> {
      static const bool value = true;
      static const int  count = 1;
    };
    template <class B, class E, class S>
    struct is_range<RangeIndex<B,E,S> > {
      static const bool value = true;
      static const int  count = 1;
    };
    
    // is_regular_index<T>::value is true if T is a valid index to a
    // dimension of an Array such that the indexed object is also an
    // Array
    template <typename T>
    struct is_regular_index {
      static const bool value = (is_scalar_int<T>::value
				 || is_null_type<T>::value
				 || is_range<T>::value);
    };

    // is_ranged<>::value is true if at least one of the template
    // arguments I0 to I[Rank-1] is of type RangeIndex, and all others
    // are of integer type
    template <int Rank, typename I0, typename I1 = null_type, 
	      typename I2 = null_type, typename I3 = null_type,
	      typename I4 = null_type, typename I5 = null_type,
	      typename I6 = null_type>
    struct is_ranged {
      static const bool value = (is_range<I0>::value || is_range<I1>::value
			      || is_range<I2>::value || is_range<I3>::value
			      || is_range<I4>::value || is_range<I5>::value
			      || is_range<I6>::value)
	&& Rank == 7 - (  is_null_type<I1>::count + is_null_type<I2>::count
			+ is_null_type<I3>::count + is_null_type<I4>::count
			+ is_null_type<I5>::count + is_null_type<I6>::count)
	&& (   is_regular_index<I0>::value && is_regular_index<I1>::value
	    && is_regular_index<I2>::value && is_regular_index<I3>::value
	    && is_regular_index<I4>::value && is_regular_index<I5>::value
	    && is_regular_index<I6>::value);
      static const int count = is_range<I0>::count + is_range<I1>::count
	+ is_range<I2>::count + is_range<I3>::count + is_range<I4>::count
	+ is_range<I5>::count + is_range<I6>::count;
    };




  } // End namespace internal

  // User-accessible functions and objects

  // The actual end object is held in a source file
  extern ::adept::internal::EndIndex end;

  // The actual "__" object is held in a source file
  extern ::adept::internal::AllIndex __;

  // Return a RangeIndex object representing all the integers between
  // "begin" and "end"; the inputs can either be Expressions or ints
  template<class BeginType, class EndType>
  inline
  adept::internal::RangeIndex<BeginType, EndType, int>
  range(const BeginType& begin, const EndType& end)
  {
    return adept::internal::RangeIndex<BeginType, EndType, int>(begin, end, 1);
  }

  // Return a RangeIndex object representing integers between "begin"
  // and "end" spaced "stride" apart
  template<class BeginType, class EndType, class StrideType>
  inline
  adept::internal::RangeIndex<BeginType, EndType, StrideType>
  stride(const BeginType& begin, const EndType& end,
	 const StrideType& stride)
  {
    return adept::internal::RangeIndex<BeginType, EndType, 
				       StrideType>(begin, end, stride);
  }



} // End namespace adept

#endif
