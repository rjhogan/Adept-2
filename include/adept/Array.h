/* Array.h -- active or inactive Array of arbitrary rank

    Copyright (C) 2014-2018 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   The Array class has functionality modelled on Fortran-90 arrays -
   they can have a rank up to 7 (above will work, but some forms of
   indexing these arrays will not work).

*/

#ifndef AdeptArray_H
#define AdeptArray_H 1

#include <iostream>
#include <sstream>
#include <limits>
#include <string>

#include <adept/base.h>

#ifdef ADEPT_CXX11_FEATURES
#include <initializer_list>
#endif

#include <adept/Storage.h>
#include <adept/Expression.h>
#include <adept/RangeIndex.h>
#include <adept/ActiveReference.h>
#include <adept/ActiveConstReference.h>
#include <adept/IndexedArray.h>
#include <adept/where.h>
#include <adept/noalias.h>

namespace adept {
  using namespace adept::internal;

  enum ArrayPrintStyle {
    PRINT_STYLE_PLAIN,
    PRINT_STYLE_CSV,
    PRINT_STYLE_CURLY,
    PRINT_STYLE_MATLAB,
  };

  namespace internal {

    enum MatrixStorageOrder {
      ROW_MAJOR=0, COL_MAJOR=1
    };

  }

  // Forward declarations to enable diag_matrix
  template <typename, class, bool> class SpecialMatrix;
  namespace internal {
    template <MatrixStorageOrder, Index, Index> struct BandEngine;
  }

  // Forward declaration to enable linking at construction and via
  // link to FixedArray
  template <typename, bool, Index, Index, Index, Index, Index, Index, Index>
  class FixedArray;

  namespace internal {

    // -------------------------------------------------------------------
    // Global variables
    // -------------------------------------------------------------------
    // The following global variables affect the behaviour of the
    // Array class, and are modified using set_*

    // This is "true" by default: row-major is the normal C/C++
    // convention
    extern bool array_row_major_order;

    // When arrays are sent to a stream the dimensions can be grouped
    // with curly brackets
    //    extern bool array_print_curly_brackets;

    // Variables describing how arrays are written to a stream
    extern ArrayPrintStyle array_print_style;
    extern std::string vector_separator;
    extern std::string vector_print_before;
    extern std::string vector_print_after;
    extern std::string array_opening_bracket;
    extern std::string array_closing_bracket;
    extern std::string array_contiguous_separator;
    extern std::string array_non_contiguous_separator;
    extern std::string array_print_before;
    extern std::string array_print_after;
    extern std::string array_print_empty_before;
    extern std::string array_print_empty_after;
    extern bool array_print_indent;
    extern bool array_print_empty_rank;

    // Forward declaration to enable Array::where()
    //    template <class A, class B> class Where;

    // -------------------------------------------------------------------
    // Helper classes
    // -------------------------------------------------------------------

    // The following are used by expression_string()
    template <int Rank, bool IsActive>
    struct array_helper            { const char* name() { return "Array";  } };
    template <int Rank>
    struct array_helper<Rank,true> { const char* name() { return "aArray";  } };

    template <>
    struct array_helper<1,false>   { const char* name() { return "Vector"; } };
    template <>
    struct array_helper<1,true>    { const char* name() { return "aVector"; } };

    template <>
    struct array_helper<2,false>   { const char* name() { return "Matrix"; } };
    template <>
    struct array_helper<2,true>    { const char* name() { return "aMatrix"; } };

    // Arrays inherit from this class to provide optional storage of
    // the gradient index of the first value of the array depending on
    // whether the array is active or not
    template <bool IsActive>
    struct GradientIndex {
      // Constructor used when linking to existing data where gradient
      // index is known
      GradientIndex(Index val = -9999) : value_(val) { }
      // Constructor used for fixed array objects where length is
      // known
      GradientIndex(Index n, bool) : value_(ADEPT_ACTIVE_STACK->register_gradients(n)) { }
      GradientIndex(Index val, Index offset) : value_(val+offset) { }
      Index get() const { return value_; }
      void set(Index val) { value_ = val; }
      void clear() { value_ = -9999; }
      template <typename Type>
      void set(const Type* data, const Storage<Type>* storage) {
	value_ = (storage->gradient_index() + (data - storage->data()));
      }
      void assert_inactive() {
	throw invalid_operation("Operation applied that is invalid with active arrays"
				ADEPT_EXCEPTION_LOCATION);
      }
      void unregister(Index n) { ADEPT_ACTIVE_STACK->unregister_gradients(value_, n); }
#ifdef ADEPT_MOVE_SEMANTICS
      void swap(GradientIndex& rhs) noexcept {
	Index tmp_value = rhs.get();
	rhs.set(value_);
	value_ = tmp_value;
      }
#endif
    private:
      Index value_;
    };

    template <>
    struct GradientIndex<false> {
      GradientIndex(Index val = -9999) { }
      GradientIndex(Index, bool) { }
      GradientIndex(Index val, Index offset) { }
      Index get() const { return -9999; }
      void set(Index val) { }
      void clear() { }
      template <typename Type>
      void set(const Type* data, const Storage<Type>* storage) { }
      void assert_inactive() { }
      void unregister(Index) { }
#ifdef ADEPT_MOVE_SEMANTICS
      void swap(GradientIndex& rhs) noexcept { }
#endif
    };


  } // End namespace internal


  // -------------------------------------------------------------------
  // Definition of Array class
  // -------------------------------------------------------------------
  template<int Rank, typename Type = Real, bool IsActive = false>
  class Array
    : public Expression<Type,Array<Rank,Type,IsActive> >,
      protected internal::GradientIndex<IsActive> {

  public:
    // -------------------------------------------------------------------
    // Array: 1. Static Definitions
    // -------------------------------------------------------------------

    // The Expression base class needs access to some protected member
    // functions in section 5
    friend struct Expression<Type,Array<Rank,Type,IsActive> >;

    // Static definitions to enable the properties of this type of
    // expression to be discerned at compile time
    static const bool is_active  = IsActive;
    static const bool is_lvalue  = true;
    static const int  rank       = Rank;
    static const int  n_active   = IsActive * (1 + is_complex<Type>::value);
    static const int  n_scratch  = 0;
    static const int  n_arrays   = 1;
    static const bool is_vectorizable = Packet<Type>::is_vectorized;

    // -------------------------------------------------------------------
    // Array: 2. Constructors
    // -------------------------------------------------------------------
    
    // Initialize an empty array
    Array() : data_(0), storage_(0), dimensions_(0)
    { ADEPT_STATIC_ASSERT(!(std::numeric_limits<Type>::is_integer
			    && IsActive), CANNOT_CREATE_ACTIVE_ARRAY_OF_INTEGERS); }

    // Initialize an array with specified size
    Array(const Index* dims) : storage_(0)
    { resize(dims); }
    Array(const ExpressionSize<Rank>& dims) : storage_(0)
    { resize(dims); }

    // A way to only enable construction if the correct number of
    // arguments is provided (resize_<x> is only defined for x==Rank)
    Array(Index m0) : storage_(0) { resize_<1>(m0); }
    Array(Index m0, Index m1) : storage_(0) { resize_<2>(m0,m1); }
    Array(Index m0, Index m1, Index m2) : storage_(0) { resize_<3>(m0,m1,m2); }
    Array(Index m0, Index m1, Index m2, Index m3) : storage_(0) 
    { resize_<4>(m0,m1,m2,m3); }
    Array(Index m0, Index m1, Index m2, Index m3, Index m4)  : storage_(0)
    { resize_<5>(m0,m1,m2,m3,m4); }
    Array(Index m0, Index m1, Index m2, Index m3, Index m4, Index m5)  : storage_(0)
    { resize_<6>(m0,m1,m2,m3,m4,m5); }
    Array(Index m0, Index m1, Index m2, Index m3, Index m4, Index m5, Index m6) 
      : storage_(0) 
    { resize_<7>(m0,m1,m2,m3,m4,m5,m6); }

    // A way to directly create arrays, needed when subsetting
    // other arrays
    Array(Type* data, Storage<Type>* s, const ExpressionSize<Rank>& dims,
	  const ExpressionSize<Rank>& offset)
      : data_(data), storage_(s), dimensions_(dims), offset_(offset) { 
      if (storage_) {
	storage_->add_link(); 
	GradientIndex<IsActive>::set(data_, storage_);
      }
      else {
	// Active arrays need a gradient index so it is an error for
	// them to get to this point
	GradientIndex<IsActive>::assert_inactive();
      }
    }

    // Similar to the above, but with the gradient index supplied explicitly,
    // needed when an active FixedArray is being sliced, which
    // produces an active Array
    Array(const Type* data0, Index data_offset, const ExpressionSize<Rank>& dims,
	  const ExpressionSize<Rank>& offset, Index gradient_index0)
      : GradientIndex<IsActive>(gradient_index0, data_offset),
	data_(const_cast<Type*>(data0)+data_offset), storage_(0), dimensions_(dims), offset_(offset) { }

    // Initialize an array pointing at existing data: the fact that
    // storage_ is a null pointer is used to convey the information
    // that it is not necessary to deallocate the data when this array
    // is destructed
    Array(Type* data, const ExpressionSize<Rank>& dims)
      : data_(data), storage_(0), dimensions_(dims) {
      ADEPT_STATIC_ASSERT(!IsActive, CANNOT_CONSTRUCT_ACTIVE_ARRAY_WITHOUT_GRADIENT_INDEX);
      // Active arrays need a gradient index so it is an error for
      // them to get to this point
      GradientIndex<IsActive>::assert_inactive();
      pack_contiguous_(); 
    }

    // Copy constructor: links to the source data rather than copying
    // it.  This is needed because we want a function returning an
    // Array not to make a deep copy, but rather to perform a
    // (computationally cheaper) shallow copy; when the Array within
    // the function is destructed, it will remove its link to the
    // data, and the responsibility for deallocating the data will
    // then pass to the Array in the calling function.
    Array(Array& rhs) 
      : GradientIndex<IsActive>(rhs.gradient_index()), 
	data_(rhs.data()), storage_(rhs.storage()), 
	dimensions_(rhs.dimensions()), offset_(rhs.offset())
    {
      if (storage_) storage_->add_link(); 
#ifdef ADEPT_VERBOSE_FUNCTIONS
      std::cout << "  running constructor Array(Array&)\n";
#endif
    }

    // Copy constructor with const argument does exactly the same
    // thing
    Array(const Array& rhs) 
      : GradientIndex<IsActive>(rhs.gradient_index()),
	dimensions_(rhs.dimensions()), offset_(rhs.offset())
    { 
      link_(const_cast<Array&>(rhs));
#ifdef ADEPT_VERBOSE_FUNCTIONS
      std::cout << "  running constructor Array(const Array&)\n";
#endif
    }
  private:
    void link_(Array& rhs) {
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
    Array(const Expression<EType, E>& rhs,
	  typename enable_if<E::rank == Rank && (Rank > 0),int>::type = 0)
      : data_(0), storage_(0), dimensions_(0)
    {
#ifdef ADEPT_VERBOSE_FUNCTIONS
      std::cout << "  running constructor Array(const Expression&), implemented by assignment\n";
#endif
      *this = rhs; 
    }

#ifdef ADEPT_CXX11_FEATURES
    // Initialize from initializer list
    template <typename T>
    Array(std::initializer_list<T> list) : data_(0), storage_(0), dimensions_(0) {
      *this = list;
    }

    // The unfortunate restrictions on initializer_list constructors
    // mean that each possible Array rank needs explicit treatment
    template <typename T>
    Array(std::initializer_list<
	  std::initializer_list<T> > list)
      : data_(0), storage_(0), dimensions_(0) { *this = list; }

    template <typename T>
    Array(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > list)
      : data_(0), storage_(0), dimensions_(0) { *this = list; }

    template <typename T>
    Array(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > > list)
      : data_(0), storage_(0), dimensions_(0) { *this = list; }

    template <typename T>
    Array(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > > > list)
      : data_(0), storage_(0), dimensions_(0) { *this = list; }

    template <typename T>
    Array(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > > > > list)
      : data_(0), storage_(0), dimensions_(0) { *this = list; }

    template <typename T>
    Array(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > > > > > list)
      : data_(0), storage_(0), dimensions_(0) { *this = list; }
    

#endif


    // Destructor: if the data are stored in a Storage object then we
    // tell it that one fewer object is linking to it; if the number
    // of links to it drops to zero, it will destruct itself and
    // deallocate the memory.
    ~Array()
    { if (storage_) storage_->remove_link(); }

    // -------------------------------------------------------------------
    // Array: 3. Assignment operators
    // -------------------------------------------------------------------

    // Assignment to another matrix: copy the data...
    // Ideally we would like this to fall back to the operator=(const
    // Expression&) function, but if we don't define a copy assignment
    // operator then C++ will generate a default one :-(
    Array& operator=(const Array& rhs) {
#ifdef ADEPT_VERBOSE_FUNCTIONS
      std::cout << "  running Array::operator=(const Array&), implemented with operator=(const Expression&)\n";
#endif
      return (*this = static_cast<const Expression<Type,Array>&> (rhs));
    }

#ifdef ADEPT_MOVE_SEMANTICS
    Array& operator=(Array&& rhs) {
#ifdef ADEPT_VERBOSE_FUNCTIONS
      std::cout << "  running Array::operator=(Array&&)\n";
#endif
      // A fast "swap" operation can be performed only if the present
      // ("this") array is either empty, or its data is contained in a
      // Storage object with only one link to it (corresponding to the
      // present array). We may not perform a swap if its data is not
      // in a Storage object, since it might be linked to another
      // location that is expecting the result of the assignment to
      // change the data in that location. We also require that the
      // RHS data would otherwise be lost (but it is not clear that
      // this is necessary).
      if ((empty() || (storage_ && storage_->n_links() == 1))
	  && (!rhs.storage() || rhs.storage()->n_links() == 1)) {
	// We still need to check that the dimensions match
	if (empty() || compatible(dimensions_, rhs.dimensions())) {
	  swap(*this, rhs);
	}
	else {
	  std::string str = rhs.expression_string()
	    + " assigned to " + expression_string_();
	  throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
	}
      }
      else {
	// Need a full copy because other arrays are linked to the
	// Storage object
	*this = static_cast<const Expression<Type,Array>&> (rhs);
      }
      return *this;
    }

    friend void swap(Array& l, Array& r) noexcept {
#ifdef ADEPT_VERBOSE_FUNCTIONS
      std::cout << "  running swap(Array&,Array&)\n";
#endif
      Type* tmp_data = l.data_;
      l.data_ = r.data_;
      r.data_ = tmp_data;
      Storage<Type>* tmp_storage = l.storage_;
      l.storage_ = r.storage_;
      r.storage_ = tmp_storage;
      swap(l.dimensions_, r.dimensions_);
      swap(l.offset_, r.offset_);
      static_cast<GradientIndex<IsActive>&>(l).swap(static_cast<GradientIndex<IsActive>&>(r));
    }

#endif


    // Assignment to an array expression of the same rank
    template <typename EType, class E>
    inline //__attribute__((always_inline))
    typename enable_if<E::rank == Rank, Array&>::type
    operator=(const Expression<EType,E>&  __restrict rhs) {
#ifdef ADEPT_VERBOSE_FUNCTIONS
      std::cout << "  running Array::operator=(const Expression&)\n";
#endif
#ifndef ADEPT_NO_DIMENSION_CHECKING
      ExpressionSize<Rank> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (empty()) {
	resize(dims);
      }
      else if (!compatible(dims, dimensions_)) {
	std::string str = "Expr";
	str += dims.str() + " object assigned to " + expression_string_();
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
#else
      if (empty()) {
	ExpressionSize<Rank> dims;
	if (!rhs.get_dimensions(dims)) {
	  std::string str = "Array size mismatch in "
	    + rhs.expression_string() + ".";
	  throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
	}	
	resize(dims);
      }
#endif
      if (!empty()) {
#ifndef ADEPT_NO_ALIAS_CHECKING
	// Check for aliasing first
	Type const * ptr_begin;
	Type const * ptr_end;
	data_range(ptr_begin, ptr_end);
	if (rhs.is_aliased(ptr_begin, ptr_end)) {
	  Array<Rank,Type,IsActive> copy;
	  // It would be nice to wrap noalias around rhs, but then
	  // this leads to infinite template recursion since the "="
	  // operator calls the current function but with a modified
	  // expression type. perhaps a better way would be to make
	  // copy.assign_no_alias(rhs) work.
	  copy = rhs;
	  assign_expression_<Rank, IsActive, E::is_active>(copy);
	}
	else {
#endif
	  // Select active/passive version by delegating to a
	  // protected function
	  // The cast() is needed because assign_expression_ accepts
	  // its argument by value
	  assign_expression_<Rank, IsActive, E::is_active>(rhs.cast());
#ifndef ADEPT_NO_ALIAS_CHECKING
	}
#endif
      }
      return *this;
    }


    // Assignment to an array expression of the same rank in which the
    // activeness of the right-hand-side is ignored
    template <typename EType, class E>
    typename enable_if<E::rank == Rank, Array&>::type
    assign_inactive(const Expression<EType,E>& rhs) {
      ExpressionSize<Rank> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (empty()) {
	resize(dims);
      }
      else if (!compatible(dims, dimensions_)) {
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
	  Array<Rank,Type,IsActive> copy;
	  copy.assign_inactive(rhs);
	  //	  *this = copy;
	  assign_expression_<Rank, IsActive, false>(copy);
	}
	else {
	  assign_expression_<Rank, IsActive, false>(rhs.cast());
	}
      }
      return *this;
    }

    // Assignment to a single value copies to every element
    template <typename RType>
    typename enable_if<is_not_expression<RType>::value
                       // FIX
                       || internal::is_active<Type>::value
		       , Array&>::type
    operator=(RType rhs) {
      if (!empty()) {
	assign_inactive_scalar_<Rank,IsActive>(rhs);
      }
      return *this;
    }

    // Assign active scalar expression to an active array by first
    // converting the RHS to an active scalar
    template <typename EType, class E>
    typename enable_if<E::rank == 0 && (Rank > 0) && IsActive && !E::is_lvalue,
      Array&>::type
    operator=(const Expression<EType,E>& rhs) {
      Active<EType> x = rhs;
      *this = x;
      return *this;
    }

    // Assign an active scalar to an active array
    template <typename PType>
    // FIX
    typename enable_if<!internal::is_active<PType>::value && IsActive, Array&>::type
    //    Array& 
    operator=(const Active<PType>& rhs) {
      ADEPT_STATIC_ASSERT(IsActive, ATTEMPT_TO_ASSIGN_ACTIVE_SCALAR_TO_INACTIVE_ARRAY);
      if (!empty()) {
#ifdef ADEPT_RECORDING_PAUSABLE
	if (!ADEPT_ACTIVE_STACK->is_recording()) {
	  assign_inactive_scalar_<Rank,false>(rhs.scalar_value());
	  return *this;
	}
#endif
	ExpressionSize<Rank> i(0);
	Index index = 0;
	int my_rank;
	static const int last = Rank-1;
	// In case PType != Type we make a local copy to minimize type
	// conversions
	Type val = rhs.scalar_value();
	
	ADEPT_ACTIVE_STACK->check_space(size());
	do {
	  i[last] = 0;
	  // Innermost loop
	  for ( ; i[last] < dimensions_[last]; ++i[last],
		  index += offset_[last]) {
	    data_[index] = val;
	    ADEPT_ACTIVE_STACK->push_rhs(1.0, rhs.gradient_index());
	    ADEPT_ACTIVE_STACK->push_lhs(gradient_index()+index);
	  }
	  advance_index(index, my_rank, i);
	} while (my_rank >= 0);
      }
      return *this;
    }

#define ADEPT_DEFINE_OPERATOR(OPERATOR, OPSYMBOL)		\
    template <class RType>				\
    Array& OPERATOR(const RType& rhs) {			\
      return *this = noalias(*this OPSYMBOL rhs);	\
    }
    ADEPT_DEFINE_OPERATOR(operator+=, +);
    ADEPT_DEFINE_OPERATOR(operator-=, -);
    ADEPT_DEFINE_OPERATOR(operator*=, *);
    ADEPT_DEFINE_OPERATOR(operator/=, /);
  //    ADEPT_DEFINE_OPERATOR(operator&=, &);
  //    ADEPT_DEFINE_OPERATOR(operator|=, |);
#undef ADEPT_DEFINE_OPERATOR

    // Enable the A.where(B) = C construct.

    // Firstly implement the A.where(B) to return a "Where<A,B>" object
    template <class B>
    typename enable_if<B::rank == Rank, Where<Array,B> >::type
    where(const Expression<bool,B>& bool_expr) {
#ifndef ADEPT_NO_DIMENSION_CHECKING
      ExpressionSize<Rank> dims;
      if (!bool_expr.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + bool_expr.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (dims != dimensions_) {
	throw size_mismatch("Boolean expression of different size"
			    ADEPT_EXCEPTION_LOCATION);
      }
#endif
      return Where<Array,B>(*this, bool_expr.cast());
    }
    
    // When Where<A,B> = C is invoked, it calls
    // A.assign_conditional(B,C). This is implemented separately for
    // the case when C is an inactive scalar and when it is an array
    // expression.
    template <class B, typename C>
    typename enable_if<is_not_expression<C>::value, void>::type
    assign_conditional(const Expression<bool,B>& bool_expr,
			    C rhs) {
      if (!empty()) {
	assign_conditional_inactive_scalar_<IsActive>(bool_expr, rhs);
      }
    }

    template <class B, typename T, class C>
    void assign_conditional(const Expression<bool,B>& bool_expr,
			    const Expression<T,C>& rhs) {
      // Assume size of bool_expr already checked
#ifndef ADEPT_NO_DIMENSION_CHECKING
      ExpressionSize<Rank> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (!compatible(dims,dimensions_)) {
	throw size_mismatch("Right-hand-side of \"where\" construct of incompatible size"
			    ADEPT_EXCEPTION_LOCATION);
      }
#endif
      // Check for aliasing first
      Type const * ptr_begin;
      Type const * ptr_end;
      data_range(ptr_begin, ptr_end);
      if (rhs.is_aliased(ptr_begin, ptr_end)) {
	Array<Rank,Type,IsActive> copy;
	copy = rhs;
	assign_conditional_<IsActive>(bool_expr.cast(), copy);
      }
      else {
	// Select active/passive version by delegating to a
	// protected function
	assign_conditional_<IsActive>(bool_expr.cast(), rhs.cast());
      }
      //      return *this;
    }

#ifdef ADEPT_CXX11_FEATURES
    // Assignment of an Array to an initializer list; the first ought
    // to only work for Vectors
    template <typename T>
    typename enable_if<std::is_convertible<T,Type>::value, Array&>::type
    operator=(std::initializer_list<T> list) {
      ADEPT_STATIC_ASSERT(Rank==1,RANK_MISMATCH_IN_INITIALIZER_LIST);

      if (empty()) {
	resize(list.size());
      }
      else if (list.size() > static_cast<std::size_t>(dimensions_[0])) {
	throw size_mismatch("Initializer list is larger than Vector in assignment"
			    ADEPT_EXCEPTION_LOCATION);
      }
      // Zero the whole array first in order that automatic
      // differentiation works
      *this = 0;
      Index index = 0;
      for (auto i = std::begin(list); i < std::end(list); ++i,
	   ++index) {
	data_[index*offset_[0]] = *i;	
      }
      return *this;
    }

    // Assignment of a higher rank Array to a list of lists...
    template <class IType>
    Array& operator=(std::initializer_list<std::initializer_list<IType> > list) {
      ADEPT_STATIC_ASSERT(Rank==initializer_list_rank<IType>::value+2,
      			  RANK_MISMATCH_IN_INITIALIZER_LIST);
      if (empty()) {
	Index dims[ADEPT_MAX_ARRAY_DIMENSIONS];
	int ndims = 0;
	shape_initializer_list_(list, dims, ndims);
	resize(dims);
      }
      else if (list.size() > static_cast<std::size_t>(dimensions_[0])) {
	throw size_mismatch("Multi-dimensional initializer list larger than slowest-varying dimension of Array"
			    ADEPT_EXCEPTION_LOCATION);
      }
      Index index = 0;
      for (auto i = std::begin(list); i < std::end(list); ++i,
	   ++index) {
	(*this)[index] = *i;
      }
      return *this;
    }


  protected:
    template <typename T>
    typename enable_if<std::is_convertible<T,Type>::value>::type
    shape_initializer_list_(std::initializer_list<T> list,
			    Index* dims, int& ndims) const {
      dims[ndims] = list.size();
      ndims++;
    }
    template <class IType>
    void
    shape_initializer_list_(std::initializer_list<std::initializer_list<IType> > list,
			    Index* dims, int& ndims) const {
      dims[ndims] = list.size();
      ndims++;
      shape_initializer_list_(*(list.begin()), dims, ndims);
    }


  public:

#endif


  
    // -------------------------------------------------------------------
    // Array: 4. Access functions, particularly operator()
    // -------------------------------------------------------------------
  
    // Get l-value of the element at the specified coordinates
    typename active_reference<Type,IsActive>::type
    get_lvalue(const ExpressionSize<Rank>& i) {
      return get_lvalue_<IsActive>(index_(i));
    }
    
    typename active_scalar<Type,IsActive>::type
    get_rvalue(const ExpressionSize<Rank>& i) const {
      return get_rvalue_<IsActive>(index_(i));
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

    template <bool MyIsActive>
    typename enable_if<MyIsActive, Active<Type> >::type
    get_rvalue_(const Index& loc) const {
      return Active<Type>(data_[loc], gradient_index()+loc);
    }
    template <bool MyIsActive>
    typename enable_if<!MyIsActive, const Type&>::type
    get_rvalue_(const Index& loc) const {
      return data_[loc];
    }

  public:
    // Get a constant reference to the element at the specified
    // location, ignoring whether it is active or not
    //    const Type& get(const ExpressionSize<Rank>& i) const {
    //      return data_[index_(i)];
    //    }

    // The following provide a way to access individual elements of
    // the array.  There must be the same number of arguments to
    // operator() as the rank of the array.  Each argument must be of
    // integer type, or a rank-0 expression of integer type (such as
    // "end" or "end-3"). Inactive arrays return a reference to the
    // element, while active arrays return an ActiveReference<Type>
    // object.  Up to 7 dimensions are supported.

    // l-value access to inactive array with function-call operator
    template <typename I0>
    typename enable_if<Rank==1 && all_scalar_ints<1,I0>::value && !IsActive, Type&>::type
    operator()(I0 i0) 
    { return data_[get_index_with_len(i0,dimensions_[0])*offset_[0]]; }

    // r-value access to inactive array with function-call operator
    template <typename I0>
    typename enable_if<Rank==1 && all_scalar_ints<1,I0>::value && !IsActive, const Type&>::type
    operator()(I0 i0) const
    { return data_[get_index_with_len(i0,dimensions_[0])*offset_[0]]; }

    // l-value access to inactive array with element-access operator
    template <typename I0>
    typename enable_if<Rank==1 && all_scalar_ints<1,I0>::value && !IsActive, Type&>::type
    operator[](I0 i0) 
    { return data_[get_index_with_len(i0,dimensions_[0])*offset_[0]]; }

    // r-value access to inactive array with element-access operator
    template <typename I0>
    typename enable_if<Rank==1 && all_scalar_ints<1,I0>::value && !IsActive, const Type&>::type
    operator[](I0 i0) const
    { return data_[get_index_with_len(i0,dimensions_[0])*offset_[0]]; }

  protected:
    template <bool MyIsActive>
    typename enable_if<!MyIsActive,Type&>::type
    get_scalar_reference(const Index& offset)
    { return data_[offset]; }

    template <bool MyIsActive>
    typename enable_if<!MyIsActive,const Type&>::type
    get_scalar_reference(const Index& offset) const
    { return data_[offset]; }

    template <bool MyIsActive>
    typename enable_if<MyIsActive,ActiveReference<Type> >::type
    get_scalar_reference(const Index& offset) 
    { return ActiveReference<Type>(data_[offset], gradient_index()+offset); }
    template <bool MyIsActive>
    typename enable_if<MyIsActive,ActiveConstReference<Type> >::type
    get_scalar_reference(const Index& offset) const
    { return ActiveConstReference<Type>(data_[offset], gradient_index()+offset); }

  public:

    // l-value access to active array with function-call operator
    template <typename I0>
    typename enable_if<Rank==1 && all_scalar_ints<1,I0>::value && IsActive,
		       ActiveReference<Type> >::type
    operator()(I0 i0) {
      Index offset = get_index_with_len(i0,dimensions_[0])*offset_[0];
      return ActiveReference<Type>(data_[offset], gradient_index()+offset);
    }
    
    // r-value access to active array with function-call operator
    template <typename I0>
    typename enable_if<Rank==1 && all_scalar_ints<1,I0>::value && IsActive,
		       ActiveConstReference<Type> >::type
    operator()(I0 i0) const {
      Index offset = get_index_with_len(i0,dimensions_[0])*offset_[0];
      return ActiveConstReference<Type>(data_[offset], gradient_index()+offset);
    }

    // l-value access to active array with element-access operator
    template <typename I0>
    typename enable_if<Rank==1 && all_scalar_ints<1,I0>::value && IsActive,
		       ActiveReference<Type> >::type
    operator[](I0 i0) {
      Index offset = get_index_with_len(i0,dimensions_[0])*offset_[0];
      return ActiveReference<Type>(data_[offset], gradient_index()+offset);
    }
    
    // r-value access to active array with element-access operator
    template <typename I0>
    typename enable_if<Rank==1 && all_scalar_ints<1,I0>::value && IsActive,
		       ActiveConstReference<Type> >::type
    operator[](I0 i0) const {
      Index offset = get_index_with_len(i0,dimensions_[0])*offset_[0];
      return ActiveConstReference<Type>(data_[offset], gradient_index()+offset);
    }
    
    // 2D array l-value and r-value access
    template <typename I0, typename I1>
    typename enable_if<Rank==2 && all_scalar_ints<2,I0,I1>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1) {
      return get_scalar_reference<IsActive>(
		    get_index_with_len(i0,dimensions_[0])*offset_[0]
		  + get_index_with_len(i1,dimensions_[1])*offset_[1]);
    }
    template <typename I0, typename I1>
    typename enable_if<Rank==2 && all_scalar_ints<2,I0,I1>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1) const {
      return get_scalar_reference<IsActive>(
		    get_index_with_len(i0,dimensions_[0])*offset_[0]
		  + get_index_with_len(i1,dimensions_[1])*offset_[1]);
    }

    // 3D array l-value and r-value access
    template <typename I0, typename I1, typename I2>
    typename enable_if<Rank==3 && all_scalar_ints<3,I0,I1,I2>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2) {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]);
    }
    template <typename I0, typename I1, typename I2>
    typename enable_if<Rank==3 && all_scalar_ints<3,I0,I1,I2>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2) const {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]);
    }

    // 4D array l-value and r-value access
    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<Rank==4 && all_scalar_ints<4,I0,I1,I2,I3>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3) {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]
		   + get_index_with_len(i3,dimensions_[3])*offset_[3]);
    }
    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<Rank==4 && all_scalar_ints<4,I0,I1,I2,I3>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3) const {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]
		   + get_index_with_len(i3,dimensions_[3])*offset_[3]);
    }

    // 5D array l-value and r-value access
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4>
    typename enable_if<Rank==5 && all_scalar_ints<5,I0,I1,I2,I3,I4>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]
		   + get_index_with_len(i3,dimensions_[3])*offset_[3]
		   + get_index_with_len(i4,dimensions_[4])*offset_[4]);
    }
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4>
    typename enable_if<Rank==5 && all_scalar_ints<5,I0,I1,I2,I3,I4>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]
		   + get_index_with_len(i3,dimensions_[3])*offset_[3]
		   + get_index_with_len(i4,dimensions_[4])*offset_[4]);
    }

    // 6D array l-value and r-value access
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5>
    typename enable_if<Rank==6 && all_scalar_ints<6,I0,I1,I2,I3,I4,I5>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]
		   + get_index_with_len(i3,dimensions_[3])*offset_[3]
		   + get_index_with_len(i4,dimensions_[4])*offset_[4]
		   + get_index_with_len(i5,dimensions_[5])*offset_[5]);
    }
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5>
    typename enable_if<Rank==6 && all_scalar_ints<6,I0,I1,I2,I3,I4,I5>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) const {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]
		   + get_index_with_len(i3,dimensions_[3])*offset_[3]
		   + get_index_with_len(i4,dimensions_[4])*offset_[4]
		   + get_index_with_len(i5,dimensions_[5])*offset_[5]);
    }

    // 7D array l-value and r-value access
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5, typename I6>
    typename enable_if<Rank==7 && all_scalar_ints<7,I0,I1,I2,I3,I4,I5,I6>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6) {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]
		   + get_index_with_len(i3,dimensions_[3])*offset_[3]
		   + get_index_with_len(i4,dimensions_[4])*offset_[4]
		   + get_index_with_len(i5,dimensions_[5])*offset_[5]
		   + get_index_with_len(i6,dimensions_[6])*offset_[6]);
    }
     template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5, typename I6>
    typename enable_if<Rank==7 && all_scalar_ints<7,I0,I1,I2,I3,I4,I5,I6>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6) const {
      return get_scalar_reference<IsActive>(
		     get_index_with_len(i0,dimensions_[0])*offset_[0]
		   + get_index_with_len(i1,dimensions_[1])*offset_[1]
		   + get_index_with_len(i2,dimensions_[2])*offset_[2]
		   + get_index_with_len(i3,dimensions_[3])*offset_[3]
		   + get_index_with_len(i4,dimensions_[4])*offset_[4]
		   + get_index_with_len(i5,dimensions_[5])*offset_[5]
		   + get_index_with_len(i6,dimensions_[6])*offset_[6]);
    }
   

    // The following define the case when operator() is called and one
    // of the arguments is a "range" object (an object that describes
    // a range of indices that are either contiguous or separated by a
    // fixed stride), while all others are of integer type (or a
    // rank-0 expression of integer type). An array object is returned
    // with a rank that may be reduced from that of the original
    // array, by one for each dimension that was indexed by an
    // integer. The new array points to a subset of the original data,
    // so modifying it will modify the original array.

    // First the case of a vector where we know the argument must be a
    // "range" object
    template <typename I0>
    typename enable_if<is_ranged<Rank,I0>::value,
		       Array<1,Type,IsActive> >::type
    operator()(I0 i0) {
      ExpressionSize<1> new_dim((i0.end(dimensions_[0])
				 + i0.stride(dimensions_[0])
				 -i0.begin(dimensions_[0]))
				/i0.stride(dimensions_[0]));
      ExpressionSize<1> new_offset(i0.stride(dimensions_[0])*offset_[0]);
#ifdef ADEPT_VERBOSE_FUNCTIONS
      std::cout << "  running Array::operator()(RANGED)\n";
#endif
      return Array<1,Type,IsActive>(data_ + i0.begin(dimensions_[0])*offset_[0],
	storage_, new_dim, new_offset);
    }
    template <typename I0>
    typename enable_if<is_ranged<Rank,I0>::value,
		       const Array<1,Type,IsActive> >::type
    operator()(I0 i0) const {
      ExpressionSize<1> new_dim((i0.end(dimensions_[0])
				 + i0.stride(dimensions_[0])
				 -i0.begin(dimensions_[0]))
				/i0.stride(dimensions_[0]));
      ExpressionSize<1> new_offset(i0.stride(dimensions_[0])*offset_[0]);
#ifdef ADEPT_VERBOSE_FUNCTIONS
      std::cout << "  running Array::operator()(RANGED) const\n";
#endif
      return Array<1,Type,IsActive>(data_ + i0.begin(dimensions_[0])*offset_[0],
				    storage_, new_dim, new_offset);
    }

  private:
    // For multi-dimensional arrays, we need a helper function

    // Treat the indexing of dimension "irank" in the case that the
    // index is of integer type
    template <typename T, int NewRank>
    typename enable_if<is_scalar_int<T>::value, void>::type
    update_index(const Index& irank, const T& i, Index& inew_rank, Index& ibegin,
		 ExpressionSize<NewRank>& new_dim, 
		 ExpressionSize<NewRank>& new_offset) const {
      ibegin += get_index_with_len(i,dimensions_[irank])*offset_[irank];
    }

    // Treat the indexing of dimension "irank" in the case that the
    // index is a "range" object
    template <typename T, int NewRank>
    typename enable_if<is_range<T>::value, void>::type
    update_index(const Index& irank, const T& i, Index& inew_rank, Index& ibegin,
		 ExpressionSize<NewRank>& new_dim, 
		 ExpressionSize<NewRank>& new_offset) const {
      ibegin += i.begin(dimensions_[irank])*offset_[irank];
      new_dim[inew_rank]
      = (i.end(dimensions_[irank])
	 + i.stride(dimensions_[irank])-i.begin(dimensions_[irank]))
      / i.stride(dimensions_[irank]);
      new_offset[inew_rank] = i.stride(dimensions_[irank])*offset_[irank];
      ++inew_rank;
    }

  public:

    // Now the individual overloads for each number of arguments, up
    // to 7, with separate r-value (const) and l-value (non-const)
    // versions
    template <typename I0, typename I1>
    typename enable_if<is_ranged<Rank,I0,I1>::value,
		       Array<is_ranged<Rank,I0,I1>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1) {
      static const int new_rank = is_ranged<Rank,I0,I1>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }

    template <typename I0, typename I1>
    typename enable_if<is_ranged<Rank,I0,I1>::value,
		       const Array<is_ranged<Rank,I0,I1>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1) const {
      static const int new_rank = is_ranged<Rank,I0,I1>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }

    template <typename I0, typename I1, typename I2>
    typename enable_if<is_ranged<Rank,I0,I1,I2>::value,
	       Array<is_ranged<Rank,I0,I1,I2>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2) {
      static const int new_rank = is_ranged<Rank,I0,I1,I2>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }

    template <typename I0, typename I1, typename I2>
    typename enable_if<is_ranged<Rank,I0,I1,I2>::value,
	       const Array<is_ranged<Rank,I0,I1,I2>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2) const {
      static const int new_rank = is_ranged<Rank,I0,I1,I2>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }

    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<is_ranged<Rank,I0,I1,I2,I3>::value,
       Array<is_ranged<Rank,I0,I1,I2,I3>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3) {
      static const int new_rank = is_ranged<Rank,I0,I1,I2,I3>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      update_index(3, i3, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }

    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<is_ranged<Rank,I0,I1,I2,I3>::value,
       const Array<is_ranged<Rank,I0,I1,I2,I3>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3) const {
      static const int new_rank = is_ranged<Rank,I0,I1,I2,I3>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      update_index(3, i3, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }

    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4>
    typename enable_if<is_ranged<Rank,I0,I1,I2,I3,I4>::value,
       Array<is_ranged<Rank,I0,I1,I2,I3,I4>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) {
      static const int new_rank = is_ranged<Rank,I0,I1,I2,I3,I4>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      update_index(3, i3, inew_rank, ibegin, new_dim, new_offset);
      update_index(4, i4, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }
  
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4>
    typename enable_if<is_ranged<Rank,I0,I1,I2,I3,I4>::value,
       const Array<is_ranged<Rank,I0,I1,I2,I3,I4>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const {
      static const int new_rank = is_ranged<Rank,I0,I1,I2,I3,I4>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      update_index(3, i3, inew_rank, ibegin, new_dim, new_offset);
      update_index(4, i4, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }
  
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5>
    typename enable_if<is_ranged<Rank,I0,I1,I2,I3,I4,I5>::value,
       Array<is_ranged<Rank,I0,I1,I2,I3,I4,I5>::count,Type,IsActive> >::type
     operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) {
      static const int new_rank = is_ranged<Rank,I0,I1,I2,I3,I4,I5>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      update_index(3, i3, inew_rank, ibegin, new_dim, new_offset);
      update_index(4, i4, inew_rank, ibegin, new_dim, new_offset);
      update_index(5, i5, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }


    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5>
    typename enable_if<is_ranged<Rank,I0,I1,I2,I3,I4,I5>::value,
       const Array<is_ranged<Rank,I0,I1,I2,I3,I4,I5>::count,Type,IsActive> >::type
     operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) const {
      static const int new_rank = is_ranged<Rank,I0,I1,I2,I3,I4,I5>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      update_index(3, i3, inew_rank, ibegin, new_dim, new_offset);
      update_index(4, i4, inew_rank, ibegin, new_dim, new_offset);
      update_index(5, i5, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }

    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5, typename I6>
    typename enable_if<is_ranged<Rank,I0,I1,I2,I3,I4,I5,I6>::value,
       Array<is_ranged<Rank,I0,I1,I2,I3,I4,I5,I6>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6) {
      static const int new_rank = is_ranged<Rank,I0,I1,I2,I3,I4,I5,I6>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      update_index(3, i3, inew_rank, ibegin, new_dim, new_offset);
      update_index(4, i4, inew_rank, ibegin, new_dim, new_offset);
      update_index(5, i5, inew_rank, ibegin, new_dim, new_offset);
      update_index(6, i6, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }

    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5, typename I6>
    typename enable_if<is_ranged<Rank,I0,I1,I2,I3,I4,I5,I6>::value,
       const Array<is_ranged<Rank,I0,I1,I2,I3,I4,I5,I6>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6) const {
      static const int new_rank = is_ranged<Rank,I0,I1,I2,I3,I4,I5,I6>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index(0, i0, inew_rank, ibegin, new_dim, new_offset);
      update_index(1, i1, inew_rank, ibegin, new_dim, new_offset);
      update_index(2, i2, inew_rank, ibegin, new_dim, new_offset);
      update_index(3, i3, inew_rank, ibegin, new_dim, new_offset);
      update_index(4, i4, inew_rank, ibegin, new_dim, new_offset);
      update_index(5, i5, inew_rank, ibegin, new_dim, new_offset);
      update_index(6, i6, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_ + ibegin, storage_,
					   new_dim, new_offset);
    }
  
    // If one or more of the indices is not guaranteed to be monotonic
    // at compile time then we must return an IndexedArray, now done
    // for all possible numbers of arguments

    // Indexing a 1D array
    template <typename I0>
    typename enable_if<Rank == 1 && is_int_vector<I0>::value
		       && !is_ranged<Rank,I0>::value,
		       IndexedArray<Rank,Type,IsActive,Array,I0> >::type
    operator()(const I0& i0) {
      return IndexedArray<Rank,Type,IsActive,Array,I0>(*this, i0);
    }
    template <typename I0>
    typename enable_if<Rank == 1 && is_int_vector<I0>::value
		       && !is_ranged<Rank,I0>::value,
		       const IndexedArray<Rank,Type,IsActive,
					  Array,I0> >::type
    operator()(const I0& i0) const {
      return IndexedArray<Rank,Type,IsActive,
			  Array,I0>(*const_cast<Array*>(this), i0);
    }
  
    // Indexing a 2D array
    template <typename I0, typename I1>
    typename enable_if<Rank == 2 && is_irreg_indexed<Rank,I0,I1>::value,
		       IndexedArray<is_irreg_indexed<Rank,I0,I1>::count,
				    Type,IsActive,Array,I0,I1> >::type
    operator()(const I0& i0, const I1& i1) {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,I0,I1>(*this, i0, i1);
    }
    template <typename I0, typename I1>
    typename enable_if<Rank == 2 && is_irreg_indexed<Rank,I0,I1>::value,
		       const IndexedArray<is_irreg_indexed<Rank,I0,I1>::count,
				    Type,IsActive,Array,I0,I1> >::type
    operator()(const I0& i0, const I1& i1) const {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1>::count;
      return IndexedArray<new_rank,Type,IsActive,
			  Array,I0,I1>(*const_cast<Array*>(this), i0, i1);
    }

    // Indexing a 3D array
    template <typename I0, typename I1, typename I2>
    typename enable_if<Rank == 3 && is_irreg_indexed<Rank,I0,I1,I2>::value,
		       IndexedArray<is_irreg_indexed<Rank,I0,I1,I2>::count,
				    Type,IsActive,Array,I0,I1,I2> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2) {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,
			  I0,I1,I2>(*this, i0, i1, i2);
    }
    template <typename I0, typename I1, typename I2>
    typename enable_if<Rank == 3 && is_irreg_indexed<Rank,I0,I1,I2>::value,
		       const IndexedArray<is_irreg_indexed<Rank,
							   I0,I1,I2>::count,
				    Type,IsActive,Array,I0,I1,I2> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2) const {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,
			  I0,I1,I2>(*const_cast<Array*>(this), i0, i1, i2);
    }

    // Indexing a 4D array
    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<Rank == 4 && is_irreg_indexed<Rank,I0,I1,I2,I3>::value,
		       IndexedArray<is_irreg_indexed<Rank,I0,I1,I2,I3>::count,
				    Type,IsActive,Array,I0,I1,I2,I3> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, const I3& i3) {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2,I3>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,
			  I0,I1,I2,I3>(*this, i0, i1, i2, i3);
    }
    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<Rank == 4 && is_irreg_indexed<Rank,I0,I1,I2,I3>::value,
		       const IndexedArray<is_irreg_indexed<Rank,I0,I1,
							   I2,I3>::count,
				    Type,IsActive,Array,I0,I1,I2,I3> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, const I3& i3) const {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2,I3>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,I0,I1,I2,
			  I3>(*const_cast<Array*>(this), i0, i1, i2, i3);
    }

    // Indexing a 5D array
    template <typename I0, typename I1, typename I2, typename I3, typename I4>
    typename enable_if<Rank == 5
		       && is_irreg_indexed<Rank,I0,I1,I2,I3,I4>::value,
		       IndexedArray<is_irreg_indexed<Rank,I0,I1,I2,
						     I3,I4>::count,
			    Type,IsActive,Array,I0,I1,I2,I3,I4> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, 
	       const I3& i3, const I4& i4) {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2,I3,
						   I4>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,I0,I1,I2,I3,
			  I4>(*this, i0, i1, i2, i3, i4);
    }
    template <typename I0, typename I1, typename I2, typename I3, typename I4>
    typename enable_if<Rank == 5
		       && is_irreg_indexed<Rank,I0,I1,I2,I3,I4>::value,
		       const IndexedArray<is_irreg_indexed<Rank,I0,I1,I2,
							   I3,I4>::count,
				  Type,IsActive,Array,I0,I1,I2,I3,I4> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, 
	       const I3& i3, const I4& i4) const {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2,I3,
						   I4>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,I0,I1,I2,I3,
			  I4>(*const_cast<Array*>(this), i0, i1, i2, i3, i4);
    }

    // Indexing a 6D array
    template <typename I0, typename I1, typename I2,
	      typename I3, typename I4, typename I5>
    typename enable_if<Rank == 6
		       && is_irreg_indexed<Rank,I0,I1,I2,I3,I4,I5>::value,
		       IndexedArray<is_irreg_indexed<Rank,I0,I1,I2,I3,
							   I4,I5>::count,
			  Type,IsActive,Array,I0,I1,I2,I3,I4,I5> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, 
	       const I3& i3, const I4& i4, const I5& i5) {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2,I3,
						   I4,I5>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,I0,I1,I2,I3,I4,
			  I5>(*this,i0,i1,i2,i3,i4,i5);
    }
    template <typename I0, typename I1, typename I2,
	      typename I3, typename I4, typename I5>
    typename enable_if<Rank == 6
		       && is_irreg_indexed<Rank,I0,I1,I2,I3,I4,I5>::value,
		       const IndexedArray<is_irreg_indexed<Rank,I0,I1,I2,I3,
							   I4,I5>::count,
			  Type,IsActive,Array,I0,I1,I2,I3,I4,I5> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, 
	       const I3& i3, const I4& i4, const I5& i5) const {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2,I3,
						   I4,I5>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,I0,I1,I2,I3,I4,
			  I5>(*const_cast<Array*>(this),i0,i1,i2,i3,i4,i5);
    }

    // Indexing a 7D array
    template <typename I0, typename I1, typename I2,
	      typename I3, typename I4, typename I5, typename I6>
    typename enable_if<Rank == 7
		       && is_irreg_indexed<Rank,I0,I1,I2,I3,I4,I5>::value,
		       IndexedArray<is_irreg_indexed<Rank,I0,I1,I2,I3,
						     I4,I5,I6>::count,
			  Type,IsActive,Array,I0,I1,I2,I3,I4,I5,I6> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, const I3& i3,
	       const I4& i4, const I5& i5, const I6& i6) {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2,I3,
						   I4,I5,I6>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,I0,I1,I2,I3,I4,I5,
			  I6>(*this,i0,i1,i2,i3,i4,i5,i6);
    }
    template <typename I0, typename I1, typename I2,
	      typename I3, typename I4, typename I5, typename I6>
    typename enable_if<Rank == 7
		       && is_irreg_indexed<Rank,I0,I1,I2,I3,I4,I5>::value,
		       const IndexedArray<is_irreg_indexed<Rank,I0,I1,I2,I3,
							   I4,I5,I6>::count,
			  Type,IsActive,Array,I0,I1,I2,I3,I4,I5,I6> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, const I3& i3,
	       const I4& i4, const I5& i5, const I6& i6) const {
      static const int new_rank = is_irreg_indexed<Rank,I0,I1,I2,I3,
						   I4,I5,I6>::count;
      return IndexedArray<new_rank,Type,IsActive,Array,I0,I1,I2,I3,I4,I5,
			  I6>(*const_cast<Array*>(this),i0,i1,i2,i3,i4,i5,i6);
    }


    // Provide a C-array-like array access: for a multidimensional
    // array, operator[](i), where i is of integer type, returns an
    // array of rank one less than the original array, where the new
    // array is "sliced" at index i of dimension 0.  For a vector,
    // operator[](i) returns an l-value to the element at i.  Thus for
    // a 3D array A, A[1][2][3] returns a single element. Note that
    // this will be slower than A(1,2,3) because each operator[]
    // creates a new array (although does not copy the data).
    template <typename T>
    typename enable_if<is_scalar_int<T>::value && (Rank > 1),
      Array<Rank-1,Type,IsActive> >::type
    operator[](T i) {
      int index = get_index_with_len(i,dimensions_[0])*offset_[0];
      ExpressionSize<Rank-1> new_dim;
      ExpressionSize<Rank-1> new_offset;
      for (int j = 1; j < Rank; ++j) {
	new_dim[j-1] = dimensions_[j];
	new_offset[j-1] = offset_[j];
      }
      return Array<Rank-1,Type,IsActive>(data_ + index,
					 storage_,
					 new_dim, new_offset);
    }


    // diag_matrix(), where *this is a 1D array, returns a DiagMatrix
    // containing the data as the diagonal pointing to the original
    // data, Can be used as an lvalue.
    SpecialMatrix<Type, internal::BandEngine<internal::ROW_MAJOR,0,0>, IsActive>
    diag_matrix();

    Array<1,Type,IsActive>
    diag_vector(Index offdiag = 0) {
      ADEPT_STATIC_ASSERT(Rank == 2, DIAG_VECTOR_ONLY_WORKS_ON_SQUARE_MATRICES);
      if (empty()) {
	// Return an empty vector
	return Array<1,Type,IsActive>();
      }
      else if (dimensions_[0] != dimensions_[1]) {
	throw invalid_operation("diag_vector member function only applicable to square matrices"
				ADEPT_EXCEPTION_LOCATION);
      }
      else if (offdiag >= 0) {
	Index new_dim = std::min(dimensions_[0], dimensions_[1]-offdiag);
	return Array<1,Type,IsActive>(data_+offset_[1]*offdiag, storage_, 
				      ExpressionSize<1>(new_dim),
				      ExpressionSize<1>(offset_[0]+offset_[1]));
      }
      else {
	Index new_dim = std::min(dimensions_[0]+offdiag, dimensions_[1]);
	return Array<1,Type,IsActive>(data_-offset_[0]*offdiag, storage_, 
				      ExpressionSize<1>(new_dim),
				      ExpressionSize<1>(offset_[0]+offset_[1]));
      }
    }
  
    Array
    submatrix_on_diagonal(Index ibegin, Index iend) {
      ADEPT_STATIC_ASSERT(Rank == 2,
		SUBMATRIX_ON_DIAGONAL_ONLY_WORKS_ON_SQUARE_MATRICES);
      if (dimensions_[0] != dimensions_[1]) {
	throw invalid_operation("submatrix_on_diagonal member function only applicable to square matrices"
				ADEPT_EXCEPTION_LOCATION);
      }
      else if (ibegin < 0 || ibegin > iend || iend >= dimensions_[0]) {
	throw index_out_of_bounds("Dimensions out of range in submatrix_on_diagonal"
				  ADEPT_EXCEPTION_LOCATION);
      }
      else {
	Index len = iend-ibegin+1;
	ExpressionSize<2> dim(len,len);
	return Array(data_+ibegin*(offset_[0]+offset_[1]),
		     storage_, dim, offset_);
      }
    }

    // For extracting contiguous sections out of an array use the
    // following. Currently this just indexes each dimension with the
    // contiguous range(a,b) index, but in future it may be optimized.

    // 1D array subset
    template <typename B0, typename E0>
    Array
    subset(const B0& ibegin0, const E0& iend0) {
      ADEPT_STATIC_ASSERT(Rank == 1,
			  SUBSET_WITH_2_ARGS_ONLY_ON_RANK_1_ARRAY);
      return (*this)(range(ibegin0,iend0));
    }
    template <typename B0, typename E0>
    const Array
    subset(const B0& ibegin0, const E0& iend0) const {
      ADEPT_STATIC_ASSERT(Rank == 1,
			  SUBSET_WITH_2_ARGS_ONLY_ON_RANK_1_ARRAY);
      return (*this)(range(ibegin0,iend0));
    }

    // 2D array subset
    template <typename B0, typename E0, typename B1, typename E1>
    Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1) {
      ADEPT_STATIC_ASSERT(Rank == 2,
			  SUBSET_WITH_4_ARGS_ONLY_ON_RANK_2_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1));
    }
    template <typename B0, typename E0, typename B1, typename E1>
    const Array
    subset(const B0& ibegin0, const E0& iend0, 
	  const B1& ibegin1, const E1& iend1) const {
      ADEPT_STATIC_ASSERT(Rank == 2,
			  SUBSET_WITH_4_ARGS_ONLY_ON_RANK_2_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1));
    }

    // 3D array subset
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2>
    Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2) {
      ADEPT_STATIC_ASSERT(Rank == 3,
			  SUBSET_WITH_6_ARGS_ONLY_ON_RANK_3_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2));
    }     
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2>
    const Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2) const {
      ADEPT_STATIC_ASSERT(Rank == 3,
			  SUBSET_WITH_6_ARGS_ONLY_ON_RANK_3_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2));
    }

    // 4D array subset
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3>
    Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3) {
      ADEPT_STATIC_ASSERT(Rank == 4,
			  SUBSET_WITH_8_ARGS_ONLY_ON_RANK_4_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3));
    }
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3>
    const Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3) const {
      ADEPT_STATIC_ASSERT(Rank == 4,
			  SUBSET_WITH_8_ARGS_ONLY_ON_RANK_4_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3));
    } 

    // 5D array subset
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4>
    Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4) {
      ADEPT_STATIC_ASSERT(Rank == 5,
			  SUBSET_WITH_10_ARGS_ONLY_ON_RANK_5_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4));
    }
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4>
    const Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4) const {
      ADEPT_STATIC_ASSERT(Rank == 5,
			  SUBSET_WITH_10_ARGS_ONLY_ON_RANK_5_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4));
    }

    // 6D array subset
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4, typename B5, typename E5>
    Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4,
	   const B5& ibegin5, const E5& iend5) {
      ADEPT_STATIC_ASSERT(Rank == 6,
			  SUBSET_WITH_12_ARGS_ONLY_ON_RANK_6_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4),range(ibegin5,iend5));
    }
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4, typename B5, typename E5>
    const Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4,
	   const B5& ibegin5, const E5& iend5) const {
      ADEPT_STATIC_ASSERT(Rank == 6,
			  SUBSET_WITH_12_ARGS_ONLY_ON_RANK_6_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4),range(ibegin5,iend5));
    }

    // 7D array subset
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4, typename B5, typename E5,
	      typename B6, typename E6>
    Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4,
	   const B5& ibegin5, const E5& iend5,
	   const B6& ibegin6, const E6& iend6) {
      ADEPT_STATIC_ASSERT(Rank == 7,
			  SUBSET_WITH_14_ARGS_ONLY_ON_RANK_7_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4),range(ibegin5,iend5),
		     range(ibegin6,iend6));
    }
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4, typename B5, typename E5,
	      typename B6, typename E6>
    const Array
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4,
	   const B5& ibegin5, const E5& iend5,
	   const B6& ibegin6, const E6& iend6) const {
      ADEPT_STATIC_ASSERT(Rank == 7,
			  SUBSET_WITH_14_ARGS_ONLY_ON_RANK_7_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4),range(ibegin5,iend5),
		     range(ibegin6,iend6));
    }

    // -------------------------------------------------------------------
    // Array: 5. Public member functions
    // -------------------------------------------------------------------
  
    // Link to an existing array of the same rank, type and activeness
    Array& link(Array& rhs) {
      if (!rhs.data()) {
	throw empty_array("Attempt to link to empty array"
			  ADEPT_EXCEPTION_LOCATION);
      }
      else {
	clear();
	data_ = rhs.data();
	storage_ = rhs.storage();
	dimensions_.copy(rhs.dimensions());
	offset_.copy(rhs.offset());
	if (storage_) {
	  storage_->add_link();
	}
	if (IsActive) {
	  GradientIndex<IsActive>::set(data_, storage_);
	}
      }
      return *this;
    }

    // Fortran-like link syntax A >>= B
    Array& operator>>=(Array& rhs)
    { return link(rhs); }  

#ifndef ADEPT_MOVE_SEMANTICS
    // A common pattern is to link to a subset of another Array,
    // e.g. vec1.link(vec2(range(2,4))), but the problem is that the
    // argument to link is a temporary so will not bind to Array&. In
    // C++98 we therefore need a function taking const Array& and then
    // cast away the const-ness. This has the unfortunate side effect
    // that a non-const Array can be linked to a const Array.
    Array&        link(const Array& rhs) { return link(const_cast<Array&>(rhs)); }
    Array& operator>>=(const Array& rhs) { return link(const_cast<Array&>(rhs)); }
#else
    // But in C++11 we can solve this problem and only bind to
    // temporary non-const Arrays
    Array&        link(Array&& rhs) { return link(const_cast<Array&>(rhs)); }
    Array& operator>>=(Array&& rhs) { return link(const_cast<Array&>(rhs)); }
#endif

    // To prevent linking to an rvalue expression we write a templated
    // function that will fail to compile
    template<class E>
    typename internal::enable_if<!E::is_lvalue,void>::type
    link(const Expression<Type,E>&) {
      ADEPT_STATIC_ASSERT(E::is_lvalue, CAN_ONLY_LINK_TO_AN_LVALUE_EXPRESSION);
    }
    template<class E>
    typename internal::enable_if<!E::is_lvalue,void>::type
    operator>>=(const Expression<Type,E>&) {
      ADEPT_STATIC_ASSERT(E::is_lvalue, CAN_ONLY_LINK_TO_AN_LVALUE_EXPRESSION);
    }

    // STL-like size() returns total length of array
    Index size() const {
      Index s = 1;
      for (int i = 0; i < Rank; ++i) {
	s *= dimensions_[i];
      }
      return s; 
    }

    // Return constant reference to dimensions
    const ExpressionSize<Rank>& dimensions() const {
      return dimensions_;
    }

    bool get_dimensions_(ExpressionSize<Rank>& dim) const {
      dim = dimensions_;
      return true;
    }

    // Return individual dimension - probably deprecate "dimension" in
    // favour of "size"
    Index dimension(int j) const {
      return dimensions_[j];
    }
    Index size(int j) const {
      return dimensions_[j];
    }

    // Return individual offset
    Index offset(int j) const {
      return offset_[j];
    }

    // Return constant reference to offsets
    const ExpressionSize<Rank>& offset() const {
      return offset_;
    }

    const Index& last_offset() const { return offset_[Rank-1]; }

    // Return true if the array is empty
    bool empty() const { return (dimensions_[0] == 0); }

    // Return a string describing the array
    std::string info_string() const {
      std::stringstream str;
      str << "Array<" << Rank << ">, dim=" << dimensions_ << ", offset=" << offset_ << ", data_location=" << data_;
      if (IsActive) {
	str << ", gradient_index=" << gradient_index();
      }
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

    // For vectors only, we allow a pointer to be returned to a
    // specified element
    Type* data_pointer(Index i) { 
      ADEPT_STATIC_ASSERT(Rank == 1, CAN_ONLY_USE_DATA_POINTER_WITH_INDEX_ON_VECTORS);
      if (data_) {
	return data_ + offset_[0]*i;
      }
      else {
	return 0;
      }
    }
    const Type* const_data_pointer(Index i) const { 
      ADEPT_STATIC_ASSERT(Rank == 1, CAN_ONLY_USE_CONST_DATA_POINTER_WITH_INDEX_ON_VECTORS);
      if (data_) {
	return data_ + offset_[0]*i;
      }
      else {
	return 0;
      }
    }
   
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
      dimensions_.set_all(0);
      offset_.set_all(0);
      GradientIndex<IsActive>::clear();
    }

    // Resize an array
    void
    resize(const Index* dim) {

      ADEPT_STATIC_ASSERT(!(std::numeric_limits<Type>::is_integer
	    && IsActive), CANNOT_CREATE_ACTIVE_ARRAY_OF_INTEGERS);

      if (storage_) {
	storage_->remove_link();
	storage_ = 0;
      }
      // Check requested dimensions
      for (int i = 0; i < Rank; ++i) {
	if (dim[i] < 0) {
	  throw invalid_dimension("Negative array dimension requested"
				  ADEPT_EXCEPTION_LOCATION);
	}
	else if (dim[i] == 0) {
	  // If any of the dimensions is zero, we clear the array
	  // completely and all dimensions will be zero
	  clear();
	  return;
	}
      }
      dimensions_.copy(dim); // Copy dimensions
      pack_();
      Index data_vol;
      if (array_row_major_order) {
	data_vol = offset_[0]*dimensions_[0];
      }
      else {
	data_vol = size();
      }
      storage_ = new Storage<Type>(data_vol, IsActive);
      data_ = storage_->data();
      GradientIndex<IsActive>::set(data_, storage_);
    }

    // Resize with an ExpressionSize object
    void resize(const ExpressionSize<Rank>& dim) {
      resize(&dim[0]);
    }

    // Resize specifying order
    void resize_row_major(const ExpressionSize<Rank>& dim) {
      resize(&dim[0]);
      pack_row_major_();
    }
    void resize_column_major(const ExpressionSize<Rank>& dim) {
      resize(&dim[0]);
      pack_column_major_();
    }

    void
    resize(Index m0, Index m1=-1, Index m2=-1, Index m3=-1,
	   Index m4=-1, Index m5=-1, Index m6=-1) {
      Index dim[7] = {m0, m1, m2, m3, m4, m5, m6};
      // Check invalid dimensions
      for (int i = 0; i < Rank; ++i) {
	if (dim[i] < 0) {
	  throw invalid_dimension("Invalid dimensions in array resize"
				  ADEPT_EXCEPTION_LOCATION);
	}
      }
      resize(dim);
    }

    void
    resize_row_major(Index m0, Index m1=-1, Index m2=-1, Index m3=-1,
	   Index m4=-1, Index m5=-1, Index m6=-1) {
      Index dim[7] = {m0, m1, m2, m3, m4, m5, m6};
      // Check invalid dimensions
      for (int i = 0; i < Rank; ++i) {
	if (dim[i] < 0) {
	  throw invalid_dimension("Invalid dimensions in array resize"
				  ADEPT_EXCEPTION_LOCATION);
	}
      }
      resize_row_major(dim);
    }

    void
    resize_column_major(Index m0, Index m1=-1, Index m2=-1, Index m3=-1,
	   Index m4=-1, Index m5=-1, Index m6=-1) {
      Index dim[7] = {m0, m1, m2, m3, m4, m5, m6};
      // Check invalid dimensions
      for (int i = 0; i < Rank; ++i) {
	if (dim[i] < 0) {
	  throw invalid_dimension("Invalid dimensions in array resize"
				  ADEPT_EXCEPTION_LOCATION);
	}
      }
      resize_column_major(dim);
    }

  protected:
    // Initialize with "MyRank" explicit dimensions, the function
    // only being defined if MyRank is equal to the actual Rank of
    // the Array
    template <int MyRank>
    typename enable_if<Rank == MyRank,void>::type
    resize_(Index m0, Index m1=-1, Index m2=-1, Index m3=-1,
	   Index m4=-1, Index m5=-1, Index m6=-1) {
      Index dim[7] = {m0, m1, m2, m3, m4, m5, m6};
      resize(dim);
    }

    // Vectorization of arrays of rank>1 is possible provided that the
    // fastest varying dimension has padding, if necessary, to ensure
    // alignment
    template <int ARank>
    typename enable_if<ARank==1 || ((ARank>1)&&!Packet<Type>::is_vectorized), bool>::type
    columns_aligned_() const {
      return true;
    }
    template <int ARank>
    typename enable_if<(ARank>1)&&Packet<Type>::is_vectorized,bool>::type
    columns_aligned_() const {
      return offset_[Rank-2] % Packet<Type>::size == 0;
    }

  public:
  
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
    bool all_arrays_contiguous_() const { return offset_[Rank-1] == 1 && columns_aligned_<Rank>(); }

    // Is the first data element aligned to a packet boundary?
    bool is_aligned_() const {
      return !(reinterpret_cast<std::size_t>(data_) & Packet<Type>::align_mask);
      // If we could union data with a uintptr_t object then we could
      // do the following, but there is no guarantee that uintptr_t
      // exists :-(
      //      return !(data_unsigned_int_ & Packet<Type>::align_mask);
    }

    // Return the number of unaligned elements before reaching the
    // first element on an alignment boundary, which is in units of
    // "n" Types.
    template <int n>
    int alignment_offset_() const {
      // This is rather slow!
      return (reinterpret_cast<std::size_t>(reinterpret_cast<void*>(data_))/sizeof(Type)) % n;
    }

    Type value_with_len_(const Index& j, const Index& len) const {
      ADEPT_STATIC_ASSERT(Rank == 1, CANNOT_USE_VALUE_WITH_LEN_ON_ARRAY_OF_RANK_OTHER_THAN_1);
      return data_[j*offset_[0]];
    }

    std::string expression_string_() const {
      if (true) {
	std::string a = array_helper<Rank,IsActive>().name();
	a += dimensions_.str();
	return a;
      }
      else {
	std::stringstream s;
	print(s);
	return s.str();
      }
    }

    // The same as operator=(inactive scalar) but does not put
    // anything on the stack
    template <typename RType>
    typename enable_if<is_not_expression<RType>::value, Array&>::type
    set_value(RType x) {
      if (!empty()) {
	assign_inactive_scalar_<Rank,false>(x);
      }
      return *this;
    }
  

    // Is the array contiguous in memory?
    bool is_contiguous() const {
      Index offset_expected = 1;
      for (int i = Rank-1; i >= 0; ++i) {
	if (offset_[i] != offset_expected) {
	  return false;
	}
	offset_expected *= dimensions_[i];
      }
      return true;
    }
    
    // Determine whether rows or columns are contiguous in memory and
    // increasing, needed for calling the BLAS matrix multipliciation
    // functions; the first can be used to check if the fastest
    // varying dimension is contiguous, to see if array indexes can be
    // incremented simply.
    bool is_row_contiguous() const {
      //      ADEPT_STATIC_ASSERT(Rank == 2, CANNOT_CHECK_ROW_CONTIGUOUS_IF_NOT_MATRIX);
      //      return offset_[1] == 1;
      if (Rank > 1) {
	return offset_[Rank-1] == 1 && offset_[Rank-2] >= dimensions_[Rank-1];
      }
      else {
	return offset_[Rank-1] == 1;
      }
    }
    bool is_column_contiguous() const {
      ADEPT_STATIC_ASSERT(Rank == 2, CANNOT_CHECK_COLUMN_CONTIGUOUS_IF_NOT_MATRIX);
      return offset_[0] == 1;
    }

  public:
    // Return the gradient index for the first element in the array,
    // or -1 if not active
    Index gradient_index() const {
      //      ADEPT_STATIC_ASSERT(IsActive, CANNOT_ACCESS_GRADIENT_INDEX_OF_INACTIVE_ARRAY);
      //      return my_gradient_index<IsActive>();
      return GradientIndex<IsActive>::get();
    }

    /*
    std::ostream& print(std::ostream& os) const {
      if (empty()) {
	os << "(empty " << Rank << "-D array)";
      }
      else if (adept::internal::array_print_curly_brackets) {
	adept::ExpressionSize<Rank> i(0);
	int my_rank = -1;
	if (Rank > 1) {
	  os << "\n";
	}
	do {
	  for (int r = 0; r < my_rank+1; r++)
	    { os << " "; }
	  for (int r = my_rank+1; r < Rank; r++)
	    { os << "{"; }
	  for (i[Rank-1] = 0; i[Rank-1] < dimensions_[Rank-1]-1; ++i[Rank-1])
	    { os << data_[index_(i)] << ", "; }
	  os << data_[index_(i)];
	  my_rank = Rank-1;
	  while (--my_rank >= 0) {
	    if (++i[my_rank] >= dimensions_[my_rank]) {
	      i[my_rank] = 0;
	      os << "}";
	    }
	    else {
	      os << "},\n";
	      break;
	    }
	  }
	} while (my_rank >= 0);
	if (Rank > 1) {
	  os << "}"; // "}/n"
	}
	else {
	  os << "}";
	}
      }
      else {
	adept::ExpressionSize<Rank> i(0);
	int my_rank;
	do {
	  for (i[Rank-1] = 0; i[Rank-1] < dimensions_[Rank-1]; ++i[Rank-1]) {
	    os << " " << data_[index_(i)];
	  }
	  my_rank = Rank-1;
	  while (--my_rank >= 0) {
	    if (++i[my_rank] >= dimensions_[my_rank]) {
	      i[my_rank] = 0;
	    }
	    else {
	      break;
	    }
	  }
	  os << "\n";
	} while (my_rank >= 0);
      }
      return os;
    }
    */

    std::ostream& print(std::ostream& os) const {
      if (empty()) {
	os << array_print_empty_before;
	if (array_print_empty_rank) {
	  os << Rank;
	}
	os << array_print_empty_after;
      }
      else if (Rank == 1) {
	// Print a vector
	os << vector_print_before << data_[0];
	for (int i = 1; i < dimensions_[0]; ++i) {
	  os << vector_separator << data_[i*offset_[0]];
	}
	os << vector_print_after;
      }
      else {
	// Print a multi-dimensional array
	adept::ExpressionSize<Rank> i(0);
	int my_rank = -1;
	os << array_print_before;
	do {
	  if (array_print_indent) {
	    if (my_rank >= 0) {
	      os << " ";
	      for (int r = 0; r < my_rank*static_cast<int>(array_opening_bracket.size()); r++) {
		os << " ";
	      }
	    }
	  }
	  if (my_rank == -1) {
	    for (int r = 1; r < Rank; r++) {
	      os << array_opening_bracket;
	    }
	  }
	  else {
	    for (int r = my_rank+1; r < Rank; r++) {
	      os << array_opening_bracket;
	    }
	  }
	  for (i[Rank-1] = 0; i[Rank-1] < dimensions_[Rank-1]-1; ++i[Rank-1]) {
	    os << data_[index_(i)] << array_contiguous_separator;
	  }
	  os << data_[index_(i)];
	  my_rank = Rank-1;
	  while (--my_rank >= 0) {
	    if (++i[my_rank] >= dimensions_[my_rank]) {
	      i[my_rank] = 0;
	      os << array_closing_bracket;
	    }
	    else {
	      os << array_closing_bracket << array_non_contiguous_separator;
	      break;
	    }
	  }
	} while (my_rank >= 0);
	os << array_print_after;
      }
      return os;
    }

    // Get pointers to the first and last data members in memory.  
    void data_range(Type const * &data_begin, Type const * &data_end) const {
      data_begin = data_;
      data_end = data_;
      for (int i = 0; i < Rank; i++) {
	if (offset_[i] >= 0) {
	  data_end += (dimensions_[i]-1)*offset_[i];
	}
	else {
	  data_begin += (dimensions_[i]-1)*offset_[i];
	}
      }
    }

  
    // The Stack::independent(x) and Stack::dependent(y) functions add
    // the gradient_index of objects x and y to std::vector<uIndex>
    // objects in Stack. Since x and y may be scalars or arrays, this
    // is best done by delegating to the Active or Array classes.
    template <typename IndexType>
    void push_gradient_indices(std::vector<IndexType>& vec) const {
      ADEPT_STATIC_ASSERT(IsActive,
		  CANNOT_PUSH_GRADIENT_INDICES_FOR_INACTIVE_ARRAY); 
      ExpressionSize<Rank> i(0);
      Index gradient_ind = gradient_index();
      Index index = 0;
      int my_rank;
      vec.reserve(vec.size() + size());
      do {
	// Innermost loop - note that the counter is index, not max_index
	for (Index max_index = index + dimensions_[Rank-1]*offset_[Rank-1];
	     index < max_index;
	     index += offset_[Rank-1]) {
	  vec.push_back(gradient_ind + index);
	}
	// Increment counters appropriately depending on which
	// dimensions have been finished
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }

    // Return inactive array linked to original data
    Array<Rank, Type, false> inactive_link() {
      return Array<Rank, Type, false>(data_, storage_, dimensions_, offset_);
    }

    // Perform an in-place transpose for 2D arrays only
    Array& in_place_transpose() {
      ADEPT_STATIC_ASSERT(Rank == 2, 
			  IN_PLACE_TRANSPOSE_ONLY_POSSIBLE_WITH_2D_ARRAYS);
      Index tmp;
      // Swap dimensions
      tmp = dimensions_[0];
      dimensions_[0] = dimensions_[1];
      dimensions_[1] = tmp;
      // Swap offsets
      tmp = offset_[0];
      offset_[0] = offset_[1];
      offset_[1] = tmp;
      return *this;
    }

    // Transpose helper functions
  protected:
    template<int MyRank>
    typename enable_if<MyRank == 2, Array<2,Type,IsActive> >::type
    my_T() {
      // Transpose 2D array: create output array initially as link
      // to input array
      Array<2,Type,IsActive> out(*this);
      // Swap dimensions
      return out.in_place_transpose();
    }
    template<int MyRank>
    typename enable_if<MyRank == 2, const Array<2,Type,IsActive> >::type
    my_T() const {
      // Transpose 2D array: create output array initially as link
      // to input array
      Array<2,Type,IsActive> out(const_cast<Array&>(*this));
      // Swap dimensions
      return out.in_place_transpose();
    }

  public:
    // Out-of-place transpose
    Array<2,Type,IsActive>
    T() {
      ADEPT_STATIC_ASSERT(Rank == 1 || Rank == 2, 
			  TRANSPOSE_ONLY_POSSIBLE_WITH_1D_OR_2D_ARRAYS);
      return my_T<Rank>();
    }
    const Array<2,Type,IsActive>
    T() const {
      ADEPT_STATIC_ASSERT(Rank == 1 || Rank == 2, 
			  TRANSPOSE_ONLY_POSSIBLE_WITH_1D_OR_2D_ARRAYS);
      return my_T<Rank>();
    }

    // "permute" is a generalized transpose, returning an Array linked
    // to the current one but with the dimensions rearranged according
    // to idim: idim[0] is the 0-based number of the dimension of the
    // current array that will be dimension 0 of the new array,
    // idim[1] is the number of the dimension of the current array
    // that will be dimension 1 of the new array and so on.
    Array permute(const Index* idim) {
      if (empty()) {
	throw empty_array("Attempt to permute an empty array"
			  ADEPT_EXCEPTION_LOCATION);
      }
      ExpressionSize<Rank> new_dims(0);
      ExpressionSize<Rank> new_offset;
      for (int i = 0; i < Rank; ++i) {
	if (idim[i] >= 0 && idim[i] < Rank) {
	  new_dims[i] = dimensions_[idim[i]];
	  new_offset[i] = offset_[idim[i]];
	}
	else {
	  throw invalid_dimension("Dimensions must be in range 0 to Rank-1 in permute"
				  ADEPT_EXCEPTION_LOCATION);
	}
      }
      for (int i = 0; i < Rank; ++i) {
	if (new_dims[i] == 0) {
	  throw invalid_dimension("Missing dimension in permute"
				  ADEPT_EXCEPTION_LOCATION);
	}
      }
      return Array(data_, storage_, new_dims, new_offset);
    }

    Array permute(const ExpressionSize<Rank>& idim) {
      return permute(&idim[0]);
    }

    // Up to 7 dimensions we can specify the dimensions as separate
    // arguments
    typename enable_if<(Rank < 7), Array>::type
    permute(Index i0, Index i1, Index i2 = -1, Index i3 = -1, Index i4 = -1,
	    Index i5 = -1, Index i6 = -1) {
      Index idim[7] = {i0, i1, i2, i3, i4, i5, i6};
      for (int i = 0; i < Rank; ++i) {
	if (idim[i] == -1) {
	  throw invalid_dimension("Incorrect number of dimensions provided to permute"
				  ADEPT_EXCEPTION_LOCATION);
	}
      }
      return permute(idim);
    }

    // Only applicable to vectors, return a multi-dimensional array
    // that links to the data in the vector
    template <int NewRank>
    Array<NewRank,Type,IsActive> reshape(const ExpressionSize<NewRank>& dims) {
      ADEPT_STATIC_ASSERT(Rank == 1, CANNOT_RESHAPE_MULTIDIMENSIONAL_ARRAY);
      Index new_size = 1;
      for (int i = 0; i < NewRank; ++i) {
	new_size *= dims[i];
      }
      if (new_size != dimensions_[0]) {
	throw invalid_dimension("Size of reshaped array does not match original vector");
      }
      ExpressionSize<NewRank> offset;
      offset[NewRank-1] = offset_[0];
      for (int i = NewRank-2; i >= 0; --i) {
	offset[i] = dims[i+1]*offset[i+1];
      }
      return Array<NewRank,Type,IsActive>(data_,storage_,dims,offset);
    }

    // More convenient interfaces to reshape providing a list of
    // integer dimensions
    Array<2,Type,IsActive> reshape(Index i0, Index i1)
    { return reshape(ExpressionSize<2>(i0,i1)); }
    Array<3,Type,IsActive> reshape(Index i0, Index i1, Index i2)
    { return reshape(ExpressionSize<2>(i0,i1,i2)); }
    Array<4,Type,IsActive> reshape(Index i0, Index i1, Index i2, Index i3)
    { return reshape(ExpressionSize<2>(i0,i1,i2,i3)); }
    Array<5,Type,IsActive> reshape(Index i0, Index i1, Index i2, Index i3, Index i4)
    { return reshape(ExpressionSize<2>(i0,i1,i2,i3,i4)); }
    Array<6,Type,IsActive> reshape(Index i0, Index i1, Index i2, Index i3,
				   Index i4, Index i5)
    { return reshape(ExpressionSize<2>(i0,i1,i2,i3,i4,i5)); }
    Array<7,Type,IsActive> reshape(Index i0, Index i1, Index i2, Index i3,
				   Index i4, Index i5, Index i6)
    { return reshape(ExpressionSize<2>(i0,i1,i2,i3,i4,i5,i6)); }


    // Return an Array that is a "soft" link to the data in the
    // present array; that is, it does not copy the Storage object and
    // increase the reference counter therein. This is useful in a
    // multi-threaded environment when multiple threads may wish to
    // subset the same array.
    Array soft_link() {
      return Array(data_,0,dimensions_,offset_,gradient_index());
    }
    const Array soft_link() const {
      return Array(data_,0,dimensions_,offset_,gradient_index());
    }


    // Place gradients associated with the present active array into
    // the equivalent passive array provided as an argument
    template <typename MyType>
    void get_gradient(Array<Rank,MyType,false>& gradient) const {
      ADEPT_STATIC_ASSERT(IsActive,CANNOT_USE_GET_GRADIENT_ON_INACTIVE_ARRAY);
      if (gradient.empty()) {
	gradient.resize(dimensions_);
      }
      else if (gradient.dimensions() != dimensions_) {
	throw size_mismatch("Attempt to get_gradient with array of different dimensions"
			    ADEPT_EXCEPTION_LOCATION);
      }
      static const int last = Rank-1;
      ExpressionSize<Rank> target_offset = gradient.offset();
      ExpressionSize<Rank> i(0);
      Index index = 0;
      int my_rank;
      Index index_target = 0;
      Index last_dim_stretch = dimensions_[last]*offset_[last];
      MyType* target = gradient.data();
      do {
	i[last] = 0;
	index_target = 0;
	for (int r = 0; r < Rank-1; r++) {
	  index_target += i[r]*target_offset[r];
	}
	ADEPT_ACTIVE_STACK->get_gradients(gradient_index()+index,
				  gradient_index()+index+last_dim_stretch,
				  target+index_target, offset_[last], target_offset[last]);
	index += last_dim_stretch;
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }

    // Return an inactive array of the same type and rank as the
    // present active array containing the gradients associated with
    // it
    Array<Rank,Type,false> get_gradient() const {
      Array<Rank,Type,false> gradient;
      get_gradient(gradient);
      return gradient;
    }

    // std::vector<typename internal::active_scalar<Type,IsActive>::type>
    // std_vector() const {
    //   ADEPT_STATIC_ASSERT(Rank == 1, STD_VECTOR_ONLY_AVAILABLE_FOR_RANK_1_ARRAYS);
    //   std::vector<typename internal::active_scalar<Type,IsActive>::type> data(dimensions_[0]);
    //   for (Index i = 0; i < dimensions_[0]; ++i) {
    // 	data[i] = (*this)(i);
    //   }
    //   return data;
    // }

    void
    put(std::vector<typename internal::active_scalar<Type,IsActive>::type>& data) const {
      ADEPT_STATIC_ASSERT(Rank == 1, PUT_ONLY_AVAILABLE_FOR_RANK_1_ARRAYS);
      if (data.size() != dimensions_[0]) {
	data.resize(dimensions_[0]);
      }
      for (Index i = 0; i < dimensions_[0]; ++i) {
	data[i] = (*this)(i);
      }  
    }

    void
    get(const std::vector<typename internal::active_scalar<Type,IsActive>::type>& data) {
      ADEPT_STATIC_ASSERT(Rank == 1, GET_ONLY_AVAILABLE_FOR_RANK_1_ARRAYS);
      if (data.size() != dimensions_[0]) {
	resize(data.size());
      }
      for (Index i = 0; i < dimensions_[0]; ++i) {
	(*this)(i) = data[i];
      }  
    }


    // -------------------------------------------------------------------
    // Array: 6. Member functions accessed by the Expression class
    // -------------------------------------------------------------------

    template <int MyArrayNum, int NArrays>
    void set_location_(const ExpressionSize<Rank>& i, 
		       ExpressionSize<NArrays>& index) const {
      index[MyArrayNum] = index_(i);
    }
    
    template <int MyArrayNum, int NArrays>
    Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
      return data_[loc[MyArrayNum]];
    }
    template <int MyArrayNum, int NArrays>
    Packet<Type> packet_at_location_(const ExpressionSize<NArrays>& loc) const {
      return Packet<Type>(data_+loc[MyArrayNum]);
    }

    Type& lvalue_at_location(const Index& loc) {
      return data_[loc];
    }

    // Return a scalar
    template <bool IsAligned, int MyArrayNum, typename PacketType,
	      int NArrays>
    typename enable_if<is_same<Type,PacketType>::value, Type>::type
    values_at_location_(const ExpressionSize<NArrays>& loc) const {
      return data_[loc[MyArrayNum]];
    }

    // Return a Paket from an aligned memory address
    template <bool IsAligned, int MyArrayNum, typename PacketType,
	      int NArrays>
    typename enable_if<IsAligned && is_same<Packet<Type>,PacketType>::value, PacketType>::type
    values_at_location_(const ExpressionSize<NArrays>& loc) const {
      return Packet<Type>(data_+loc[MyArrayNum]);
    }    

    // Return a Paket from an unaligned memory address
    template <bool IsAligned, int MyArrayNum, typename PacketType,
	      int NArrays>
    typename enable_if<!IsAligned && is_same<Packet<Type>,PacketType>::value, PacketType>::type
    values_at_location_(const ExpressionSize<NArrays>& loc) const {
      // integer dummy second argument indicates unaligned load
      return Packet<Type>(data_+loc[MyArrayNum], 0); 
    }    

    // Return a scalar
    template <bool UseStored, bool IsAligned, int MyArrayNum, int MyScratchNum,
	      typename PacketType, int NArrays, int NScratch>
    typename enable_if<is_same<Type,PacketType>::value, Type>::type
    values_at_location_store_(const ExpressionSize<NArrays>& loc,
			      ScratchVector<NScratch,PacketType>& scratch) const {
      return data_[loc[MyArrayNum]];
    }

    // Return a Paket from an aligned memory address
    template <bool UseStored, bool IsAligned, int MyArrayNum, int MyScratchNum,
	      typename PacketType, int NArrays, int NScratch>
    typename enable_if<IsAligned && is_same<Packet<Type>,PacketType>::value, PacketType>::type
    values_at_location_store_(const ExpressionSize<NArrays>& loc,
			      ScratchVector<NScratch,PacketType>& scratch) const {
      return Packet<Type>(data_+loc[MyArrayNum]);
    }
    // Return a Paket from an unaligned memory address
    template <bool UseStored, bool IsAligned, int MyArrayNum, int MyScratchNum,
	      typename PacketType, int NArrays, int NScratch>
    typename enable_if<!IsAligned && is_same<Packet<Type>,PacketType>::value, PacketType>::type
    values_at_location_store_(const ExpressionSize<NArrays>& loc,
			      ScratchVector<NScratch,PacketType>& scratch) const {
      return Packet<Type>(data_+loc[MyArrayNum], 0);
    }
   
    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
    Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				  ScratchVector<NScratch>& scratch) const {
      return data_[loc[MyArrayNum]];

    }

    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
    Type value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
      return data_[loc[MyArrayNum]];
    }

    template <int MyArrayNum, int NArrays>
    void advance_location_(ExpressionSize<NArrays>& loc) const {
      loc[MyArrayNum] += offset_[Rank-1];
    }

    // If an expression leads to calc_gradient being called on an
    // active object, we push the multiplier and the gradient index on
    // to the operation stack (or 1.0 if no multiplier is specified
    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
    void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			const ScratchVector<NScratch>& scratch) const {
      stack.push_rhs(1.0, gradient_index() + loc[MyArrayNum]);
    }
    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, typename MyType>
    void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			const ScratchVector<NScratch>& scratch,
			const MyType& multiplier) const {
      stack.push_rhs(multiplier, gradient_index() + loc[MyArrayNum]);
    }
  
    template <int MyArrayNum, int MyScratchNum, int MyActiveNum,
	      int NArrays, int NScratch, int NActive>
    void calc_gradient_packet_(Stack& stack, 
			       const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch,Packet<Real> >& scratch,
			       ScratchVector<NActive,Packet<Real> >& gradients) const {
      stack.push_rhs_indices<Packet<Real>::size,NActive>(gradient_index() + loc[MyArrayNum]);
      gradients[MyActiveNum] = Packet<Real>(1.0);
    }

    template <int MyArrayNum, int MyScratchNum, int MyActiveNum,
	      int NArrays, int NScratch, int NActive, typename MyType>
    void calc_gradient_packet_(Stack& stack, 
			       const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch,Packet<Real> >& scratch,
			       ScratchVector<NActive,Packet<Real> >& gradients,
			       const MyType& multiplier) const {
      stack.push_rhs_indices<Packet<Real>::size,NActive>(gradient_index() + loc[MyArrayNum]);
      gradients[MyActiveNum] = multiplier;
    }


    // -------------------------------------------------------------------
    // Array: 7. Protected member functions
    // -------------------------------------------------------------------
  protected:

    // Set the memory offsets from the array dimensions either
    // assuming C++-style row-major order, or Fortran-style
    // column-major order. The pack_() function spaces the data so
    // that all arrays are aligned to packet boundaries, to facilitate
    // vectorization.
    void pack_row_major_() {
      offset_[Rank-1] = 1;
      if (Rank > 1) {
	// Round up to nearest packet size so that all rows are aligned
	if (dimensions_[Rank-1] >= Packet<Type>::size*2) {
	  offset_[Rank-2] = ((dimensions_[Rank-1] + Packet<Type>::size - 1) / Packet<Type>::size) * Packet<Type>::size;
	}
	else {
	  offset_[Rank-2] = dimensions_[Rank-1];
	}
	for (int i = Rank-3; i >= 0; --i) {
	  offset_[i] = dimensions_[i+1]*offset_[i+1];
	}
      }
    }
    void pack_column_major_() {
      offset_[0] = 1;
      for (int i = 1; i < Rank; ++i) {
	offset_[i] = dimensions_[i-1]*offset_[i-1];
      }
    }
    void pack_() {
      if (array_row_major_order) {
	pack_row_major_();
      }
      else {
	pack_column_major_();
      }
    }

    // ...while the pack_contiguous_() function makes sure all data
    // are contiguous in memory
    void pack_row_major_contiguous_() {
      offset_[Rank-1] = 1;
      for (int i = Rank-2; i >= 0; --i) {
	offset_[i] = dimensions_[i+1]*offset_[i+1];
      }
    }

    void pack_contiguous_() {
      if (array_row_major_order) {
	pack_row_major_contiguous_();
      }
      else {
	pack_column_major_();
      }
    }

    // Return the memory index (relative to data_) for array element
    // indicated by j
    Index index_(Index j[Rank]) const {
      Index o = 0;
      for (int i = 0; i < Rank; i++) {
	o += j[i]*offset_[i];
      }
      return o;
    }
    Index index_(const ExpressionSize<Rank>& j) const {
      Index o = 0;
      for (int i = 0; i < Rank; i++) {
	o += j[i]*offset_[i];
      }
      return o;
    }

    // Used in traversing through an array
    void advance_index(Index& index, int& rank, ExpressionSize<Rank>& i) const {
      index -= offset_[Rank-1]*dimensions_[Rank-1];
      rank = Rank-1;
      while (--rank >= 0) {
	if (++i[rank] >= dimensions_[rank]) {
	  i[rank] = 0;
	  index -= offset_[rank]*(dimensions_[rank]-1);
	}
	else {
	  index += offset_[rank];
	  break;
	}
      }
    }

    // When assigning a scalar to a whole array, there may be
    // advantage in specialist behaviour depending on the rank of the
    // array. This is a generic one that copies the number but treats
    // the present array as passive.
    template <int LocalRank, bool LocalIsActive, typename X>
    typename enable_if<!LocalIsActive,void>::type
    assign_inactive_scalar_(X x) {
      ExpressionSize<LocalRank> i(0);
      Index index = 0;
      int my_rank;
      do {
	// Innermost loop - note that the counter is index, not max_index
	for (Index max_index = index + dimensions_[LocalRank-1]*offset_[LocalRank-1];
	     index < max_index;
	     index += offset_[LocalRank-1]) {
	  data_[index] = x;
	}
	// Increment counters appropriately depending on which
	// dimensions have been finished
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }

    // An active array being assigned the value of an inactive scalar
    template <int LocalRank, bool LocalIsActive, typename X>
    typename enable_if<LocalIsActive,void>::type
    assign_inactive_scalar_(X x) {
      // If not recording we call the inactive version instead
#ifdef ADEPT_RECORDING_PAUSABLE
      if (! ADEPT_ACTIVE_STACK->is_recording()) {
	assign_inactive_scalar_<LocalRank, false, X>(x);
	return;
      }
#endif

      ExpressionSize<LocalRank> i(0);
      Index gradient_ind = gradient_index();
      Index index = 0;
      int my_rank;
      do {
	// Innermost loop
	ADEPT_ACTIVE_STACK->push_lhs_range(gradient_ind+index, dimensions_[LocalRank-1],
					   offset_[LocalRank-1]);
	for (Index max_index = index + dimensions_[LocalRank-1]*offset_[LocalRank-1];
	     index < max_index; index += offset_[LocalRank-1]) {
	  data_[index] = x;
	}

	// Increment counters appropriately depending on which
	// dimensions have been finished
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }


    // When copying an expression to a whole array, there may be
    // advantage in specialist behaviour depending on the rank of the
    // array
    template<int LocalRank, bool LocalIsActive, bool EIsActive, class E>
    inline
    typename enable_if<!LocalIsActive && (!expr_cast<E>::is_vectorizable
					  || !is_same<typename E::type,Type>::value),void>::type
    assign_expression_(const E& rhs) {
      ADEPT_STATIC_ASSERT(!EIsActive, CANNOT_ASSIGN_ACTIVE_EXPRESSION_TO_INACTIVE_ARRAY);
      ExpressionSize<LocalRank> i(0);
      ExpressionSize<expr_cast<E>::n_arrays> ind(0);
      Index index = 0;
      int my_rank;
      static const int last = LocalRank-1;
      // FIX!!!
      if (false) { //rhs.all_arrays_contiguous()) {
	do {
	  i[last] = 0;
	  rhs.set_location(i, ind);
	  // Innermost loop
	  for ( ; i[last] < dimensions_[last]; ++i[last],
		  index += offset_[last]) {
	    // Note that this is faster as we know that all indices
	    // need to be incremented by 1
	    data_[index] = rhs.next_value_contiguous(ind);
	  }
	  advance_index(index, my_rank, i);
	} while (my_rank >= 0);
      }
      else {
	do {
	  i[last] = 0;
	  rhs.set_location(i, ind);
	  // Innermost loop
	  for ( ; i[last] < dimensions_[last]; ++i[last],
		  index += offset_[last]) {
	    data_[index] = rhs.next_value(ind);
	  }
	  advance_index(index, my_rank, i);
	} while (my_rank >= 0);
      }
    }

    // Vectorized version for Rank-1 arrays
    template<int LocalRank, bool LocalIsActive, bool EIsActive, class E>
    inline //__attribute__((always_inline))
    typename enable_if<!LocalIsActive && expr_cast<E>::is_vectorizable && LocalRank == 1
		       && is_same<typename E::type,Type>::value,void>::type
      // Removing the reference speeds things up because otherwise E
      // is dereferenced each loop
      //  assign_expression_(const E& __restrict rhs) {
      assign_expression_(const E rhs) {
      ADEPT_STATIC_ASSERT(!EIsActive, CANNOT_ASSIGN_ACTIVE_EXPRESSION_TO_INACTIVE_ARRAY);
      ExpressionSize<1> i(0);
      ExpressionSize<expr_cast<E>::n_arrays> ind(0);

      if (dimensions_[0] >= Packet<Type>::size*2
	  && offset_[0] == 1
	  && rhs.all_arrays_contiguous()
	  ) {
	// Contiguous source and destination data
	Index istartvec = 0;
	Index iendvec = 0;

	istartvec = rhs.alignment_offset();
	
	if (istartvec < 0 || istartvec != alignment_offset_<Packet<Type>::size>()) {
	  istartvec = iendvec = 0;
	}
	else  {
	  iendvec = (dimensions_[0]-istartvec);
	  iendvec -= (iendvec % Packet<Type>::size);
	  iendvec += istartvec;
	}
	i[0] = 0;
	rhs.set_location(i, ind);
	Type* const __restrict t = data_; // Avoids an unnecessary load for some reason
	// Innermost loop
	for (int index = 0; index < istartvec; ++index) {
	  // Scalar version
	  t[index] = rhs.next_value_contiguous(ind);
	}
	for (int index = istartvec ; index < iendvec;
	     index += Packet<Type>::size) {
	  // Vectorized version
	  //	    rhs.next_packet(ind).put(data_+index)
	  // FIX may need unaligned store
	  rhs.next_packet(ind).put(t+index);
	}
	for (int index = iendvec ; index < dimensions_[0]; ++index) {
	  // Scalar version
	  t[index] = rhs.next_value_contiguous(ind);
	}
      }
      else {
	// Non-contiguous source or destination data
	i[0] = 0;
	rhs.set_location(i, ind);
	Type* const __restrict t = data_; // Avoids an unnecessary load for some reason
	for (int index = 0; i[0] < dimensions_[0]; ++i[0],
	       index += offset_[0]) {
	  t[index] = rhs.next_value(ind);
	}
      }
    }

    // Vectorized version
    template<int LocalRank, bool LocalIsActive, bool EIsActive, class E>
    inline
    typename enable_if<!LocalIsActive && expr_cast<E>::is_vectorizable && (LocalRank > 1)
                       && is_same<typename E::type,Type>::value,void>::type
    // Removing the reference speeds things up because otherwise E
    // is dereferenced each loop
    //  assign_expression_(const E& rhs) 
      assign_expression_(const E rhs) {
      ADEPT_STATIC_ASSERT(!EIsActive, CANNOT_ASSIGN_ACTIVE_EXPRESSION_TO_INACTIVE_ARRAY);
      ExpressionSize<LocalRank> i(0);
      ExpressionSize<expr_cast<E>::n_arrays> ind(0);
      Index index = 0;
      int my_rank;
      static const int last = LocalRank-1;
      
      if (dimensions_[last] >= Packet<Type>::size*2
	  && all_arrays_contiguous_()
	  && rhs.all_arrays_contiguous()) {
	// Contiguous source and destination data
	int iendvec;
	int istartvec = rhs.alignment_offset();
	if (istartvec < 0 || istartvec != alignment_offset_<Packet<Type>::size>()) {
	  istartvec = iendvec = 0;
	}
	else {
	  iendvec = (dimensions_[last]-istartvec);
	  iendvec -= (iendvec % Packet<Type>::size);
	  iendvec += istartvec;
	}


	do {
	  i[last] = 0;
	  rhs.set_location(i, ind);
	  // Innermost loop
	  for ( ; i[last] < istartvec; ++i[last], ++index) {
	    // Scalar version
	    data_[index] = rhs.next_value_contiguous(ind);
	  }
	  Type* const __restrict t = data_; // Avoids an unnecessary load for some reason
	  for ( ; i[last] < iendvec; i[last] += Packet<Type>::size,
		  index += Packet<Type>::size) {
	    // Vectorized version
	    //	    rhs.next_packet(ind).put(data_+index);
	    // FIX may need unaligned store
	    rhs.next_packet(ind).put(t+index);
	  }
	  for ( ; i[last] < dimensions_[last]; ++i[last], ++index) {
	    // Scalar version
	    data_[index] = rhs.next_value_contiguous(ind);
	  }
	  advance_index(index, my_rank, i);
	} while (my_rank >= 0);
      }
      else {
	// Non-contiguous source or destination data
	do {
	  i[last] = 0;
	  rhs.set_location(i, ind);
	  // Innermost loop
	  for ( ; i[last] < dimensions_[last]; ++i[last],
		  index += offset_[last]) {
	    data_[index] = rhs.next_value(ind);
	  }
	  advance_index(index, my_rank, i);
	} while (my_rank >= 0);
      }
    }

    template<int LocalRank, bool LocalIsActive, bool EIsActive, class E>
    inline
    typename enable_if<LocalIsActive && EIsActive,void>::type
  //    assign_expression_(const E& rhs) {
    assign_expression_(const E rhs) {
      // If recording has been paused then call the inactive version
#ifdef ADEPT_RECORDING_PAUSABLE
      if (!ADEPT_ACTIVE_STACK->is_recording()) {
	assign_expression_<LocalRank,false,false>(rhs);
	return;
      }
#endif
      ExpressionSize<LocalRank> i(0);
      ExpressionSize<expr_cast<E>::n_arrays> ind(0);
      Index index = 0;
      int my_rank;
      static const int last = LocalRank-1;

      ADEPT_ACTIVE_STACK->check_space(expr_cast<E>::n_active * size());

      if (expr_cast<E>::is_vectorizable && rhs.all_arrays_contiguous()) {
	// Contiguous source and destination data
	Type* const __restrict t = data_; // Avoids an unnecessary load for some reason
	do {
	  i[last] = 0;
	  rhs.set_location(i, ind);
	  // Innermost loop
	  for ( ; i[last] < dimensions_[last]; ++i[last],
		  index += offset_[last]) {
	    t[index] = rhs.next_value_and_gradient_contiguous(*ADEPT_ACTIVE_STACK, ind);
	    ADEPT_ACTIVE_STACK->push_lhs(gradient_index()+index); // What if RHS not active?
	  }
	  advance_index(index, my_rank, i);
	} while (my_rank >= 0);
      }
      else {
	// Non-contiguous source or destination data
	Type* const __restrict t = data_; // Avoids an unnecessary load for some reason
	do {
	  i[last] = 0;
	  rhs.set_location(i, ind);
	  // Innermost loop
	  for ( ; i[last] < dimensions_[last]; ++i[last],
		  index += offset_[last]) {
	    t[index] = rhs.next_value_and_gradient(*ADEPT_ACTIVE_STACK, ind);
	    ADEPT_ACTIVE_STACK->push_lhs(gradient_index()+index); // What if RHS not active?
	  }
	  advance_index(index, my_rank, i);
	} while (my_rank >= 0);
      }
    }

    template<int LocalRank, bool LocalIsActive, bool EIsActive, class E>
    inline
    typename enable_if<LocalIsActive && !EIsActive,void>::type
    assign_expression_(const E& rhs) {
      // If recording has been paused then call the inactive version
#ifdef ADEPT_RECORDING_PAUSABLE
      if (!ADEPT_ACTIVE_STACK->is_recording()) {
	assign_expression_<LocalRank,false,false>(rhs);
	return;
      }
#endif
      ExpressionSize<LocalRank> i(0);
      ExpressionSize<expr_cast<E>::n_arrays> ind(0);
      Index index = 0;
      int my_rank;
      Index gradient_ind = gradient_index();
      static const int last = LocalRank-1;
      do {
	i[last] = 0;
	rhs.set_location(i, ind);
	// Innermost loop
	ADEPT_ACTIVE_STACK->push_lhs_range(gradient_ind+index, dimensions_[LocalRank-1],
					   offset_[LocalRank-1]);
	for ( ; i[last] < dimensions_[last]; ++i[last],
	       index += offset_[last]) {
	  data_[index] = rhs.next_value(ind);
	}
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }



    template<bool LocalIsActive, class B, typename C>
    typename enable_if<!LocalIsActive,void>::type
    assign_conditional_inactive_scalar_(const B& bool_expr, C rhs) {
      ExpressionSize<Rank> i(0);
      ExpressionSize<expr_cast<B>::n_arrays> bool_ind(0);
      Index index = 0;
      int my_rank;
      static const int last = Rank-1;

      do {
	i[last] = 0;
	bool_expr.set_location(i, bool_ind);
	// Innermost loop
	for ( ; i[last] < dimensions_[last]; ++i[last],
	       index += offset_[last]) {
	  if (bool_expr.next_value(bool_ind)) {
	    data_[index] = rhs;
	  }
	}
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }

    template<bool LocalIsActive, class B, typename C>
    typename enable_if<LocalIsActive,void>::type
    assign_conditional_inactive_scalar_(const B& bool_expr, C rhs) {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (! ADEPT_ACTIVE_STACK->is_recording()) {
	assign_conditional_inactive_scalar_<false, B, C>(bool_expr, rhs);
	return;
      }
#endif

      ExpressionSize<Rank> i(0);
      ExpressionSize<expr_cast<B>::n_arrays> bool_ind(0);
      Index index = 0;
      int my_rank;
      static const int last = Rank-1;

      do {
	i[last] = 0;
	bool_expr.set_location(i, bool_ind);
	// Innermost loop
	for ( ; i[last] < dimensions_[last]; ++i[last],
	       index += offset_[last]) {
	  if (bool_expr.next_value(bool_ind)) {
	    data_[index] = rhs;
	    ADEPT_ACTIVE_STACK->push_lhs(gradient_index()+index);
	  }
	}
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }

    template<bool LocalIsActive, class B, class C>
    typename enable_if<!LocalIsActive,void>::type
    assign_conditional_(const B& bool_expr, const C& rhs) {
      ExpressionSize<Rank> i(0);
      ExpressionSize<expr_cast<B>::n_arrays> bool_ind(0);
      ExpressionSize<expr_cast<C>::n_arrays> rhs_ind(0);
      Index index = 0;
      int my_rank;
      static const int last = Rank-1;
      bool is_gap = false;

      do {
	i[last] = 0;
	rhs.set_location(i, rhs_ind);
	bool_expr.set_location(i, bool_ind);
	// Innermost loop
	for ( ; i[last] < dimensions_[last]; ++i[last],
	       index += offset_[last]) {
	  if (bool_expr.next_value(bool_ind)) {
	    if (is_gap) {
	      rhs.set_location(i, rhs_ind);
	      is_gap = false;
	    }
	    data_[index] = rhs.next_value(rhs_ind);
	  }
	  else {
	    is_gap = true;
	  }
	}
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }


    template<bool LocalIsActive, class B, class C>
    typename enable_if<LocalIsActive,void>::type
    assign_conditional_(const B& bool_expr, const C& rhs) {
      // If recording has been paused then call the inactive version
#ifdef ADEPT_RECORDING_PAUSABLE
      if (!ADEPT_ACTIVE_STACK->is_recording()) {
	assign_conditional_<false>(bool_expr, rhs);
	return;
      }
#endif
      ExpressionSize<Rank> i(0);
      ExpressionSize<expr_cast<B>::n_arrays> bool_ind(0);
      ExpressionSize<expr_cast<C>::n_arrays> rhs_ind(0);
      Index index = 0;
      int my_rank;
      static const int last = Rank-1;
      bool is_gap = false;

      ADEPT_ACTIVE_STACK->check_space(expr_cast<C>::n_active * size());
      do {
	i[last] = 0;
	rhs.set_location(i, rhs_ind);
	bool_expr.set_location(i, bool_ind);
	// Innermost loop
	for ( ; i[last] < dimensions_[last]; ++i[last],
	       index += offset_[last]) {
	  if (bool_expr.next_value(bool_ind)) {
	    if (is_gap) {
	      rhs.set_location(i, rhs_ind);
	      is_gap = false;
	    }
	    data_[index] = rhs.next_value_and_gradient(*ADEPT_ACTIVE_STACK, rhs_ind);
	    ADEPT_ACTIVE_STACK->push_lhs(gradient_index()+index); // What if RHS not active?
	  }
	  else {
	    is_gap = true;
	  }
	}
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }


    // -------------------------------------------------------------------
    // Array: 8. Static variables
    // -------------------------------------------------------------------
  public:


    void print_style(ArrayPrintStyle ps);


    // -------------------------------------------------------------------
    // Array: 9. Data
    // -------------------------------------------------------------------
  protected:
    Type* __restrict data_;           // Pointer to values
    Storage<Type>* storage_;          // Pointer to Storage object
    ExpressionSize<Rank> dimensions_; // Size of each dimension
    ExpressionSize<Rank> offset_;     // Memory offset for each dimension

  }; // End of Array class


  // -------------------------------------------------------------------
  // Helper functions
  // -------------------------------------------------------------------

  // Set the default ordering of arrays: if "true" use C-style
  // row-major ordering, otherwise use Fortran-style column-major
  // ordering
  inline
  void set_array_row_major_order(bool o = true) {
    ::adept::internal::array_row_major_order = o;
  }

  // Set the print style
  void set_array_print_style(ArrayPrintStyle ps);

  inline ArrayPrintStyle get_array_print_style() {
    return internal::array_print_style;
  }

  // Change whether or not curly brackets are printed when arrays are
  // sent to a stream with the << operator
  inline
  void set_array_print_curly_brackets(bool o = true) {
    //::adept::internal::array_print_curly_brackets = o;
    if (o) {
      set_array_print_style(PRINT_STYLE_CURLY);
    }
    else {
      set_array_print_style(PRINT_STYLE_PLAIN);
    }
  }

  // Print array on a stream
  template <int Rank, typename Type, bool IsActive>
  inline
  std::ostream&
  operator<<(std::ostream& os, const Array<Rank,Type,IsActive>& A) {
    return A.print(os);
  }


  // Extract inactive part of array, working correctly depending on
  // whether argument is active or inactive
  template <int Rank, typename Type>
  inline
  Array<Rank, Type, false>&
  value(Array<Rank, Type, false>& expr) {
    return expr;
  }
  template <int Rank, typename Type>
  inline
  Array<Rank, Type, false>
  value(Array<Rank, Type, true>& expr) {
    return expr.inactive_link();
  }

  // Print an array expression on a stream
  template <typename Type, class E>
  inline
  typename enable_if<(E::rank > 0), std::ostream&>::type
  operator<<(std::ostream& os, const Expression<Type,E>& expr) {
    Array<E::rank,Type,false> A;
    A.assign_inactive(expr);
    return A.print(os);
  }

  // -------------------------------------------------------------------
  // Transpose function
  // -------------------------------------------------------------------

  // Transpose 2D array
  template<typename Type, bool IsActive>
  inline
  Array<2,Type,IsActive>
  transpose(Array<2,Type,IsActive>& in) {
    // Create output array initially as link to input array 
    Array<2,Type,IsActive> out(in);
    // Swap dimensions
    return out.in_place_transpose();
  }

  // Transpose 1D array, treating it as a length N column vector, so
  // returning a 1xN 2D array
  template<typename Type, bool IsActive>
  inline
  Array<2,Type,IsActive>
  transpose(Array<1,Type,IsActive>& in) {
    return Array<2,Type,IsActive>(in.data(), in.storage(),
				  ExpressionSize<2>(1,in.dimension(0)),
				  ExpressionSize<2>(in.dimension(0)*in.offset(0),in.offset(0)));
  }

  // Transpose a 2D expression
  template<typename Type, class E>
  inline
  typename enable_if<E::rank == 2, Array<2,Type,E::is_active> >::type
  transpose(const Expression<Type,E>& in) {
    // Create output array by evaluating input expression
    Array<2,Type,E::is_active> out(in);
    // Swap dimensions
    return out.in_place_transpose();
  }

  // Transpose a 1D expression
  template<typename Type, class E>
  inline
  typename enable_if<E::rank == 1, Array<2,Type,E::is_active> >::type
  transpose(const Expression<Type,E>& in) {
    Array<1,Type,E::is_active> out_1D(in);
    return Array<2,Type,E::is_active>(out_1D.data(), out_1D.storage(),
				      ExpressionSize<2>(1,out_1D.dimension(0)),
				      ExpressionSize<2>(out_1D.dimension(0)*out_1D.offset(0),out_1D.offset(0)));
  }

  // Extract the gradients from an active Array after the
  // Stack::forward or Stack::reverse functions have been called
  template<int Rank, typename Type, typename dType>
  inline
  void get_gradients(const Array<Rank,Type,true>& a, Array<Rank,dType,false>& data)
  {
    data = a.get_gradient();
  }

} // End namespace adept

#endif
