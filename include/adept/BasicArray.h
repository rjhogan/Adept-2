/* BasicArray.h -- base for all dense arrays

    Copyright (C) 2020- European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/


#ifndef AdeptBasicArray_H
#define AdeptBasicArray_H 1

#include <type_traits>
#include <initializer_list>
#include <array>
#include <memory>

#include <adept/Expression.h>
#include <adept/BasicArray_traits.h>
#include <adept/RangeIndex.h>
#include <adept/GradientIndex.h>
//#include <adept/allocator.h>

namespace adept {

  template <typename Type,
	    internal::rank_type Rank,
	    internal::options_type Options = 0,
	    class Allocator = std::allocator<Type> >
  class BasicArray
    : public Expression<Type, BasicArray<Type,Rank,Options,Allocator> >,
      protected internal::GradientIndex<internal::opt_is_active(Options)> {

  public:

    friend struct Expression<Type,BasicArray<Type,Rank,Options,Allocator> >;
    
    typedef Type                      value_type;
    typedef internal::size_type       size_type;
    typedef internal::size_type       total_size_type;
    typedef internal::difference_type difference_type;
    typedef Type*                     pointer;
    typedef const Type*               const_pointer;
    typedef Type&                     reference;
    typedef const Type&               const_reference;
    typedef internal::rank_type       rank_type;
    typedef internal::options_type    options_type;
    typedef Allocator                 allocator_type;

    static const bool is_ref            = (Options & internal::ARRAY_IS_REF);
    static const bool is_partially_contiguous = (Options & internal::ARRAY_IS_PARTIALLY_CONTIGUOUS);
    static const bool is_all_contiguous = (Options & internal::ARRAY_IS_ALL_CONTIGUOUS);
    static const bool is_column_major   = (Options & internal::ARRAY_IS_COLUMN_MAJOR);
    static const bool is_active         = (Options & internal::ARRAY_IS_ACTIVE);
    static const bool is_constant       = std::is_const_v<value_type>;
    static const rank_type rank         = Rank;

    // Static definitions to enable this type to be treated as an
    // Expression
    static const bool      is_lvalue  = true;
    static const rank_type n_active   = is_active * (1 + internal::is_complex<Type>::value);
    static const rank_type n_scratch  = 0;
    static const rank_type n_arrays   = 1;
    static const bool      is_vectorizable = Packet<Type>::is_vectorized;
    
  protected:
    // BasicArray has a flags variable treated as a bitmask
    //    static const internal::options_type FLAG_OWNS_DATA = (1 << 0);

  public:

    // -------------------------------------------------------------------
    // Section 1. Constructors
    // -------------------------------------------------------------------
    
    // Construct an empty array
    BasicArray() //: flags_(0)
    { static_assert(!is_active
		    || std::is_floating_point<value_type>::value,
		    "Active BasicArrays must be of floating-point type");
      static_assert(!is_ref, "BasicArray refs cannot be constructed empty");
    }

    // Construct an array with a specified size, either with each
    // dimension as a separate argument...
    template <typename... Dims>
    BasicArray(Dims... dims) {
      static_assert(rank == sizeof...(dims),
		    "Incorrect number of constructor arguments");
      static_assert(!is_ref, "BasicArray refs must be constructed from another array");
      resize(dims...);
    }

    // ...or using a std::array to hold the dimensions
    BasicArray(const std::array<size_type, rank>& dims) {
      static_assert(!is_ref, "BasicArray refs must be constructed from another array");
      resize(dims);
    }

    // Construct a ref array, likely a subset of another array, where
    // the offsets are provided explicitly
    BasicArray(pointer data, const std::array<size_type,rank>& dims,
	       const std::array<difference_type,rank>& offsets,
	       internal::gradient_index_type gradient_index = -1)
      : data_(data), dimensions_(dims), offset_(offsets) /*, flags_(0) */ {
      // static_assert(is_ref, "Only ref arrays can be constructed to reference external data");
      if constexpr (is_active) {
	internal::GradientIndex<is_active>::set(gradient_index);	
      }
    }

    // Construct a ref array, likely a subset of another array, where
    // the data are assumed to be contiguous
    BasicArray(pointer data, const std::array<size_type,rank>& dims,
	       internal::gradient_index_type gradient_index = -1)
      : data_(data), dimensions_(dims) /*, flags_(0) */ {
      // static_assert(is_ref, "Only ref arrays can be constructed to reference external data");
      if constexpr (is_active) {
	internal::GradientIndex<is_active>::set(gradient_index);	
      }
      compute_offsets_contiguous_();
    }

    // Construct a const-ref array, likely a subset of another array, where
    // the offsets are provided explicitly
    BasicArray(const_pointer data, const std::array<size_type,rank>& dims,
	       const std::array<difference_type,rank>& offsets,
	       internal::gradient_index_type gradient_index = -1)
      : data_(data), dimensions_(dims), offset_(offsets) /*, flags_(0) */ {
      // static_assert(is_ref, "Only ref arrays can be constructed to reference external data");
      static_assert(is_constant, "A non-const ref array cannot reference constant external data");
      if constexpr (is_active) {
	internal::GradientIndex<is_active>::set(gradient_index);	
      }

    }

    // Construct a ref array, likely a subset of another array, where
    // the data are assumed to be contiguous
    BasicArray(const_pointer data, const std::array<size_type,rank>& dims,
	       internal::gradient_index_type gradient_index = -1)
      : data_(data), dimensions_(dims) /*, flags_(0) */ {
      // static_assert(is_ref, "Only ref arrays can be constructed to reference external data");
      static_assert(is_constant, "A non-const ref array cannot reference constant external data");
      if constexpr (is_active) {
	internal::GradientIndex<is_active>::set(gradient_index);	
      }
      compute_offsets_contiguous_();
    }

    // Copy constructor for non-const argument
    BasicArray(BasicArray& rhs) {
      if constexpr (!is_ref) {
        // Deep copy
        *this = rhs;
      }
      else {
	// Shallow copy
	data_      = rhs.data();
	dimensions_ = rhs.dimensions();
	offset_    = rhs.offsets();
	if constexpr (is_active) {
	  internal::GradientIndex<is_active>::set(rhs.gradient_index());
	}
      }
    }
    
    // Copy constructor for const argument
    BasicArray(const BasicArray& rhs) {
      if constexpr (!is_ref) {
        // Deep copy
        *this = rhs;
      }
      else {
	// Shallow copy
	static_assert(is_constant, "A non-const array ref cannot reference a const array");
	data_      = rhs.data();
	dimensions_ = rhs.dimensions();
	offset_    = rhs.offsets();
	if constexpr (is_active) {
	  internal::GradientIndex<is_active>::set(rhs.gradient_index());
	}
      }
    }

    // Pseudo copy constructor for BasicArray objects with different
    // template arguments
    template <typename T, internal::options_type Op, class Alloc>
    BasicArray(BasicArray<T,rank,Op,Alloc>& rhs) {
      if constexpr (!is_ref) {
        // Deep copy
        *this = rhs;
      }
      else {
	// Shallow copy
	static_assert(std::is_same_v<const value_type, const T>,
		      "Ref array cannot be constructed from an array of a different type");
	// Will fail (correctly) if rhs.data is const and data_ is not
	data_      = rhs.data();
	dimensions_ = rhs.dimensions();
	offset_    = rhs.offsets();
	if constexpr (is_partially_contiguous) {
          if constexpr (!is_column_major) {
	    if (offset_[rank-1] != 1) {
	      throw std::out_of_range("Row-contiguous array cannot reference a non-row-contiguous array");
	    }
	  }
	  else {
	    if (offset_[0] != 1) {
	      throw std::out_of_range("Column-contiguous array cannot reference a non-column-contiguous array");
	    }
	  }
	}
	// Ought to check for is_all_contiguous too...
	if constexpr (is_active) {
	  internal::GradientIndex<is_active>::set(rhs.gradient_index());
	}
      }
    }

    // Pseudo copy constructor for BasicArray objects with different
    // template arguments
    template <typename T, internal::options_type Op, class Alloc>
    BasicArray(const BasicArray<T,rank,Op,Alloc>& rhs) {
      if constexpr (!is_ref) {
        // Deep copy
        *this = rhs;
      }
      else {
	// Shallow copy
	static_assert(std::is_same_v<const value_type, const T>,
		      "Ref array cannot be constructed from an array of a different type");
	static_assert(is_constant, "Non-const ref array cannot reference a const-ref array");
	data_      = rhs.data();
	dimensions_ = rhs.dimensions();
	offset_    = rhs.offsets();
	if constexpr (is_partially_contiguous) {
          if constexpr (!is_column_major) {
	    if (offset_[rank-1] != 1) {
	      throw std::out_of_range("Row-contiguous array cannot reference a non-row-contiguous array");
	    }
	  }
	  else {
	    if (offset_[0] != 1) {
	      throw std::out_of_range("Column-contiguous array cannot reference a non-column-contiguous array");
	    }
	  }
	}
	// Ought to check for is_all_contiguous too...
	if constexpr (is_active) {
	  internal::GradientIndex<is_active>::set(rhs.gradient_index());
	}
	// Ought to check for is_all_contiguous too...
      }
    }

    // Construct from an expression: call the relevant assignment operator
    template <typename T, class E>
    BasicArray(const Expression<T,E>& rhs) {
      static_assert(E::rank == rank && (rank > 0), "Rank mismatch in BasicArray=Expression");
      static_assert(is_constant | !is_ref, "Non-const ref arrays cannot be constructed from Expressionl");
      *this = rhs;
    }
    
    // Construct from initializer_list...

    
    
    
    // -------------------------------------------------------------------
    // Section 2. Destructor
    // -------------------------------------------------------------------
    
    ~BasicArray() {
      std::cerr << "Destrucing array...\n";
      clear();
    }

    // -------------------------------------------------------------------
    // Section 4. Assignment operators
    // -------------------------------------------------------------------

    // Assignment to another array: copy the data by calling the
    // operator=(const Expression&) function
    BasicArray& operator=(const BasicArray& rhs) {
      return (*this = static_cast<const Expression<Type,BasicArray>&>(rhs));
    }

    // Assignment to an array that is about to be destructed
    // FIX: also need && assignment for other types of BasicArray
    BasicArray& operator=(BasicArray&& rhs) {
      if constexpr (is_ref) {
        // Array refs do a deep copy
        return (*this = static_cast<const Expression<Type,BasicArray>&>(rhs));
      }
      else {
	if (rhs.is_owner()) {
	  // Steal ownership of the data via a swap
	  swap(*this, rhs);
	}
	else {
	  // RHS is not owned so we must do a deep copy as there is no
	  // guarantee the data will stay around
	  return (*this = static_cast<const Expression<Type,BasicArray>&>(rhs));
	}
      }
    }
    
    friend void swap(BasicArray& l, BasicArray& r) noexcept {
      Type* tmp_data = l.data_;
      l.data_ = r.data_;
      r.data_ = tmp_data;
      swap(l.dimensions_, r.dimensions_);
      swap(l.offset_, r.offset_);
      std::swap(l.allocated_size_, r.allocated_size_);
      static_cast<internal::GradientIndex<is_active>&>(l).swap_value(static_cast<internal::GradientIndex<is_active>&>(r));
    }

    /*
    // Assignment to an array expression of the same rank
    template <typename EType, class E>
    inline //__attribute__((always_inline))
    typename internal::enable_if<E::rank == Rank, BasicArray&>::type
    operator=(const Expression<EType,E>&  __restrict rhs) {
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
      else if (!internal::compatible(dims, dimensions_)) {
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
    */

    
    // -------------------------------------------------------------------
    // Section 3. Resize and clear
    // -------------------------------------------------------------------

    // Deallocated contents and set size to zero
    void clear() {
      if (is_owner()) {
	std::cerr << "deallocating..." << allocated_size_ << "\n";
	std::allocator_traits<allocator_type>::deallocate(allocator_,
							  data_,
							  allocated_size_);
      }
      data_ = 0;
      allocated_size_ = 0;
      dimensions_ = {0};
      offset_ = {0};
      //is_owner(false);
    }

    // Resize specifying new dimensions as separate arguments
    template <typename... Dims>
    void resize(Dims... dims) {
      static_assert(rank == sizeof...(dims), "Array resized with wrong number of arguments");
      resize(std::array<size_type,rank>{static_cast<size_type>(dims)...});
    }

    // Resize specifying new dimensions in a std::array
    void resize(const std::array<size_type,rank>& dim) {
      static_assert(!is_ref, "Array refs cannot be resized");
      // For more efficiency we could reuse allocated space if total
      // size required is not increased
      clear();
      dimensions_ = dim;
      allocated_size_ = compute_offsets_();
      data_ = std::allocator_traits<allocator_type>::allocate(allocator_,
							      allocated_size_);
      //      is_owner(true);
    }

    // -------------------------------------------------------------------
    // Section x. Indexing functions
    // -------------------------------------------------------------------
    
    template <typename... Indices>
    typename std::enable_if_t<internal::all_scalar_indices<Indices...>::value,Type>
    operator()(Indices... indices) const {
      static_assert(rank == sizeof...(Indices),
		    "Incorrect number of arguments when subsetting array");
      return data_[index_with_len_<0>(indices...)];
    }
      
    template <typename... Indices>
    typename std::enable_if_t<internal::all_scalar_indices<Indices...>::value,Type&>
    operator()(Indices... indices) {
      static_assert(rank == sizeof...(Indices),
		    "Incorrect number of arguments when subsetting array");
      return data_[index_with_len_<0>(indices...)];
    }
    /*
    template <typename... Indices>
    typename std::enable_if_t<internal::ranged_indices<Indices...>::value,Type>
    operator()(Indices... indices) {
      static_assert(rank == sizeof...(Indices),
		    "Incorrect number of arguments when subsetting array");
      slice_<0>(indices...);
      return 0;
    }
    */

    // -------------------------------------------------------------------
    // Section 4. Inquiry functions
    // -------------------------------------------------------------------

    // Return the total size of the array
    total_size_type size() const {
      total_size_type s = 1;
      for (rank_type i = 0; i < rank; ++i) {
	s *= dimensions_[i];
      }
      return s;
    }

    // Return the size of a particular dimension
    size_type size(rank_type r) const {
      return dimensions_[r];
    }

    // Return true if the array is empty
    bool empty() const { return (dimensions_[0] == 0); }

    // Return a string describing the array
    std::string info_string() const {
      std::stringstream str;
      str << "Array<" << rank << ">, dim=" << dimensions_ << ", offset=" << offset_
          << ", data_location=" << data_ << ", allocated_size=" << allocated_size_;
      if constexpr (is_active) {
	str << ", gradient_index=" << gradient_index();
      }
      return str.str();
    }


    // Return the dimensions of an array as a std::array
    const std::array<size_type, rank>& dimensions() const {
      return dimensions_;
    }

    const std::array<difference_type, rank>& offsets() const {
      return offset_;
    }

    // Return pointer to the raw data
    pointer data()             { return data_; }
    const_pointer data() const { return data_; }
    
    // Is this array responsible for the destruction of its contents?
    bool is_owner() const {
      //return (data_ && (flags_ & FLAG_OWNS_DATA));
      return (data_ && allocated_size_ > 0);
    }
    /*
    void is_owner(bool is_own) {
      if (is_own) {
	flags_ |= FLAG_OWNS_DATA;
      }
      else {
	flags_ &= ~FLAG_OWNS_DATA;
      }
    }
    */
    void set_ownership(total_size_type size) {
      allocated_size_ = size;
    }

    // Return the gradient index for the first element in the array,
    // or -1 if not active
    Index gradient_index() const {
      return internal::GradientIndex<is_active>::get();
    }

    // -------------------------------------------------------------------
    // Section 5. Functions enabling participation in expressions
    // -------------------------------------------------------------------
    
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


    std::ostream& print(std::ostream& os) const {
      using namespace internal;
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
	adept::ExpressionSize<Rank> i{0};
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
      for (rank_type i = 0; i < rank; i++) {
	if (offset_[i] >= 0) {
	  data_end += (dimensions_[i]-1)*offset_[i];
	}
	else {
	  data_begin += (dimensions_[i]-1)*offset_[i];
	}
      }
    }


    std::string expression_string_() const {
      if (true) {
	std::string a = internal::array_helper<rank,is_active>().name();
	//	a += internal::str(dimensions_);
	a += "<ARRAY>";
	return a;
      }
      else {
	std::stringstream s;
	print(s);
	return s.str();
      }
    }

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

    template <int MyArrayNum, int NArrays>
    void advance_location_(ExpressionSize<NArrays>& loc) const {
      loc[MyArrayNum] += offset_[Rank-1];
    }

    
  protected:

    // Compute the memory offset of an element of an array given a
    // list of indices, where some indices might be expressions
    // containing "end" - hence the dimension of each array needs to
    // be passed so that "end" is resolved correctly    
    template <rank_type Dim, typename T0, typename... T>
    total_size_type index_with_len_(T0 i0, T... ii) {
      if constexpr (is_partially_contiguous && !is_column_major && Dim == 0) {
        // First dimension guaranteed to be contiguous
        return internal::get_index_with_len(i0,dimensions_[Dim])
	  + index_with_len_<Dim+1>(ii...);
      }
      else {
        return internal::get_index_with_len(i0,dimensions_[Dim])*offset_[Dim]
	  + index_with_len_<Dim+1>(ii...);
      }
    }
    // Final dimension
    template <rank_type Dim, typename T0>
    total_size_type index_with_len_(T0 i0) {
      if constexpr (is_partially_contiguous && ((is_column_major && Dim == 0)
                                             ||(!is_column_major && Dim == rank-1))) {
        // First or last dimension guaranteed to be contiguous
        return internal::get_index_with_len(i0,dimensions_[Dim]);
      }
      else {
        return internal::get_index_with_len(i0,dimensions_[Dim])*offset_[Dim];
      }
    }    

    template <typename T>
    total_size_type index_(T& j) const {
      total_size_type o = 0;
      for (rank_type i = 0; i < rank; i++) {
	o += j[i]*offset_[i];
      }
      return o;
    }

    // Used in traversing through an array
    template <typename T>
    void advance_index(total_size_type& index, rank_type& irank, T& i) const {
      index -= offset_[rank-1]*dimensions_[rank-1];
      irank = rank-1;
      while (--irank >= 0) {
	if (++i[irank] >= dimensions_[irank]) {
	  i[irank] = 0;
	  index -= offset_[irank]*(dimensions_[irank]-1);
	}
	else {
	  index += offset_[irank];
	  break;
	}
      }
    }
      
    // After resizing, compute the offsets given the dimensions, using
    // the current packing policy, and return the total number of
    // elements that must be allocated
    total_size_type compute_offsets_contiguous_() {
      offset_[rank-1] = 1;
      for (rank_type i = rank-2; i >= 0; --i) {
	offset_[i] = dimensions_[i+1]*offset_[i+1];
      }
      return offset_[0]*dimensions_[0];
    }
    total_size_type compute_offsets_aligned_() {
      // Space the data so that all arrays are aligned to packet
      // boundaries, to facilitate vectorization.
      offset_[rank-1] = 1;
      if (rank > 1) {
	// Round up to nearest packet size so that all rows are
	// aligned
	if (dimensions_[rank-1] >= adept::internal::Packet<Type>::size*2) {
	  offset_[rank-2]
	    = ((dimensions_[rank-1] + adept::internal::Packet<Type>::size - 1)
	       / adept::internal::Packet<Type>::size)
	    * adept::internal::Packet<Type>::size;
	}
	else {
	  offset_[rank-2] = dimensions_[rank-1];
	}
	for (int i = rank-3; i >= 0; --i) {
	  offset_[i] = dimensions_[i+1]*offset_[i+1];
	}
      }
      return offset_[0]*dimensions_[0];
    }

    template <typename T>
    bool get_dimensions_(T& dim) const {
      dim = dimensions_;
      return true;
    }

      
    total_size_type compute_offsets_() {
      if constexpr (is_all_contiguous) {
        return compute_offsets_contiguous_();
      }
      else {
	return compute_offsets_aligned_();
      }
    }
    
    template <internal::rank_type IDim, typename T, typename... Indices>
    void slice_(const T& mi, Indices... indices) {
      std::cout << "  index " << IDim << ": ";
      interpret_index_(mi);
      slice_<IDim+1>(indices...);
    }
    template <internal::rank_type IDim, typename T>
    void slice_(const T& mi) {
      interpret_index_(mi);
      //std::cout << "  index " << IDim << ": " << mi << "\n";
    }
    
    template <typename T>
    typename std::enable_if_t<internal::is_scalar_int<T>::value,void>
    interpret_index_(const T& mi) {
      std::cout << "scalar\n";
    }
    template <typename T>
    typename std::enable_if_t<internal::is_range<T>::value,void>
    interpret_index_(const T& mi) {
      std::cout << "range\n";
    }

    /*
    template <internal::rank_type IDim, typename... Indices>
    void slice_(std::initializer_list<size_type> mi,
		Indices... indices) {
      std::cout << "  index " << IDim << ": list of length " << mi.size() << "\n";
      slice_<IDim+1>(indices...);
    }
    template <internal::rank_type IDim>
    void slice_(std::initializer_list<size_type> mi) {
      std::cout << "  index " << IDim << ": list of length" << mi.size() << "\n";
    }
    */
    // DATA
  protected:
    pointer __restrict               data_ = 0;
    total_size_type                  allocated_size_ = 0;
    allocator_type                   allocator_ = allocator_type();
    std::array<size_type,rank>       dimensions_ = {0};
    std::array<difference_type,rank> offset_ = {0};
    //    options_type                     flags_ = (is_ref ? 0 : FLAG_OWNS_DATA);
  };

  
};


#endif
