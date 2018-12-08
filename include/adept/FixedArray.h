/* FixedArray.h -- active or inactive FixedArray of arbitrary rank

    Copyright (C) 2014-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   The FixedArray class has functionality modelled on Fortran-90 arrays -
   they can have a rank up to 7 (above will work, but some forms of
   indexing these arrays will not work).

*/

#ifndef AdeptFixedArray_H
#define AdeptFixedArray_H 1

#include <iostream>
#include <sstream>
#include <limits>

#include <adept/Array.h>
#include <adept/Allocator.h>

namespace adept {
  using namespace adept::internal;

  namespace internal {

    // -------------------------------------------------------------------
    // Helper classes
    // -------------------------------------------------------------------

    // The following are used by expression_string()
    template <int Rank, bool IsActive>
    struct fixed_array_helper            { const char* name() { return "FixedArray";  } };
    template <int Rank>
    struct fixed_array_helper<Rank,true> { const char* name() { return "aFixedArray";  } };

    template <>
    struct fixed_array_helper<1,false>   { const char* name() { return "FixedVector"; } };
    template <>
    struct fixed_array_helper<1,true>    { const char* name() { return "aFixedVector"; } };

    template <>
    struct fixed_array_helper<2,false>   { const char* name() { return "FixedMatrix"; } };
    template <>
    struct fixed_array_helper<2,true>    { const char* name() { return "aFixedMatrix"; } };

    template<Index J0, Index J1, Index J2, Index J3,
	     Index J4, Index J5, Index J6>
    struct fixed_array {
      static const int rank = (J0>0)
	* (1 + (J1>0) * (1 + (J2>0) * (1 + (J3>0) * (1 + (J4>0) * (1 + (J5>0) * (1 + (J6>0)))))));
      static const Index length = (J0 + (J0<1)) * (J1 + (J1<1)) * (J2 + (J2<1))
	* (J3 + (J3<1)) * (J4 + (J4<1)) * (J5 + (J5<1)) * (J6 + (J6<1));
    };

  } // End namespace internal


  // -------------------------------------------------------------------
  // Definition of FixedArray class
  // -------------------------------------------------------------------
  template<typename Type, bool IsActive, Index J0, Index J1 = 0, 
	   Index J2 = 0, Index J3 = 0, Index J4 = 0, Index J5 = 0, Index J6 = 0>
  class FixedArray
    : public Expression<Type,FixedArray<Type,IsActive,J0,J1,J2,J3,J4,J5,J6> >,
      protected internal::GradientIndex<IsActive> {

  public:
    // -------------------------------------------------------------------
    // FixedArray: 1. Static Definitions
    // -------------------------------------------------------------------

    // The Expression base class needs access to some protected member
    // functions in section 5
    friend struct Expression<Type,FixedArray<Type,IsActive,J0,J1,J2,J3,J4,J5,J6> >;

    // Static definitions to enable the properties of this type of
    // expression to be discerned at compile time
    static const bool is_active  = IsActive;
    static const bool is_lvalue  = true;
    static const int  rank       = fixed_array<J0,J1,J2,J3,J4,J5,J6>::rank;
    static const int  length_    = fixed_array<J0,J1,J2,J3,J4,J5,J6>::length;
    static const int  n_active   = IsActive * (1 + is_complex<Type>::value);
    static const int  n_scratch  = 0;
    static const int  n_arrays   = 1;
    static const bool is_vectorizable = Packet<Type>::is_vectorized;

  protected:
    template <int Dim, Index X0, Index X1, Index X2,
	      Index X3, Index X4, Index X5, Index X6>
    struct dimension_alias { };
    template <Index X0, Index X1, Index X2,
	      Index X3, Index X4, Index X5, Index X6>
    struct dimension_alias<0,X0,X1,X2,X3,X4,X5,X6>
    { static const Index value = X0; };
    template <Index X0, Index X1, Index X2,
	      Index X3, Index X4, Index X5, Index X6>
    struct dimension_alias<1,X0,X1,X2,X3,X4,X5,X6>
    { static const Index value = X1; };
    template <Index X0, Index X1, Index X2,
	      Index X3, Index X4, Index X5, Index X6>
    struct dimension_alias<2,X0,X1,X2,X3,X4,X5,X6>
    { static const Index value = X2; };
    template <Index X0, Index X1, Index X2,
	      Index X3, Index X4, Index X5, Index X6>
    struct dimension_alias<3,X0,X1,X2,X3,X4,X5,X6>
    { static const Index value = X3; };
    template <Index X0, Index X1, Index X2,
	      Index X3, Index X4, Index X5, Index X6>
    struct dimension_alias<4,X0,X1,X2,X3,X4,X5,X6>
    { static const Index value = X4; };
    template <Index X0, Index X1, Index X2,
	      Index X3, Index X4, Index X5, Index X6>
    struct dimension_alias<5,X0,X1,X2,X3,X4,X5,X6>
    { static const Index value = X5; };
    template <Index X0, Index X1, Index X2,
	      Index X3, Index X4, Index X5, Index X6>
    struct dimension_alias<6,X0,X1,X2,X3,X4,X5,X6>
    { static const Index value = X6; };

  public:
    template <int Dim> struct dimension_ { static const int value 
      = dimension_alias<Dim,J0,J1,J2,J3,J4,J5,J6>::value; };

    template <int RankMinusDim, int Dim>
    struct offset_helper { 
      static const Index value = // Dim == Rank-1 ? 1 :
	dimension_<Dim+1>::value*offset_helper<RankMinusDim-1, Dim+1>::value; 
    };
    template <int Dim>
    struct offset_helper<1,Dim> { static const Index value = 1; };
    template <int Dim>
    struct offset_helper<0,Dim> { static const Index value = 1; };
    template <int Dim>
    struct offset_helper<-1,Dim> { static const Index value = 1; };
    template <int Dim>
    struct offset_helper<-2,Dim> { static const Index value = 1; };
    template <int Dim>
    struct offset_helper<-3,Dim> { static const Index value = 1; };
    template <int Dim>
    struct offset_helper<-4,Dim> { static const Index value = 1; };
    template <int Dim>
    struct offset_helper<-5,Dim> { static const Index value = 1; };

    template <int Dim> struct offset_ { static const Index value
      = offset_helper<rank-Dim, Dim>::value; };


    // -------------------------------------------------------------------
    // FixedArray: 2. Constructors
    // -------------------------------------------------------------------
    
    // Initialize an empty array
    FixedArray() : GradientIndex<IsActive>(length_, false) {
      ADEPT_STATIC_ASSERT(!(std::numeric_limits<Type>::is_integer
			    && IsActive), CANNOT_CREATE_ACTIVE_FIXED_ARRAY_OF_INTEGERS); 
    }

    // Copy constructor copies the data, unlike in the Array class
    FixedArray(const FixedArray& rhs) 
      : GradientIndex<IsActive>(length_, false)
    { *this = rhs; }

  public:
    // Initialize with an expression on the right hand side by
    // evaluating the expression, requiring the ranks to be equal.
    // Note that this constructor enables expressions to be used as
    // arguments to functions that expect an array - to prevent this
    // implicit conversion, use the "explicit" keyword.
    template<typename EType, class E>
    FixedArray(const Expression<EType, E>& rhs,
	  typename enable_if<E::rank == rank,int>::type = 0)
      : GradientIndex<IsActive>(length_, false)
    { *this = rhs; }

#ifdef ADEPT_CXX11_FEATURES
    // Initialize from initializer list
    template <typename T>
    FixedArray(std::initializer_list<T> list) 
      : GradientIndex<IsActive>(length_,false) { *this = list; }

    // The unfortunate restrictions on initializer_list constructors
    // mean that each possible Array rank needs explicit treatment
    template <typename T>
    FixedArray(std::initializer_list<
	  std::initializer_list<T> > list)
      : GradientIndex<IsActive>(length_,false) { *this = list; }

    template <typename T>
    FixedArray(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > list)
      : GradientIndex<IsActive>(length_,false) { *this = list; }

    template <typename T>
    FixedArray(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > > list)
      : GradientIndex<IsActive>(length_,false) { *this = list; }

    template <typename T>
    FixedArray(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > > > list)
      : GradientIndex<IsActive>(length_,false) { *this = list; }

    template <typename T>
    FixedArray(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > > > > list)
      : GradientIndex<IsActive>(length_,false) { *this = list; }

    template <typename T>
    FixedArray(std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<
	  std::initializer_list<T> > > > > > > list)
      : GradientIndex<IsActive>(length_,false) { *this = list; }
    
#endif

    // Destructor: if the data are stored in a Storage object then we
    // tell it that one fewer object is linking to it; if the number
    // of links to it drops to zero, it will destruct itself and
    // deallocate the memory.
    ~FixedArray()
    { GradientIndex<IsActive>::unregister(length_); }

    // -------------------------------------------------------------------
    // FixedArray: 3. Assignment operators
    // -------------------------------------------------------------------

    // Assignment to another matrix: copy the data...
    // Ideally we would like this to fall back to the operator=(const
    // Expression&) function, but if we don't define a copy assignment
    // operator then C++ will generate a default one :-(
    FixedArray& operator=(const FixedArray& rhs) {
      *this = static_cast<const Expression<Type,FixedArray>&> (rhs);
      return *this;
    }

    // Assignment to an array expression of the same rank
    template <typename EType, class E>
    typename enable_if<E::rank == rank, FixedArray&>::type
    inline
    operator=(const Expression<EType,E>& rhs) {
#ifndef ADEPT_NO_DIMENSION_CHECKING
      ExpressionSize<rank> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "FixedArray size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (!compatible(dims, dimensions())) {
	std::string str = "Expr";
	str += dims.str() + " object assigned to " + expression_string_();
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
#endif
      // Select active/passive version by delegating to a protected
      // function
      assign_expression_<rank, IsActive, E::is_active>(rhs);

      return *this;
    }

    // Assignment to a single value copies to every element
    template <typename RType>
    typename enable_if<is_not_expression<RType>::value, FixedArray&>::type
    operator=(RType rhs) {
      assign_inactive_scalar_<rank,IsActive>(rhs);
      return *this;
    }

    // Assign active scalar expression to an active array by first
    // converting the RHS to an active scalar
    template <typename EType, class E>
    typename enable_if<E::rank == 0 && (rank > 0) && IsActive && !E::is_lvalue,
      FixedArray&>::type
    operator=(const Expression<EType,E>& rhs) {
      Active<EType> x = rhs;
      *this = x;
      return *this;
    }

    // Assign an active scalar to an active array
    template <typename PType>
    FixedArray& 
    operator=(const Active<PType>& rhs) {
      ADEPT_STATIC_ASSERT(IsActive, ATTEMPT_TO_ASSIGN_ACTIVE_SCALAR_TO_INACTIVE_FIXED_ARRAY);
#ifdef ADEPT_RECORDING_PAUSABLE
      if (!ADEPT_ACTIVE_STACK->is_recording()) {
	assign_inactive_scalar_<rank,IsActive>(rhs.scalar_value());
	return *this;
      }
#endif
      // In case PType != Type we make a local copy to minimize type
      // conversions
      Type val = rhs.scalar_value();
	
      ADEPT_ACTIVE_STACK->check_space(length_);
      for (Index i = 0; i < length_; ++i) {
	data_[i] = val;
	ADEPT_ACTIVE_STACK->push_rhs(1.0, rhs.gradient_index());
	ADEPT_ACTIVE_STACK->push_lhs(gradient_index()+i);
      }

      return *this;
    }
    
#define ADEPT_DEFINE_OPERATOR(OPERATOR, OPSYMBOL)		\
    template <class RType>					\
    FixedArray& OPERATOR(const RType& rhs) {			\
      return *this = noalias(*this OPSYMBOL rhs);		\
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
    typename enable_if<B::rank == rank, Where<FixedArray,B> >::type
    where(const Expression<bool,B>& bool_expr) {
#ifndef ADEPT_NO_DIMENSION_CHECKING
      ExpressionSize<rank> dims;
      if (!bool_expr.get_dimensions(dims)) {
	std::string str = "FixedArray size mismatch in "
	  + bool_expr.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (dims != dimensions()) {
	throw size_mismatch("Boolean expression of different size"
			    ADEPT_EXCEPTION_LOCATION);
      }
#endif
      return Where<FixedArray,B>(*this, bool_expr.cast());
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
#ifndef ADEPT_NO_DIMENSION_CHECKING
      // Assume size of bool_expr already checked
      ExpressionSize<rank> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "FixedArray size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (!compatible(dims,dimensions())) {
	throw size_mismatch("Right-hand-side of \"where\" construct of incompatible size"
			    ADEPT_EXCEPTION_LOCATION);
      }
#endif
      // Select active/passive version by delegating to a
      // protected function
      assign_conditional_<IsActive>(bool_expr.cast(), rhs.cast());
      //      return *this;
    }

#ifdef ADEPT_CXX11_FEATURES
    // Assignment of a FixedArray to an initializer list; the first ought
    // to only work for vectors
    template <typename T>
    typename enable_if<std::is_convertible<T,Type>::value, FixedArray&>::type
    operator=(std::initializer_list<T> list) {
      ADEPT_STATIC_ASSERT(rank==1, RANK_MISMATCH_IN_INITIALIZER_LIST);

      if (list.size() > J0) {
	throw size_mismatch("Initializer list is larger than Vector in assignment"
			    ADEPT_EXCEPTION_LOCATION);
      }
      // Zero the whole array first in order that automatic
      // differentiation works
      *this = 0;
      Index index = 0;
      for (auto i = std::begin(list); i < std::end(list); ++i,
	   ++index) {
	data_[index*offset_<0>::value] = *i;	
      }
      return *this;
    }

    // Assignment of a higher rank Array to a list of lists...
    template <class IType>
    FixedArray& operator=(std::initializer_list<std::initializer_list<IType> > list) {
      ADEPT_STATIC_ASSERT(rank==initializer_list_rank<IType>::value+2,
      			  RANK_MISMATCH_IN_INITIALIZER_LIST);
      if (list.size() > J0) {
	throw size_mismatch("Multi-dimensional initializer list larger than slowest-varying dimension of Array"
			    ADEPT_EXCEPTION_LOCATION);
      }
      // Zero the whole array first in order that automatic
      // differentiation works
      *this = 0;

      // Enact the assignment using the Array version
      inactive_link() = list;
      return *this;
    }
#endif
  
    // -------------------------------------------------------------------
    // FixedArray: 4. Access functions, particularly operator()
    // -------------------------------------------------------------------
  
    // Get l-value of the element at the specified coordinates
    typename active_reference<Type,IsActive>::type
    get_lvalue(const ExpressionSize<rank>& i) {
      return get_lvalue_<IsActive>(index_(i));
    }
    
    typename active_scalar<Type,IsActive>::type
    get_rvalue(const ExpressionSize<rank>& i) const {
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
    //    const Type& get(const ExpressionSize<rank>& i) const {
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
    typename enable_if<rank==1 && all_scalar_ints<1,I0>::value && !IsActive, Type&>::type
    operator()(I0 i0) 
    { return data_[get_index_with_len(i0,J0)]; }
    
    // r-value access to inactive array with function-call operator
    template <typename I0>
    typename enable_if<rank==1 && all_scalar_ints<1,I0>::value && !IsActive, const Type&>::type
    operator()(I0 i0) const
    { return data_[get_index_with_len(i0,J0)]; }

    // l-value access to inactive array with element-access operator
    template <typename I0>
    typename enable_if<rank==1 && all_scalar_ints<1,I0>::value && !IsActive, Type&>::type
    operator[](I0 i0) 
    { return data_[get_index_with_len(i0,J0)]; }

    // r-value access to inactive array with element-access operator
    template <typename I0>
    typename enable_if<rank==1 && all_scalar_ints<1,I0>::value && !IsActive, const Type&>::type
    operator[](I0 i0) const
    { return data_[get_index_with_len(i0,J0)]; }
 
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
    typename enable_if<rank==1 && all_scalar_ints<1,I0>::value && IsActive,
		       ActiveReference<Type> >::type
    operator()(I0 i0) {
      Index offset = get_index_with_len(i0,J0);
      return ActiveReference<Type>(data_[offset], gradient_index()+offset);
    }
    
    // r-value access to active array with function-call operator
    template <typename I0>
    typename enable_if<rank==1 && all_scalar_ints<1,I0>::value && IsActive,
		       ActiveConstReference<Type> >::type
    operator()(I0 i0) const {
      Index offset = get_index_with_len(i0,J0);
      return ActiveConstReference<Type>(data_[offset], gradient_index()+offset);
    }
  
    // l-value access to active array with element-access operator
    template <typename I0>
    typename enable_if<rank==1 && all_scalar_ints<1,I0>::value && IsActive,
		       ActiveReference<Type> >::type
    operator[](I0 i0) {
      Index offset = get_index_with_len(i0,J0);
      return ActiveReference<Type>(data_[offset], gradient_index()+offset);
    }
    
    // r-value access to active array with element-access operator
    template <typename I0>
    typename enable_if<rank==1 && all_scalar_ints<1,I0>::value && IsActive,
		       ActiveConstReference<Type> >::type
    operator[](I0 i0) const {
      Index offset = get_index_with_len(i0,J0);
      return ActiveConstReference<Type>(data_[offset], gradient_index()+offset);
    }
      
    // 2D array l-value and r-value access
    template <typename I0, typename I1>
    typename enable_if<rank==2 && all_scalar_ints<2,I0,I1>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1) {
      return get_scalar_reference<IsActive>(
		    get_index_with_len(i0,J0)*J1
		  + get_index_with_len(i1,J1));
    }
    template <typename I0, typename I1>
    typename enable_if<rank==2 && all_scalar_ints<2,I0,I1>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1) const {
      return get_scalar_reference<IsActive>(
		    get_index_with_len(i0,J0)*J1
		  + get_index_with_len(i1,J1));
    }
  
    // 3D array l-value and r-value access
    template <typename I0, typename I1, typename I2>
    typename enable_if<rank==3 && all_scalar_ints<3,I0,I1,I2>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2) {
      return get_scalar_reference<IsActive>(J2*(J1*get_index_with_len(i0,J0)
						+ get_index_with_len(i1,J1))
					    + get_index_with_len(i2,J2));
    }
    template <typename I0, typename I1, typename I2>
    typename enable_if<rank==3 && all_scalar_ints<3,I0,I1,I2>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2) const {
      return get_scalar_reference<IsActive>(J2*(J1*get_index_with_len(i0,J0)
						+ get_index_with_len(i1,J1))
					    + get_index_with_len(i2,J2));
    }

    // 4D array l-value and r-value access
    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<rank==4 && all_scalar_ints<4,I0,I1,I2,I3>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3) {
      return get_scalar_reference<IsActive>(J3*(J2*(J1*get_index_with_len(i0,J0)
						    + get_index_with_len(i1,J1))
						+ get_index_with_len(i2,J2))
					    + get_index_with_len(i3,J3));
    }
    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<rank==4 && all_scalar_ints<4,I0,I1,I2,I3>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3) const {
      return get_scalar_reference<IsActive>(J3*(J2*(J1*get_index_with_len(i0,J0)
						    + get_index_with_len(i1,J1))
						+ get_index_with_len(i2,J2))
					    + get_index_with_len(i3,J3));
    }

    // 5D array l-value and r-value access
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4>
    typename enable_if<rank==5 && all_scalar_ints<5,I0,I1,I2,I3,I4>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) {
      return get_scalar_reference<IsActive>(J4*(J3*(J2*(J1*get_index_with_len(i0,J0)
							+ get_index_with_len(i1,J1))
						    + get_index_with_len(i2,J2))
						+ get_index_with_len(i3,J3))
					    + get_index_with_len(i4,J4));
    }
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4>
    typename enable_if<rank==5 && all_scalar_ints<5,I0,I1,I2,I3,I4>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const {
      return get_scalar_reference<IsActive>(J4*(J3*(J2*(J1*get_index_with_len(i0,J0)
							+ get_index_with_len(i1,J1))
						    + get_index_with_len(i2,J2))
						+ get_index_with_len(i3,J3))
					    + get_index_with_len(i4,J4));
    }

    // 6D array l-value and r-value access
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5>
    typename enable_if<rank==6 && all_scalar_ints<6,I0,I1,I2,I3,I4,I5>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) {
      return get_scalar_reference<IsActive>(J5*(J4*(J3*(J2*(J1*get_index_with_len(i0,J0)
							    + get_index_with_len(i1,J1))
							+ get_index_with_len(i2,J2))
						    + get_index_with_len(i3,J3))
						+ get_index_with_len(i4,J4))
					    + get_index_with_len(i5,J5));
    }
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5>
    typename enable_if<rank==6 && all_scalar_ints<6,I0,I1,I2,I3,I4,I5>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) const {
      return get_scalar_reference<IsActive>(J5*(J4*(J3*(J2*(J1*get_index_with_len(i0,J0)
							    + get_index_with_len(i1,J1))
							+ get_index_with_len(i2,J2))
						    + get_index_with_len(i3,J3))
						+ get_index_with_len(i4,J4))
					    + get_index_with_len(i5,J5));
    }

    // 7D array l-value and r-value access
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5, typename I6>
    typename enable_if<rank==7 && all_scalar_ints<7,I0,I1,I2,I3,I4,I5,I6>::value,
		       typename active_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6) {
      return get_scalar_reference<IsActive>(J6*(J5*(J4*(J3*(J2*(J1*get_index_with_len(i0,J0)
								+ get_index_with_len(i1,J1))
							    + get_index_with_len(i2,J2))
							+ get_index_with_len(i3,J3))
						    + get_index_with_len(i4,J4))
						+ get_index_with_len(i5,J5))
					    + get_index_with_len(i6,J6));
    }
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5, typename I6>
    typename enable_if<rank==7 && all_scalar_ints<7,I0,I1,I2,I3,I4,I5,I6>::value,
		       typename active_const_reference<Type,IsActive>::type>::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6) const {
      return get_scalar_reference<IsActive>(J6*(J5*(J4*(J3*(J2*(J1*get_index_with_len(i0,J0)
								+ get_index_with_len(i1,J1))
							    + get_index_with_len(i2,J2))
							+ get_index_with_len(i3,J3))
						    + get_index_with_len(i4,J4))
						+ get_index_with_len(i5,J5))
					    + get_index_with_len(i6,J6));
    }
   

    // The following define the case when operator() is called and one
    // of the arguments is a "range" object (an object that describes
    // a range of indices that are either contiguous or separated by a
    // fixed stride), while all others are of integer type (or a
    // rank-0 expression of integer type). An Array object is returned
    // with a rank that may be reduced from that of the original
    // array, by one for each dimension that was indexed by an
    // integer. The new array points to a subset of the original data,
    // so modifying it will modify the original array.

    // First the case of a vector where we know the argument must be a
    // "range" object
    template <typename I0>
    typename enable_if<is_ranged<rank,I0>::value,
		       Array<1,Type,IsActive> >::type
    operator()(I0 i0) {
      ExpressionSize<1> new_dim((i0.end(J0) + i0.stride(J0) - i0.begin(J0))
				/i0.stride(J0));
      ExpressionSize<1> new_offset(i0.stride(J0));
      return Array<1,Type,IsActive>(data_, i0.begin(J0), new_dim, new_offset,
				    GradientIndex<IsActive>::get());
    }
    template <typename I0>
    typename enable_if<is_ranged<rank,I0>::value,
		       const Array<1,Type,IsActive> >::type
    operator()(I0 i0) const {
      ExpressionSize<1> new_dim((i0.end(J0) + i0.stride(J0) - i0.begin(J0))
				/i0.stride(J0));
      ExpressionSize<1> new_offset(i0.stride(J0));
      return Array<1,Type,IsActive>(data_, i0.begin(J0), new_dim, new_offset,
				    GradientIndex<IsActive>::get());
    }

  private:
    // For multi-dimensional arrays, we need a helper function

    // Treat the indexing of dimension "irank" in the case that the
    // index is of integer type
    template <int Rank, typename T, int NewRank>
    typename enable_if<is_scalar_int<T>::value, void>::type
    update_index(const T& i, Index& inew_rank, Index& ibegin,
		 ExpressionSize<NewRank>& new_dim, 
		 ExpressionSize<NewRank>& new_offset) const {
      ibegin += get_index_with_len(i,dimension_<Rank>::value)*offset_<Rank>::value;
    }

    // Treat the indexing of dimension "irank" in the case that the
    // index is a "range" object
    template <int Rank, typename T, int NewRank>
    typename enable_if<is_range<T>::value, void>::type
    update_index(const T& i, Index& inew_rank, Index& ibegin,
		 ExpressionSize<NewRank>& new_dim, 
		 ExpressionSize<NewRank>& new_offset) const {
      ibegin += i.begin(dimension_<Rank>::value)*offset_<Rank>::value;
      new_dim[inew_rank]
      = (i.end(dimension_<Rank>::value)
	 + i.stride(dimension_<Rank>::value)-i.begin(dimension_<Rank>::value))
      / i.stride(dimension_<Rank>::value);
      new_offset[inew_rank] = i.stride(dimension_<Rank>::value)*offset_<Rank>::value;
      ++inew_rank;
    }
  
  public:

    // Now the individual overloads for each number of arguments, up
    // to 7, with separate r-value (const) and l-value (non-const)
    // versions
    template <typename I0, typename I1>
    typename enable_if<is_ranged<rank,I0,I1>::value,
		       Array<is_ranged<rank,I0,I1>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1) {
      static const int new_rank = is_ranged<rank,I0,I1>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }

    template <typename I0, typename I1>
    typename enable_if<is_ranged<rank,I0,I1>::value,
		       const Array<is_ranged<rank,I0,I1>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1) const {
      static const int new_rank = is_ranged<rank,I0,I1>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }

    template <typename I0, typename I1, typename I2>
    typename enable_if<is_ranged<rank,I0,I1,I2>::value,
	       Array<is_ranged<rank,I0,I1,I2>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2) {
      static const int new_rank = is_ranged<rank,I0,I1,I2>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }

    template <typename I0, typename I1, typename I2>
    typename enable_if<is_ranged<rank,I0,I1,I2>::value,
	       const Array<is_ranged<rank,I0,I1,I2>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2) const {
      static const int new_rank = is_ranged<rank,I0,I1,I2>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }

    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<is_ranged<rank,I0,I1,I2,I3>::value,
       Array<is_ranged<rank,I0,I1,I2,I3>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3) {
      static const int new_rank = is_ranged<rank,I0,I1,I2,I3>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      update_index<3>(i3, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }

    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<is_ranged<rank,I0,I1,I2,I3>::value,
       const Array<is_ranged<rank,I0,I1,I2,I3>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3) const {
      static const int new_rank = is_ranged<rank,I0,I1,I2,I3>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      update_index<3>(i3, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }

    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4>
    typename enable_if<is_ranged<rank,I0,I1,I2,I3,I4>::value,
       Array<is_ranged<rank,I0,I1,I2,I3,I4>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) {
      static const int new_rank = is_ranged<rank,I0,I1,I2,I3,I4>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      update_index<3>(i3, inew_rank, ibegin, new_dim, new_offset);
      update_index<4>(i4, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }
  
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4>
    typename enable_if<is_ranged<rank,I0,I1,I2,I3,I4>::value,
       const Array<is_ranged<rank,I0,I1,I2,I3,I4>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4) const {
      static const int new_rank = is_ranged<rank,I0,I1,I2,I3,I4>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      update_index<3>(i3, inew_rank, ibegin, new_dim, new_offset);
      update_index<4>(i4, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }
  
    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5>
    typename enable_if<is_ranged<rank,I0,I1,I2,I3,I4,I5>::value,
       Array<is_ranged<rank,I0,I1,I2,I3,I4,I5>::count,Type,IsActive> >::type
     operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) {
      static const int new_rank = is_ranged<rank,I0,I1,I2,I3,I4,I5>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      update_index<3>(i3, inew_rank, ibegin, new_dim, new_offset);
      update_index<4>(i4, inew_rank, ibegin, new_dim, new_offset);
      update_index<5>(i5, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }


    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5>
    typename enable_if<is_ranged<rank,I0,I1,I2,I3,I4,I5>::value,
       const Array<is_ranged<rank,I0,I1,I2,I3,I4,I5>::count,Type,IsActive> >::type
     operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5) const {
      static const int new_rank = is_ranged<rank,I0,I1,I2,I3,I4,I5>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      update_index<3>(i3, inew_rank, ibegin, new_dim, new_offset);
      update_index<4>(i4, inew_rank, ibegin, new_dim, new_offset);
      update_index<5>(i5, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }

    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5, typename I6>
    typename enable_if<is_ranged<rank,I0,I1,I2,I3,I4,I5,I6>::value,
       Array<is_ranged<rank,I0,I1,I2,I3,I4,I5,I6>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6) {
      static const int new_rank = is_ranged<rank,I0,I1,I2,I3,I4,I5,I6>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      update_index<3>(i3, inew_rank, ibegin, new_dim, new_offset);
      update_index<4>(i4, inew_rank, ibegin, new_dim, new_offset);
      update_index<5>(i5, inew_rank, ibegin, new_dim, new_offset);
      update_index<6>(i6, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }

    template <typename I0, typename I1, typename I2, typename I3,
	      typename I4, typename I5, typename I6>
    typename enable_if<is_ranged<rank,I0,I1,I2,I3,I4,I5,I6>::value,
       const Array<is_ranged<rank,I0,I1,I2,I3,I4,I5,I6>::count,Type,IsActive> >::type
    operator()(I0 i0, I1 i1, I2 i2, I3 i3, I4 i4, I5 i5, I6 i6) const {
      static const int new_rank = is_ranged<rank,I0,I1,I2,I3,I4,I5,I6>::count;
      ExpressionSize<new_rank> new_dim;
      ExpressionSize<new_rank> new_offset;
      Index inew_rank = 0;
      Index ibegin = 0;
      update_index<0>(i0, inew_rank, ibegin, new_dim, new_offset);
      update_index<1>(i1, inew_rank, ibegin, new_dim, new_offset);
      update_index<2>(i2, inew_rank, ibegin, new_dim, new_offset);
      update_index<3>(i3, inew_rank, ibegin, new_dim, new_offset);
      update_index<4>(i4, inew_rank, ibegin, new_dim, new_offset);
      update_index<5>(i5, inew_rank, ibegin, new_dim, new_offset);
      update_index<6>(i6, inew_rank, ibegin, new_dim, new_offset);
      return Array<new_rank,Type,IsActive>(data_, ibegin, new_dim, new_offset,
					   GradientIndex<IsActive>::get());
    }
  
    // If one or more of the indices is not guaranteed to be monotonic
    // at compile time then we must return an IndexedArray, now done
    // for all possible numbers of arguments

    // Indexing a 1D array
    template <typename I0>
    typename enable_if<rank == 1 && is_int_vector<I0>::value
		       && !is_ranged<rank,I0>::value,
		       IndexedArray<rank,Type,IsActive,FixedArray,I0> >::type
    operator()(const I0& i0) {
      return IndexedArray<rank,Type,IsActive,FixedArray,I0>(*this, i0);
    }
    template <typename I0>
    typename enable_if<rank == 1 && is_int_vector<I0>::value
		       && !is_ranged<rank,I0>::value,
		       const IndexedArray<rank,Type,IsActive,
					  FixedArray,I0> >::type
    operator()(const I0& i0) const {
      return IndexedArray<rank,Type,IsActive,
			  FixedArray,I0>(*const_cast<FixedArray*>(this), i0);
    }
  
    // Indexing a 2D array
    template <typename I0, typename I1>
    typename enable_if<rank == 2 && is_irreg_indexed<rank,I0,I1>::value,
		       IndexedArray<is_irreg_indexed<rank,I0,I1>::count,
				    Type,IsActive,FixedArray,I0,I1> >::type
    operator()(const I0& i0, const I1& i1) {
      static const int new_rank = is_irreg_indexed<rank,I0,I1>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,I0,I1>(*this, i0, i1);
    }
    template <typename I0, typename I1>
    typename enable_if<rank == 2 && is_irreg_indexed<rank,I0,I1>::value,
		       const IndexedArray<is_irreg_indexed<rank,I0,I1>::count,
				    Type,IsActive,FixedArray,I0,I1> >::type
    operator()(const I0& i0, const I1& i1) const {
      static const int new_rank = is_irreg_indexed<rank,I0,I1>::count;
      return IndexedArray<new_rank,Type,IsActive,
			  FixedArray,I0,I1>(*const_cast<FixedArray*>(this), i0, i1);
    }

    // Indexing a 3D array
    template <typename I0, typename I1, typename I2>
    typename enable_if<rank == 3 && is_irreg_indexed<rank,I0,I1,I2>::value,
		       IndexedArray<is_irreg_indexed<rank,I0,I1,I2>::count,
				    Type,IsActive,FixedArray,I0,I1,I2> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2) {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,
			  I0,I1,I2>(*this, i0, i1, i2);
    }
    template <typename I0, typename I1, typename I2>
    typename enable_if<rank == 3 && is_irreg_indexed<rank,I0,I1,I2>::value,
		       const IndexedArray<is_irreg_indexed<rank,
							   I0,I1,I2>::count,
				    Type,IsActive,FixedArray,I0,I1,I2> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2) const {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,
			  I0,I1,I2>(*const_cast<FixedArray*>(this), i0, i1, i2);
    }

    // Indexing a 4D array
    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<rank == 4 && is_irreg_indexed<rank,I0,I1,I2,I3>::value,
		       IndexedArray<is_irreg_indexed<rank,I0,I1,I2,I3>::count,
				    Type,IsActive,FixedArray,I0,I1,I2,I3> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, const I3& i3) {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2,I3>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,
			  I0,I1,I2,I3>(*this, i0, i1, i2, i3);
    }
    template <typename I0, typename I1, typename I2, typename I3>
    typename enable_if<rank == 4 && is_irreg_indexed<rank,I0,I1,I2,I3>::value,
		       const IndexedArray<is_irreg_indexed<rank,I0,I1,
							   I2,I3>::count,
				    Type,IsActive,FixedArray,I0,I1,I2,I3> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, const I3& i3) const {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2,I3>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,I0,I1,I2,
			  I3>(*const_cast<FixedArray*>(this), i0, i1, i2, i3);
    }

    // Indexing a 5D array
    template <typename I0, typename I1, typename I2, typename I3, typename I4>
    typename enable_if<rank == 5
		       && is_irreg_indexed<rank,I0,I1,I2,I3,I4>::value,
		       IndexedArray<is_irreg_indexed<rank,I0,I1,I2,
						     I3,I4>::count,
			    Type,IsActive,FixedArray,I0,I1,I2,I3,I4> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, 
	       const I3& i3, const I4& i4) {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2,I3,
						   I4>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,I0,I1,I2,I3,
			  I4>(*this, i0, i1, i2, i3, i4);
    }
    template <typename I0, typename I1, typename I2, typename I3, typename I4>
    typename enable_if<rank == 5
		       && is_irreg_indexed<rank,I0,I1,I2,I3,I4>::value,
		       const IndexedArray<is_irreg_indexed<rank,I0,I1,I2,
							   I3,I4>::count,
				  Type,IsActive,FixedArray,I0,I1,I2,I3,I4> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, 
	       const I3& i3, const I4& i4) const {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2,I3,
						   I4>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,I0,I1,I2,I3,
			  I4>(*const_cast<FixedArray*>(this), i0, i1, i2, i3, i4);
    }

    // Indexing a 6D array
    template <typename I0, typename I1, typename I2,
	      typename I3, typename I4, typename I5>
    typename enable_if<rank == 6
		       && is_irreg_indexed<rank,I0,I1,I2,I3,I4,I5>::value,
		       IndexedArray<is_irreg_indexed<rank,I0,I1,I2,I3,
							   I4,I5>::count,
			  Type,IsActive,FixedArray,I0,I1,I2,I3,I4,I5> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, 
	       const I3& i3, const I4& i4, const I5& i5) {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2,I3,
						   I4,I5>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,I0,I1,I2,I3,I4,
			  I5>(*this,i0,i1,i2,i3,i4,i5);
    }
    template <typename I0, typename I1, typename I2,
	      typename I3, typename I4, typename I5>
    typename enable_if<rank == 6
		       && is_irreg_indexed<rank,I0,I1,I2,I3,I4,I5>::value,
		       const IndexedArray<is_irreg_indexed<rank,I0,I1,I2,I3,
							   I4,I5>::count,
			  Type,IsActive,FixedArray,I0,I1,I2,I3,I4,I5> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, 
	       const I3& i3, const I4& i4, const I5& i5) const {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2,I3,
						   I4,I5>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,I0,I1,I2,I3,I4,
			  I5>(*const_cast<FixedArray*>(this),i0,i1,i2,i3,i4,i5);
    }

    // Indexing a 7D array
    template <typename I0, typename I1, typename I2,
	      typename I3, typename I4, typename I5, typename I6>
    typename enable_if<rank == 7
		       && is_irreg_indexed<rank,I0,I1,I2,I3,I4,I5>::value,
		       IndexedArray<is_irreg_indexed<rank,I0,I1,I2,I3,
						     I4,I5,I6>::count,
			  Type,IsActive,FixedArray,I0,I1,I2,I3,I4,I5,I6> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, const I3& i3,
	       const I4& i4, const I5& i5, const I6& i6) {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2,I3,
						   I4,I5,I6>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,I0,I1,I2,I3,I4,I5,
			  I6>(*this,i0,i1,i2,i3,i4,i5,i6);
    }
    template <typename I0, typename I1, typename I2,
	      typename I3, typename I4, typename I5, typename I6>
    typename enable_if<rank == 7
		       && is_irreg_indexed<rank,I0,I1,I2,I3,I4,I5>::value,
		       const IndexedArray<is_irreg_indexed<rank,I0,I1,I2,I3,
							   I4,I5,I6>::count,
			  Type,IsActive,FixedArray,I0,I1,I2,I3,I4,I5,I6> >::type
    operator()(const I0& i0, const I1& i1, const I2& i2, const I3& i3,
	       const I4& i4, const I5& i5, const I6& i6) const {
      static const int new_rank = is_irreg_indexed<rank,I0,I1,I2,I3,
						   I4,I5,I6>::count;
      return IndexedArray<new_rank,Type,IsActive,FixedArray,I0,I1,I2,I3,I4,I5,
			  I6>(*const_cast<FixedArray*>(this),i0,i1,i2,i3,i4,i5,i6);
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
    typename enable_if<is_scalar_int<T>::value && (rank > 1),
      Array<rank-1,Type,IsActive> >::type
    operator[](T i) {
      int index = get_index_with_len(i,J0)*offset_<0>::value;
      ExpressionSize<rank-1> new_dim;
      ExpressionSize<rank-1> new_offset;
      ExpressionSize<rank> dims = dimensions();
      ExpressionSize<rank> offs = offset();
      for (int j = 1; j < rank; ++j) {
	new_dim[j-1] = dims[j];
	new_offset[j-1] = offs[j];
      }
      return Array<rank-1,Type,IsActive>(data_, index, new_dim, new_offset,
					  GradientIndex<IsActive>::get());
    }
    
    // diag_matrix(), where *this is a 1D array, returns a DiagMatrix
    // containing the data as the diagonal pointing to the original
    // data, Can be used as an lvalue.  Defined in SpecialMatrix.h
    SpecialMatrix<Type, internal::BandEngine<internal::ROW_MAJOR,0,0>, IsActive>
    diag_matrix();
    
    Array<1,Type,IsActive>
    diag_vector(Index offdiag = 0) {
      ADEPT_STATIC_ASSERT(rank == 2, DIAG_VECTOR_ONLY_WORKS_ON_SQUARE_MATRICES);
      if (empty()) {
	// Return an empty vector
	return Array<1,Type,IsActive>();
      }
      else if (J0 != J1) {
	throw invalid_operation("diag_vector member function only applicable to square matrices"
				ADEPT_EXCEPTION_LOCATION);
      }
      else if (offdiag >= 0) {
	Index new_dim = std::min(J0, J1-offdiag);
	return Array<1,Type,IsActive>(data_, offset_<1>::value*offdiag,  
				      ExpressionSize<1>(new_dim),
				      ExpressionSize<1>(offset_<0>::value+offset_<1>::value),
				      GradientIndex<IsActive>::get());
      }
      else {
	Index new_dim = std::min(J0+offdiag, J1);
	return Array<1,Type,IsActive>(data_,-offset_<0>::value*offdiag,  
				      ExpressionSize<1>(new_dim),
				      ExpressionSize<1>(offset_<0>::value+offset_<1>::value),
				      GradientIndex<IsActive>::get());
      }
    }
  

    Array<2,Type,IsActive>
    submatrix_on_diagonal(Index ibegin, Index iend) {
      ADEPT_STATIC_ASSERT(rank == 2,
		SUBMATRIX_ON_DIAGONAL_ONLY_WORKS_ON_SQUARE_MATRICES);
      if (J0 != J1) {
	throw invalid_operation("submatrix_on_diagonal member function only applicable to square matrices"
				ADEPT_EXCEPTION_LOCATION);
      }
      else if (ibegin < 0 || ibegin > iend || iend >= J0) {
	throw index_out_of_bounds("Dimensions out of range in submatrix_on_diagonal"
				  ADEPT_EXCEPTION_LOCATION);
      }
      else {
	Index len = iend-ibegin+1;
	ExpressionSize<2> dim(len,len);
	return Array<2,Type,IsActive>(data_, ibegin*(offset_<0>::value + offset_<1>::value),
				      dim, offset(), GradientIndex<IsActive>::get());
      }
    }

    // For extracting contiguous sections out of an array use the
    // following. Currently this just indexes each dimension with the
    // contiguous range(a,b) index, but in future it may be optimized.

    // 1D array subset
    template <typename B0, typename E0>
    Array<1,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0) {
      ADEPT_STATIC_ASSERT(rank == 1,
			  SUBSET_WITH_2_ARGS_ONLY_ON_RANK_1_ARRAY);
      return (*this)(range(ibegin0,iend0));
    }
    template <typename B0, typename E0>
    const Array<1,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0) const {
      ADEPT_STATIC_ASSERT(rank == 1,
			  SUBSET_WITH_2_ARGS_ONLY_ON_RANK_1_ARRAY);
      return (*this)(range(ibegin0,iend0));
    }

    // 2D array subset
    template <typename B0, typename E0, typename B1, typename E1>
    Array<2,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1) {
      ADEPT_STATIC_ASSERT(rank == 2,
			  SUBSET_WITH_4_ARGS_ONLY_ON_RANK_2_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1));
    }
    template <typename B0, typename E0, typename B1, typename E1>
    const Array<2,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	  const B1& ibegin1, const E1& iend1) const {
      ADEPT_STATIC_ASSERT(rank == 2,
			  SUBSET_WITH_4_ARGS_ONLY_ON_RANK_2_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1));
    }

    // 3D array subset
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2>
    Array<3,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2) {
      ADEPT_STATIC_ASSERT(rank == 3,
			  SUBSET_WITH_6_ARGS_ONLY_ON_RANK_3_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2));
    }     
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2>
    const Array<3,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2) const {
      ADEPT_STATIC_ASSERT(rank == 3,
			  SUBSET_WITH_6_ARGS_ONLY_ON_RANK_3_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2));
    }

    // 4D array subset
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3>
    Array<4,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3) {
      ADEPT_STATIC_ASSERT(rank == 4,
			  SUBSET_WITH_8_ARGS_ONLY_ON_RANK_4_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3));
    }
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3>
    const Array<4,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3) const {
      ADEPT_STATIC_ASSERT(rank == 4,
			  SUBSET_WITH_8_ARGS_ONLY_ON_RANK_4_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3));
    } 

    // 5D array subset
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4>
    Array<5,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4) {
      ADEPT_STATIC_ASSERT(rank == 5,
			  SUBSET_WITH_10_ARGS_ONLY_ON_RANK_5_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4));
    }
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4>
    const Array<5,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4) const {
      ADEPT_STATIC_ASSERT(rank == 5,
			  SUBSET_WITH_10_ARGS_ONLY_ON_RANK_5_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4));
    }

    // 6D array subset
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4, typename B5, typename E5>
    Array<6,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4,
	   const B5& ibegin5, const E5& iend5) {
      ADEPT_STATIC_ASSERT(rank == 6,
			  SUBSET_WITH_12_ARGS_ONLY_ON_RANK_6_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4),range(ibegin5,iend5));
    }
    template <typename B0, typename E0, typename B1, typename E1,
	      typename B2, typename E2, typename B3, typename E3,
	      typename B4, typename E4, typename B5, typename E5>
    const Array<6,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4,
	   const B5& ibegin5, const E5& iend5) const {
      ADEPT_STATIC_ASSERT(rank == 6,
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
    Array<7,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4,
	   const B5& ibegin5, const E5& iend5,
	   const B6& ibegin6, const E6& iend6) {
      ADEPT_STATIC_ASSERT(rank == 7,
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
    const Array<7,Type,IsActive>
    subset(const B0& ibegin0, const E0& iend0, 
	   const B1& ibegin1, const E1& iend1,
	   const B2& ibegin2, const E2& iend2,
	   const B3& ibegin3, const E3& iend3,
	   const B4& ibegin4, const E4& iend4,
	   const B5& ibegin5, const E5& iend5,
	   const B6& ibegin6, const E6& iend6) const {
      ADEPT_STATIC_ASSERT(rank == 7,
			  SUBSET_WITH_14_ARGS_ONLY_ON_RANK_7_ARRAY);
      return (*this)(range(ibegin0,iend0),range(ibegin1,iend1),
		     range(ibegin2,iend2),range(ibegin3,iend3),
		     range(ibegin4,iend4),range(ibegin5,iend5),
		     range(ibegin6,iend6));
    }

    // -------------------------------------------------------------------
    // FixedArray: 5. Public member functions
    // -------------------------------------------------------------------
  
    // STL-like size() returns total length of array
    Index size() const { return length_; }

    bool get_dimensions_(ExpressionSize<rank>& dims) const {
      dims[0] = J0;
      if (J1 > 0) {
	dims[1] = J1;
	if (J2 > 0) {
	  dims[2] = J2;
	  if (J3 > 0) {
	    dims[3] = J3;
	    if (J4 > 0) {
	      dims[4] = J4;
	      if (J5 > 0) {
		dims[5] = J5;
		if (J6 > 0) {
		  dims[6] = J6;
		}
	      }
	    }
	  }
	}
      }
      return true;
    }

    // Return constant reference to dimensions
    ExpressionSize<rank> dimensions() const {
      ExpressionSize<rank> dims;
      get_dimensions_(dims);
      return dims;
    }

    // Return individual dimension
    Index size(int j) const {
      if (j >= rank)  { return  0; }
      else if (j == 0) { return J0; }
      else if (j == 1) { return J1; }
      else if (j == 2) { return J2; }
      else if (j == 3) { return J3; }
      else if (j == 4) { return J4; }
      else if (j == 5) { return J5; }
      else { return J6; }
    }
    Index dimension(int j) const {
      return size(j);
    }

    // Return individual offset
    Index offset(int j) const {
      if (j >= rank)  { return  0; }
      else if (j == 0) { return offset_<0>::value; }
      else if (j == 1) { return offset_<1>::value; }
      else if (j == 2) { return offset_<2>::value; }
      else if (j == 3) { return offset_<3>::value; }
      else if (j == 4) { return offset_<4>::value; }
      else if (j == 5) { return offset_<5>::value; }
      else if (j == 6) { return offset_<6>::value; }
      else { throw invalid_dimension(); }
    }

    // Return constant reference to offsets
    ExpressionSize<rank> offset() const {
      ExpressionSize<rank> offs;
      offs[0] = offset_<0>::value;
      if (J1 > 0) {
	offs[1] = offset_<1>::value;
	if (J2 > 0) {
	  offs[2] = offset_<2>::value;
	  if (J3 > 0) {
	    offs[3] = offset_<3>::value;
	    if (J4 > 0) {
	      offs[4] = offset_<4>::value;
	      if (J5 > 0) {
		offs[5] = offset_<5>::value;
		if (J6 > 0) {
		  offs[6] = offset_<6>::value;
		}
	      }
	    }
	  }
	}
      }
      return offs;
    }

    const Index& last_offset() const { return offset_<rank-1>::value; }

    // Return true if the array is empty
    bool empty() const { return (J0 == 0); }

    // Return a string describing the array
    std::string info_string() const {
      std::stringstream str;
      str << "FixedArray<" << rank << ">, dim=" << dimensions() << ", data_location=" << data_;
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
      ADEPT_STATIC_ASSERT(rank == 1, CAN_ONLY_USE_DATA_POINTER_WITH_INDEX_ON_VECTORS);
      if (data_) {
	return data_ + i;
      }
      else {
	return 0;
      }
    }
    const Type* const_data_pointer(Index i) const { 
      ADEPT_STATIC_ASSERT(rank == 1, CAN_ONLY_USE_CONST_DATA_POINTER_WITH_INDEX_ON_VECTORS);
      if (data_) {
	return data_ + i;
      }
      else {
	return 0;
      }
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

    // By design, FixedArrays are row-major and row-wise access is
    // contiguous
    bool all_arrays_contiguous_() const { return true; }
 
    bool is_aligned_() const {
      return !(reinterpret_cast<std::size_t>(data_) & Packet<Type>::align_mask);
    }

    template <int n>
    int alignment_offset_() const {
      return (reinterpret_cast<std::size_t>(data_)/sizeof(Type)) % n; 
    }

    Type value_with_len_(const Index& j, const Index& len) const {
      ADEPT_STATIC_ASSERT(rank == 1, CANNOT_USE_VALUE_WITH_LEN_ON_ARRAY_OF_RANK_OTHER_THAN_1);
      return data_[j];
    }

    std::string expression_string_() const {
      if (true) {
	std::string a = fixed_array_helper<rank,IsActive>().name();
	a += dimensions().str();
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
    typename enable_if<is_not_expression<RType>::value, FixedArray&>::type
    set_value(RType x) {
      if (!empty()) {
	assign_inactive_scalar_<rank,false>(x);
      }
      return *this;
    }
  
    
    // Return the gradient index for the first element in the array,
    // or -1 if not active
    Index gradient_index() const {
      return GradientIndex<IsActive>::get();
    }

    std::ostream& print(std::ostream& os) const {
      const Array<rank,Type,IsActive> x(*this);
      x.print(os);
      return os;
    }

    // Get pointers to the first and last data members in memory.  
    void data_range(Type const * &data_begin, Type const * &data_end) const {
      data_begin = data_;
      data_end = data_ + length_-1;
    }

  
    // The Stack::independent(x) and Stack::dependent(y) functions add
    // the gradient_index of objects x and y to std::vector<uIndex>
    // objects in Stack. Since x and y may be scalars or arrays, this
    // is best done by delegating to the Active or FixedArray classes.
    template <typename IndexType>
    void push_gradient_indices(std::vector<IndexType>& vec) const {
      ADEPT_STATIC_ASSERT(IsActive,
		  CANNOT_PUSH_GRADIENT_INDICES_FOR_INACTIVE_ARRAY); 
      ExpressionSize<rank> i(0);
      Index gradient_ind = gradient_index();
      Index index = 0;
      int my_rank;
      vec.reserve(vec.size() + size());
      do {
	// Innermost loop - note that the counter is index, not max_index
	for (Index max_index = index + dimension_<rank-1>::value*offset_<rank-1>::value;
	     index < max_index;
	     index += offset_<rank-1>::value) {
	  vec.push_back(gradient_ind + index);
	}
	// Increment counters appropriately depending on which
	// dimensions have been finished
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }

    // Return inactive array linked to original data
    Array<rank, Type, false> inactive_link() {
      return Array<rank, Type, false>(data_, 0, dimensions(), offset(),
				       GradientIndex<IsActive>::get());
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
      Array<2,Type,IsActive> out(const_cast<FixedArray&>(*this));
      // Swap dimensions
      return out.in_place_transpose();
    }

  public:
    // Out-of-place transpose
    Array<2,Type,IsActive>
    T() {
      ADEPT_STATIC_ASSERT(rank == 1 || rank == 2, 
			  TRANSPOSE_ONLY_POSSIBLE_WITH_1D_OR_2D_ARRAYS);
      return my_T<rank>();
    }
    const Array<2,Type,IsActive>
    T() const {
      ADEPT_STATIC_ASSERT(rank == 1 || rank == 2, 
			  TRANSPOSE_ONLY_POSSIBLE_WITH_1D_OR_2D_ARRAYS);
      return my_T<rank>();
    }

    // "permute" is a generalized transpose, returning an FixedArray linked
    // to the current one but with the dimensions rearranged according
    // to idim: idim[0] is the 0-based number of the dimension of the
    // current array that will be dimension 0 of the new array,
    // idim[1] is the number of the dimension of the current array
    // that will be dimension 1 of the new array and so on.
    Array<rank,Type,IsActive> permute(const Index* idim) {
      if (empty()) {
	throw empty_array("Attempt to permute an empty array"
			  ADEPT_EXCEPTION_LOCATION);
      }
      ExpressionSize<rank> new_dims(0);
      ExpressionSize<rank> new_offset;
      ExpressionSize<rank> dims, offs;
      dims = dimensions();
      offs = offset();
      for (int i = 0; i < rank; ++i) {
	if (idim[i] >= 0 && idim[i] < rank) {
	  new_dims[i] = dims[idim[i]];
	  new_offset[i] = offs[idim[i]];
	}
	else {
	  throw invalid_dimension("Dimensions must be in range 0 to rank-1 in permute"
				  ADEPT_EXCEPTION_LOCATION);
	}
      }
      for (int i = 0; i < rank; ++i) {
	if (new_dims[i] == 0) {
	  throw invalid_dimension("Missing dimension in permute"
				  ADEPT_EXCEPTION_LOCATION);
	}
      }
      return Array<rank,Type,IsActive>(data_, 0, new_dims, new_offset,
					GradientIndex<IsActive>::get());
    }

    Array<rank,Type,IsActive> permute(const ExpressionSize<rank>& idim) {
      return permute(&idim[0]);
    }

    // Up to 7 dimensions we can specify the dimensions as separate
    // arguments
    typename enable_if<(rank < 7), Array<rank,Type,IsActive> >::type
    permute(Index i0, Index i1, Index i2 = -1, Index i3 = -1, Index i4 = -1,
	    Index i5 = -1, Index i6 = -1) {
      Index idim[7] = {i0, i1, i2, i3, i4, i5, i6};
      for (int i = 0; i < rank; ++i) {
	if (idim[i] == -1) {
	  throw invalid_dimension("Incorrect number of dimensions provided to permute"
				  ADEPT_EXCEPTION_LOCATION);
	}
      }
      return permute(idim);
    }

    // Return an inactive array of the same type and rank as the
    // present active fixed array, containing the gradients associated
    // with it
    template <typename MyType>
    void get_gradient(Array<rank,MyType,false>& gradient) const {
      ADEPT_STATIC_ASSERT(IsActive,CANNOT_USE_GET_GRADIENT_ON_INACTIVE_ARRAY);
      if (gradient.empty()) {
	gradient.resize(dimensions());
      }
      else if (gradient.dimensions() != dimensions()) {
	throw size_mismatch("Attempt to get_gradient with array of different dimensions"
			    ADEPT_EXCEPTION_LOCATION);
      }
      static const int last = rank-1;
      ExpressionSize<rank> target_offset = gradient.offset();
      ExpressionSize<rank> i(0);
      Index index = 0;
      int my_rank;
      Index index_target = 0;
      Index last_dim_stretch = dimension_<rank-1>::value*offset_<rank-1>::value;
      MyType* target = gradient.data();
      do {
	i[last] = 0;
	index_target = 0;
	for (int r = 0; r < rank-1; r++) {
	  index_target += i[r]*target_offset[r];
	}
	ADEPT_ACTIVE_STACK->get_gradients(gradient_index()+index,
				  gradient_index()+index+last_dim_stretch,
					  target+index_target, offset_<rank-1>::value, target_offset[last]);
	index += last_dim_stretch;
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }


    // Return an inactive array of the same type and rank as the
    // present active array containing the gradients associated with
    // it
    Array<rank,Type,false> get_gradient() const {
      Array<rank,Type,false> gradient;
      get_gradient(gradient);
      return gradient;
    }

    void
    put(std::vector<typename internal::active_scalar<Type,IsActive>::type>& data) const {
      ADEPT_STATIC_ASSERT(rank == 1, PUT_ONLY_AVAILABLE_FOR_RANK_1_ARRAYS);
      if (data.size() != J0) {
	data.resize(J0);
      }
      for (Index i = 0; i < J0; ++i) {
	data[i] = (*this)(i);
      }  
    }

    void
    get(const std::vector<typename internal::active_scalar<Type,IsActive>::type>& data) {
      ADEPT_STATIC_ASSERT(rank == 1, GET_ONLY_AVAILABLE_FOR_RANK_1_ARRAYS);
      if (data.size() != J0) {
	resize(data.size());
      }
      for (Index i = 0; i < J0; ++i) {
	(*this)(i) = data[i];
      }  
    }


    // -------------------------------------------------------------------
    // FixedArray: 6. Member functions accessed by the Expression class
    // -------------------------------------------------------------------

    template <int MyArrayNum, int NArrays>
    void set_location_(const ExpressionSize<rank>& i, 
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
      loc[MyArrayNum] += offset_<rank-1>::value;
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
  


    // -------------------------------------------------------------------
    // FixedArray: 7. Protected member functions
    // -------------------------------------------------------------------
  protected:

    // Return the memory index (relative to data_) for array element
    // indicated by j
    Index index_(Index j[rank]) const {
      Index o = 0;
      ExpressionSize<rank> offs = offset();
      for (int i = 0; i < rank; i++) {
	o += j[i]*offs[i];
      }
      return o;
    }
    Index index_(const ExpressionSize<rank>& j) const {
      Index o = 0;
      for (int i = 0; i < rank; i++) {
	o += j[i]*offset(i);
      }
      return o;
    }

    // Used in traversing through an array
    void advance_index(Index& index, int& my_rank, ExpressionSize<rank>& i) const {
      index -= offset_<rank-1>::value*dimension_<rank-1>::value;
      my_rank = rank-1;
      while (--my_rank >= 0) {
	if (++i[my_rank] >= dimension(my_rank)) {
	  i[my_rank] = 0;
	  index -= offset(my_rank)*(dimension(my_rank)-1);
	}
	else {
	  index += offset(my_rank);
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
	for (Index max_index = index + dimension_<LocalRank-1>::value*offset_<LocalRank-1>::value;
	     index < max_index;
	     index += offset_<LocalRank-1>::value) {
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
	ADEPT_ACTIVE_STACK->push_lhs_range(gradient_ind+index, dimension_<LocalRank-1>::value,
					   offset_<LocalRank-1>::value);
	for (Index max_index = index + dimension_<LocalRank-1>::value*offset_<LocalRank-1>::value;
	     index < max_index; index += offset_<LocalRank-1>::value) {
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
    typename enable_if<!LocalIsActive,void>::type
    assign_expression_(const E& rhs) {
      ADEPT_STATIC_ASSERT(!EIsActive, CANNOT_ASSIGN_ACTIVE_EXPRESSION_TO_INACTIVE_ARRAY);
      ExpressionSize<LocalRank> i(0);
      ExpressionSize<expr_cast<E>::n_arrays> ind(0);
      Index index = 0;
      int my_rank;
      static const int last = LocalRank-1;
      do {
	i[last] = 0;
	rhs.set_location(i, ind);
	// Innermost loop
	for ( ; i[last] < dimension_<LocalRank-1>::value; ++i[last],
		index += offset_<LocalRank-1>::value) {
	  data_[index] = rhs.next_value(ind);
	}
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }

    template<int LocalRank, bool LocalIsActive, bool EIsActive, class E>
    typename enable_if<LocalIsActive && EIsActive,void>::type
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
      static const int last = LocalRank-1;

      ADEPT_ACTIVE_STACK->check_space(expr_cast<E>::n_active * size());
      do {
	i[last] = 0;
	rhs.set_location(i, ind);
	// Innermost loop
	for ( ; i[last] < dimension_<LocalRank-1>::value; ++i[last],
		index += offset_<LocalRank-1>::value) {
	  data_[index] = rhs.next_value_and_gradient(*ADEPT_ACTIVE_STACK, ind);
	  ADEPT_ACTIVE_STACK->push_lhs(gradient_index()+index); // What if RHS not active?
	}
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }

    template<int LocalRank, bool LocalIsActive, bool EIsActive, class E>
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
	ADEPT_ACTIVE_STACK->push_lhs_range(gradient_ind+index, dimension_<LocalRank-1>::value,
					   offset_<LocalRank-1>::value);
	for ( ; i[last] < dimension_<LocalRank-1>::value; ++i[last],
		index += offset_<LocalRank-1>::value) {
	  data_[index] = rhs.next_value(ind);
	}
	advance_index(index, my_rank, i);
      } while (my_rank >= 0);
    }



    template<bool LocalIsActive, class B, typename C>
    typename enable_if<!LocalIsActive,void>::type
    assign_conditional_inactive_scalar_(const B& bool_expr, C rhs) {
      ExpressionSize<rank> i(0);
      ExpressionSize<expr_cast<B>::n_arrays> bool_ind(0);
      Index index = 0;
      int my_rank;
      static const int last = rank-1;

      do {
	i[last] = 0;
	bool_expr.set_location(i, bool_ind);
	// Innermost loop
	for ( ; i[last] < dimension_<rank-1>::value; ++i[last],
		index += offset_<rank-1>::value) {
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

      ExpressionSize<rank> i(0);
      ExpressionSize<expr_cast<B>::n_arrays> bool_ind(0);
      Index index = 0;
      int my_rank;
      static const int last = rank-1;

      do {
	i[last] = 0;
	bool_expr.set_location(i, bool_ind);
	// Innermost loop
	for ( ; i[last] < dimension_<rank-1>::value; ++i[last],
		index += offset_<rank-1>::value) {
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
      ExpressionSize<rank> i(0);
      ExpressionSize<expr_cast<B>::n_arrays> bool_ind(0);
      ExpressionSize<expr_cast<C>::n_arrays> rhs_ind(0);
      Index index = 0;
      int my_rank;
      static const int last = rank-1;
      bool is_gap = false;

      do {
	i[last] = 0;
	rhs.set_location(i, rhs_ind);
	bool_expr.set_location(i, bool_ind);
	// Innermost loop
	for ( ; i[last] < dimension_<rank-1>::value; ++i[last],
		index += offset_<rank-1>::value) {
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
      ExpressionSize<rank> i(0);
      ExpressionSize<expr_cast<B>::n_arrays> bool_ind(0);
      ExpressionSize<expr_cast<C>::n_arrays> rhs_ind(0);
      Index index = 0;
      int my_rank;
      static const int last = rank-1;
      bool is_gap = false;

      ADEPT_ACTIVE_STACK->check_space(expr_cast<C>::n_active * size());
      do {
	i[last] = 0;
	rhs.set_location(i, rhs_ind);
	bool_expr.set_location(i, bool_ind);
	// Innermost loop
	for ( ; i[last] < dimension_<rank-1>::value; ++i[last],
		index += offset_<rank-1>::value) {
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
    // FixedArray: 8. Data
    // -------------------------------------------------------------------
  protected:
    Type data_[length_]; // Stored on the stack

  }; // End of FixedArray class


  // -------------------------------------------------------------------
  // Helper functions
  // -------------------------------------------------------------------

  // Print array on a stream
  template <typename Type, bool IsActive, Index J0,Index J1,
	    Index J2,Index J3,Index J4,Index J5,Index J6>
  inline
  std::ostream&
  operator<<(std::ostream& os, const FixedArray<Type,IsActive,J0,J1,J2,J3,J4,J5,J6>& A) {
    const Array<fixed_array<J0,J1,J2,J3,J4,J5,J6>::rank,Type,IsActive> B = A; // link to original data
    return B.print(os);
  }


  // Extract inactive part of array, working correctly depending on
  // whether argument is active or inactive
  template <typename Type, Index J0,Index J1,Index J2,Index J3,
	   Index J4,Index J5,Index J6>
  inline
  FixedArray<Type, false,J0,J1,J2,J3,J4,J5,J6>&
  value(FixedArray<Type, false,J0,J1,J2,J3,J4,J5,J6>& expr) {
    return expr;
  }
  template <typename Type, Index J0,Index J1,Index J2, Index J3,
	   Index J4,Index J5,Index J6>
  inline
  FixedArray<Type, false,J0,J1,J2,J3,J4,J5,J6>
  value(FixedArray<Type, true,J0,J1,J2,J3,J4,J5,J6>& expr) {
    return expr.inactive_link();
  }

  // -------------------------------------------------------------------
  // Transpose function
  // -------------------------------------------------------------------

  // Transpose 2D array
  template<typename Type, bool IsActive, Index J0, Index J1>
  inline
  Array<2,Type,IsActive>
  transpose(FixedArray<Type,IsActive,J0,J1>& in) {
    // Create output array initially as link to input array 
    Array<2,Type,IsActive> out(in);
    // Swap dimensions
    return out.in_place_transpose();
  }

  // Extract the gradients from an active FixedArray after the
  // Stack::forward or Stack::reverse functions have been called
  template<typename Type, typename dType, Index J0, Index J1,
	   Index J2, Index J3, Index J4, Index J5, Index J6>
  inline
  void get_gradients(const FixedArray<Type,true,J0,J1,J2,J3,J4,J5,J6>& a,
		     FixedArray<dType,false,J0,J1,J2,J3,J4,J5,J6>& data)
  {
    data = a.get_gradient();
  }

  template <typename T, bool IsActive, typename E, Index J0, 
	    Index J1, Index J2, Index J3, Index J4, Index J5, Index J6>
  internal::Allocator<internal::fixed_array<J0,J1,J2,J3,J4,J5,J6>::rank,
		      FixedArray<T,IsActive,J0,J1,J2,J3,J4,J5,J6> > 
  operator<<(FixedArray<T,IsActive,J0,J1,J2,J3,J4,J5,J6>& array, const E& x) {
    return internal::Allocator<internal::fixed_array<J0,J1,J2,J3,J4,J5,J6>::rank,
      FixedArray<T,IsActive,J0,J1,J2,J3,J4,J5,J6> >(array, x);
  }


} // End namespace adept

#endif
