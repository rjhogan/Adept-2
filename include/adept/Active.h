/* Active.h -- Active scalar type for automatic differentiation

    Copyright (C) 2012-2014 University of Reading
    Copyright (C) 2015-2018 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

  
   The Active class describes a scalar variable that can participate
   in expressions to be differentiated. It is a generalization of the
   aReal (or adouble) class in Adept 1.0, which was always double
   precision; Active<T> takes a template argument T that is any
   floating-point type.

*/

#ifndef AdeptActive_H
#define AdeptActive_H

#include <iostream>
#include <vector>

#include <adept/Expression.h>
#include <adept/exception.h>
#include <adept/ExpressionSize.h>

namespace adept {

  using namespace internal;


  // ---------------------------------------------------------------------
  // Definition of Active class
  // ---------------------------------------------------------------------
  template <typename Type>
  class Active : public Expression<Type, Active<Type> > {
    // CONTENTS
    // 1. Preamble
    // 2. Constructors
    // 3. Operators
    // 4. Public member functions that don't modify the object
    // 5. Public member functions that modify the object
    // 6. Protected member functions
    // 7. Data

  public:
    // -------------------------------------------------------------------
    // 1. Preamble
    // -------------------------------------------------------------------

    // Static definitions to enable the properties of this type of
    // expression to be discerned at compile time
    static const bool is_active = true;
    static const bool is_lvalue = true;
    static const int  rank      = 0;
    static const int  n_active  = 1 + is_complex<Type>::value;
    static const int  n_arrays  = 0;
    static const int  n_scratch = 0;

    // -------------------------------------------------------------------
    // 2. Constructors
    // -------------------------------------------------------------------

    // Constructor registers the new Active object with the currently
    // active stack.  Note that this object is not explicitly
    // initialized with a particular number; the user should not
    // assume that it is set to zero but should later assign it to a
    // particular value. Otherwise in the reverse pass the
    // corresponding gradient will not be set to zero.
    Active()
      : val_(0.0), gradient_index_(ADEPT_ACTIVE_STACK->register_gradient()) { }
    
    // Constructor with a passive argument; this constructor is
    // invoked with either of the following:
    //   aReal x = 1.0;
    //   aReal x(1.0);
    template <typename PType>
    Active(const PType& rhs,
	   typename enable_if<is_not_expression<PType>::value>::type* dummy = 0)
      : val_(rhs), gradient_index_(ADEPT_ACTIVE_STACK->register_gradient())
    {
      // By pushing this to the statement stack without pushing
      // anything on to the operation stack we ensure that in the
      // reverse pass the gradient of this object will be set to zero
      // after it has been manipulated. This is important because the
      // gradient entry might be reused.
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
	ADEPT_ACTIVE_STACK->push_lhs(gradient_index_);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
    }

    // Constructor taking an element from an active array: the value
    // and gradient_index of the element are provided
    template <typename PType>
    Active(const PType& rhs, Index gradient_index)
      : val_(rhs), gradient_index_(ADEPT_ACTIVE_STACK->register_gradient())
    {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
	ADEPT_ACTIVE_STACK->push_rhs(1.0,gradient_index);
	ADEPT_ACTIVE_STACK->push_lhs(gradient_index_);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
    }
   
    // Constructor with an active argument

    // Normal copy construction: register the new object then treat
    // this as an assignment.  We need two versions because if we
    // don't provide the first then the compiler will provide it and
    // not use the second if Type==AType
    Active(const Active<Type>& rhs) 
      : val_(0.0), gradient_index_(ADEPT_ACTIVE_STACK->register_gradient())
    {
      *this = rhs;
    }
    template <typename AType>
    Active(const Active<AType>& rhs) 
      : val_(0.0), gradient_index_(ADEPT_ACTIVE_STACK->register_gradient())
    {
      *this = rhs;
    }

    // Construction with an expression.  This is primarily used so
    // that if we define a function func(aReal a), it will also accept
    // active expressions by implicitly converting them to an aReal.
    template<typename AType, class E>
    //          explicit
    Active(const Expression<AType, E>& rhs,
	   typename enable_if<E::rank==0
			      && E::is_active>::type* dummy = 0)
      : gradient_index_(ADEPT_ACTIVE_STACK->register_gradient())
    {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
      	// Check there is enough space in the operation stack
	ADEPT_ACTIVE_STACK->check_space_static<E::n_active>();
#endif
	// Get the value and push the gradients on to the operation
	// stack, thereby storing the right-hand-side of the statement
	val_ = rhs.scalar_value_and_gradient(*ADEPT_ACTIVE_STACK);
	// Push the gradient offet of this object on to the statement
	// stack, thereby storing the left-hand-side of the statement
	ADEPT_ACTIVE_STACK->push_lhs(gradient_index_);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
      else {
	val_ = rhs.scalar_value();
      }
#endif
    }
	   
    // Destructor simply unregisters the object from the stack,
    // freeing up the gradient index for another
    ~Active() {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif

	ADEPT_ACTIVE_STACK->unregister_gradient(gradient_index_);

#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
    }


    // -------------------------------------------------------------------
    // 3. Operators
    // -------------------------------------------------------------------
	   
    // Assignment operator with an inactive variable on the rhs
    template <typename PType>
    typename enable_if<is_not_expression<PType>::value,
		       Active&>::type
    operator=(const PType& rhs) {
      val_ = rhs;
      // Pushing the gradient index on to the statement stack with no
      // corresponding operations ensures that the gradient will be
      // set to zero in the reverse pass when it is finished with
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
	ADEPT_ACTIVE_STACK->push_lhs(gradient_index_);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
      return *this;
    }

    // Assignment operator with an active variable on the rhs: first a
    // non-template version because otherwise compiler will generate
    // its own
    Active& operator=(const Active& rhs) {
      // Check there is space in the operation stack for one more
      // entry
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	ADEPT_ACTIVE_STACK->check_space(1);
#endif
	// Same as construction with an expression (defined above)
	val_ = rhs.scalar_value_and_gradient(*ADEPT_ACTIVE_STACK);
	ADEPT_ACTIVE_STACK->push_lhs(gradient_index_);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
      else {
	val_ = rhs.scalar_value();
      }
#endif
      return *this; 
    }

    // Assignment operator with an active variable on the rhs
    template <class AType>
    Active& operator=(const Active<AType>& rhs) {
      // Check there is space in the operation stack for one more
      // entry
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	ADEPT_ACTIVE_STACK->check_space(1);
#endif
	// Same as construction with an expression (defined above)
	val_ = rhs.scalar_value_and_gradient(*ADEPT_ACTIVE_STACK);
	ADEPT_ACTIVE_STACK->push_lhs(gradient_index_);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
      else {
	val_ = rhs.scalar_value();
      }
#endif
      return *this;
    }
    
    // Assignment operator with an expression on the rhs: very similar
    // to construction with an expression (defined above)
    template <typename AType, class E>
    typename enable_if<E::is_active && E::rank==0, Active&>::type
    operator=(const Expression<AType, E>& rhs) {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	ADEPT_ACTIVE_STACK->check_space_static<E::n_active>();
#endif
	val_ = rhs.scalar_value_and_gradient(*ADEPT_ACTIVE_STACK);
	ADEPT_ACTIVE_STACK->push_lhs(gradient_index_);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
      else {
	val_ = rhs.scalar_value();
      }
#endif
      return *this;
    }
  
    // All the compound assignment operators are unpacked, i.e. a+=b
    // becomes a=a+b; first for an Expression on the rhs
    template<typename AType, class E>
    typename enable_if<E::rank==0, Active&>::type
    operator+=(const Expression<AType,E>& rhs) {
      return *this = (*this + rhs);
    }
    template<typename AType, class E>
    typename enable_if<E::rank==0, Active&>::type
    operator-=(const Expression<AType,E>& rhs) {
      return *this = (*this - rhs);
    }
    template<typename AType, class E>
    typename enable_if<E::rank==0, Active&>::type
    operator*=(const Expression<AType,E>& rhs) {
      return *this = (*this * rhs);
    }
    template<typename AType, class E>
    typename enable_if<E::rank==0, Active&>::type
    operator/=(const Expression<AType,E>& rhs) {
      return *this = (*this / rhs);
    }

    // And likewise for a passive scalar on the rhs
    template <typename PType>
    typename enable_if<is_not_expression<PType>::value, Active&>::type
    operator+=(const PType& rhs) {
      val_ += rhs;
      return *this;
    }
    template <typename PType>
    typename enable_if<is_not_expression<PType>::value, Active&>::type
    operator-=(const PType& rhs) {
      val_ -= rhs;
      return *this;
    }
    template <typename PType>
    typename enable_if<is_not_expression<PType>::value, Active&>::type
    operator*=(const PType& rhs) {
      return *this = (*this * rhs);
    }
    template <typename PType>
    typename enable_if<is_not_expression<PType>::value, Active&>::type
    operator/=(const PType& rhs) {
      return *this = (*this / rhs);
    }

      
    // -------------------------------------------------------------------
    // 4. Public member functions that don't modify the object
    // -------------------------------------------------------------------

    // Get the underlying passive value of this object
    Type value() const {
      return val_; 
    }

    // Get the index of the gradient information for this object
    const Index& gradient_index() const { return gradient_index_; }

    // If an expression leads to calc_gradient being called on an
    // active object, we push the multiplier and the gradient index on
    // to the operation stack (or 1.0 if no multiplier is specified
    template <int Rank>
    void calc_gradient(Stack& stack, const ExpressionSize<Rank>&) const {
      stack.push_rhs(1.0, gradient_index_);
    }

    template <int Rank, typename MyType>
    void calc_gradient(Stack& stack, const MyType& multiplier, 
		       const ExpressionSize<Rank>&) const {
      stack.push_rhs(multiplier, gradient_index_);
    }

    // Set the value of the gradient, for initializing an adjoint;
    // note that the value of the gradient is not held in the active
    // object but rather held by the stack
    template <typename MyType>
    void set_gradient(const MyType& gradient) const {
      return ADEPT_ACTIVE_STACK->set_gradients(gradient_index_,
					       gradient_index_+1, 
					       &gradient);
    }

    // Get the value of the gradient, for extracting the adjoint after
    // calling reverse() on the stack
    template <typename MyType>
    void get_gradient(MyType& gradient) const {
      return ADEPT_ACTIVE_STACK->get_gradients(gradient_index_,
					       gradient_index_+1, &gradient);
    }
    Type get_gradient() const {
      Type gradient = 0;
      ADEPT_ACTIVE_STACK->get_gradients(gradient_index_,
					gradient_index_+1, &gradient);
      return gradient;
    }
 

    // For modular codes, some modules may have an existing
    // Jacobian code and possibly be unsuitable for automatic
    // differentiation using Adept (e.g. because they are written in
    // Fortran).  In this case, we can use the following two functions
    // to "wrap" the non-Adept code.

    // Suppose the non-adept code uses the double values from n aReal
    // objects pointed to by "x" to produce a single double value
    // "y_val" (to be assigned to an aReal object "y"), plus a pointer
    // to an array of forward derivatives "dy_dx".  Firstly you should
    // assign the value using simply "y = y_val;", then call
    // "y.add_derivative_dependence(x, dy_dx, n);" to specify how y
    // depends on x. A fourth argument "multiplier_stride" may be used
    // to stride the indexing to the derivatives, in case they are
    // part of a matrix that is oriented in a different sense.
    template <typename MyReal>
    typename internal::enable_if<internal::is_floating_point<MyReal>::value,
		       void>::type
    add_derivative_dependence(const Active* rhs,
			      const MyReal* multiplier,
			      int n, 
			      int multiplier_stride = 1) const {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	// Check there is space in the operation stack for n entries
	ADEPT_ACTIVE_STACK->check_space(n);
#endif
	for (int i = 0; i < n; i++) {
	  Real mult = multiplier[i*multiplier_stride];
	  if (mult != 0.0) {
	    // For each non-zero multiplier, add a pseudo-operation to
	    // the operation stack
	    ADEPT_ACTIVE_STACK->push_rhs(mult,
					 rhs[i].gradient_index());
	  }
	}
	ADEPT_ACTIVE_STACK->push_lhs(gradient_index_);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
    }

    // Suppose the non-Adept code uses double values from n aReal
    // objects pointed to by "x" and m aReal objects pointed to by "z"
    // to produce a single double value, plus pointers to arrays of
    // forward derivatives "dy_dx" and "dy_dz".  Firstly, as above,
    // you should assign the value using simply "y = y_val;", then
    // call "y.add_derivative_dependence(x, dy_dx, n);" to specify how
    // y depends on x.  To specify also how y depends on z, call
    // "y.append_derivative_dependence(z, dy_dz, n);".
    template <typename MyReal>
    typename internal::enable_if<internal::is_floating_point<MyReal>::value,
		       void>::type
    append_derivative_dependence(const Active* rhs,
				 const MyReal* multiplier,
				 int n,
				 int multiplier_stride = 1) const {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	// Check there is space in the operation stack for n entries
	ADEPT_ACTIVE_STACK->check_space(n);
#endif
	for (int i = 0; i < n; ++i) {
	  Real mult = multiplier[i*multiplier_stride];
	  if (mult != 0.0) {
	    // For each non-zero multiplier, add a pseudo-operation to
	    // the operation stack
	    ADEPT_ACTIVE_STACK->push_rhs(mult,
					 rhs[i].gradient_index());
	  }
	}
	if (!(ADEPT_ACTIVE_STACK->update_lhs(gradient_index_))) {
	  throw wrong_gradient("Wrong gradient: append_derivative_dependence called on a different aReal object from the most recent add_derivative_dependence call"
			       ADEPT_EXCEPTION_LOCATION);
	}
#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
    }

    // For only one independent variable on the rhs, these two
    // functions are convenient as they don't involve pointers
    template <class T>
    void add_derivative_dependence(const T& rhs, Real multiplier) const {
      ADEPT_ACTIVE_STACK->add_derivative_dependence(gradient_index_,
						    rhs.gradient_index(),
						    multiplier);
    }
    template <class T>
    void append_derivative_dependence(const T& rhs, Real multiplier) const {
      ADEPT_ACTIVE_STACK->append_derivative_dependence(gradient_index_,
						       rhs.gradient_index(),
						       multiplier);
    }
 
    // -------------------------------------------------------------------
    // 4.1. Public member functions used by other expressions
    // -------------------------------------------------------------------
    bool get_dimensions_(ExpressionSize<0>& dim) const { return true; }

    std::string expression_string_() const {
      std::stringstream s;
      s << "Active(" << val_ << ")";
      return s.str();
    }

    bool is_aliased_(const Type* mem1, const Type* mem2) const { 
      return false;
    }

    Type value_with_len_(const Index& j, const Index& len) const
    { return val_; }

    template <int MyArrayNum, int NArrays>
    void advance_location_(ExpressionSize<NArrays>& loc) const { } 

    template <int MyArrayNum, int NArrays>
    Type value_at_location_(const ExpressionSize<NArrays>& loc) const
    { return val_; }
    
    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
    Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				ScratchVector<NScratch>& scratch) const
    { return val_; }

    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
    Type value_stored_(const ExpressionSize<NArrays>& loc,
		     const ScratchVector<NScratch>& scratch) const
    { return val_; }

    template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
    void calc_gradient_(Stack& stack, 
			const ExpressionSize<NArrays>& loc,
			const ScratchVector<NScratch>& scratch) const {
      stack.push_rhs(1.0, gradient_index_);
    }

    template <int MyArrayNum, int MyScratchNum, 
	      int NArrays, int NScratch, typename MyType>
    void calc_gradient_(Stack& stack, 
			const ExpressionSize<NArrays>& loc,
			const ScratchVector<NScratch>& scratch,
			const MyType& multiplier) const {
      stack.push_rhs(multiplier, gradient_index_);
    }

    template <int MyArrayNum, int Rank, int NArrays>
    void set_location_(const ExpressionSize<Rank>& i, 
		       ExpressionSize<NArrays>& index) const {}


    // The Stack::independent(x) and Stack::dependent(y) functions add
    // the gradient_index of objects x and y to std::vector<uIndex>
    // objects in Stack. Since x and y may be scalars or arrays, this
    // is best done by delegating to the Active or Array classes.
    template <typename IndexType>
    void push_gradient_indices(std::vector<IndexType>& vec) const {
      vec.push_back(gradient_index_);
    }

    // -------------------------------------------------------------------
    // 5. Public member functions that modify the object
    // -------------------------------------------------------------------

    // Set the value 
    template <typename MyType>
    void set_value(const MyType& x) { val_ = x; }

    // For use in creating active references, to get a non-const
    // reference to the underlying passive data
    Type& lvalue() { return val_; }

    
    // -------------------------------------------------------------------
    // 6. Protected member functions
    // -------------------------------------------------------------------
  protected:
    
    // -------------------------------------------------------------------
    // 7. Data
    // -------------------------------------------------------------------
  private:
    Type val_;                     // The numerical value
    Index gradient_index_;         // Index to where the corresponding
				   // gradient will be held during the
				   // adjoint calculation
  }; // End of definition of Active


  // ---------------------------------------------------------------------
  // Helper function for Active class
  // ---------------------------------------------------------------------

  // A way of setting the initial values of an array of n aReal
  // objects without the expense of placing them on the stack
  template<typename Type>
  inline
  void set_values(Active<Type>* a, Index n, const Type* data)
  {
    for (Index i = 0; i < n; i++) {
      a[i].set_value(data[i]);
    }
  }

  // Extract the values of an array of n aReal objects
  template<typename Type>
  inline
  void get_values(const Active<Type>* a, Index n, Type* data)
  {
    for (Index i = 0; i < n; i++) {
      data[i] = a[i].value();
    }
  }
  
  // Set the initial gradients of an array of n aReal objects; this
  // should be done after the algorithm has called and before the
  // Stack::forward or Stack::reverse functions are called
  template<typename Type>
  inline
  void set_gradients(Active<Type>* a, Index n, const Type* data)
  {
    for (Index i = 0; i < n; i++) {
      a[i].set_gradient(data[i]);
    }
  }
  
  // Extract the gradients from an array of aReal objects after the
  // Stack::forward or Stack::reverse functions have been called
  template<typename Type>
  inline
  void get_gradients(const Active<Type>* a, Index n, Type* data)
  {
    for (Index i = 0; i < n; i++) {
      a[i].get_gradient(data[i]);
    }
  }

  // Print an active scalar to a stream
  template<typename Type>
  inline
  std::ostream&
  operator<<(std::ostream& os, const Active<Type>& v)
  {
    os << v.value();
    return os;
  }

  // Print an active scalar expression to a stream
  template <typename Type, class E>
  inline
  typename enable_if<E::rank == 0 && E::is_active, std::ostream&>::type
  operator<<(std::ostream& os, const Expression<Type,E>& expr) {
    os << expr.scalar_value();
    return os;
  }

  namespace internal {
    // ---------------------------------------------------------------------
    // Definition of active_scalar
    // ---------------------------------------------------------------------
    
    // Return the active scalar version of Type if it is active,
    // otherwise just return Type
    
    template <class Type, bool IsActive> struct active_scalar {
      typedef Type type;
    };

    template <class Type> struct active_scalar<Type, true> {
      typedef Active<Type> type;
    };

  }

} // End namespace adept

#endif
