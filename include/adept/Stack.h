/* Stack.h -- Storage of automatic differentiation information

    Copyright (C) 2012-2014 University of Reading
    Copyright (C) 2015-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   The Stack class is where all the derivative information of an
   algorithm, from which the Jacobian matrix can be constructed, as
   well as tangent-linear and adjoint operations being carried out for
   suitable input derivatives.  When a Stack object is created it puts
   a pointer to itself in a global but thread-local variable that is
   then accessed whenever an active expression is evaluated.

*/

#ifndef AdeptStack_H
#define AdeptStack_H 1

#include <cmath>
#include <iostream>
#include <typeinfo>
#include <utility>
#include <string>
#include <vector>
#include <list>
#include <cstddef>
#include <limits>

#ifdef ADEPT_STACK_STORAGE_STL
#include <valarray>
#endif

#include <adept/base.h>
#include <adept/exception.h>
#include <adept/StackStorageOrig.h>
#include <adept/StackStorageOrigStl.h>
#include <adept/traits.h>

namespace adept {

  // ---------------------------------------------------------------------
  // Access to Stack object via global pointer
  // ---------------------------------------------------------------------

  // Declare a thread-safe and a thread-unsafe global pointer to the
  // current stack
  class Stack;
  extern ADEPT_THREAD_LOCAL Stack* _stack_current_thread;
  extern Stack* _stack_current_thread_unsafe;

  // Define ADEPT_ACTIVE_STACK to be the currently active version
  // regardless of whether we are in thread safe or unsafe mode
#ifdef ADEPT_STACK_THREAD_UNSAFE
#define ADEPT_ACTIVE_STACK adept::_stack_current_thread_unsafe
#else
#define ADEPT_ACTIVE_STACK adept::_stack_current_thread
#endif

  // ---------------------------------------------------------------------
  // Helper classes
  // ---------------------------------------------------------------------

  // Structure holding a fixed-size array of objects (intended for
  // double or float)
  template<int Size, class Type>
  struct Block {
    Block() { zero(); }
    const Type& operator[](uIndex i) const { return data[i]; }
    Type& operator[](uIndex i) { return data[i]; }
    void zero() { for (uIndex i = 0; i < Size; i++) data[i] = 0.0; }
    Type data[Size] ADEPT_SSE2_ALIGNED;
  };

  // Structure for describing a gap in the current list of gradients
  struct Gap {
    Gap(uIndex value) : start(value), end(value) {}
    Gap(uIndex start_, uIndex end_) : start(start_), end(end_) {}
    uIndex start;
    uIndex end;
  };


  // ---------------------------------------------------------------------
  // Definition of Stack class
  // ---------------------------------------------------------------------

  // "Stack" inherits from a class defining the storage of the stack
  // information, which is controlled by preprocessor
  // variables. Member functions not defined here are in Stack.cpp.
  class Stack 
#ifdef ADEPT_STACK_STORAGE_STL
    : public internal::StackStorageOrigStl
#else
    : public internal::StackStorageOrig
#endif
  {
  public:
    // -------------------------------------------------------------------
    // Stack: 1. Static Definitions
    // -------------------------------------------------------------------
    typedef std::list<Gap> GapList;
    typedef std::list<Gap>::iterator GapListIterator;

    // -------------------------------------------------------------------
    // Stack: 2. Constructor and destructor
    // -------------------------------------------------------------------

    // Only one constructor, which is normally called with no
    // arguments, but if "false" is provided as the argument it will
    // construct as normal but not attempt to make itself the active stack
    Stack(bool activate_immediately = true) :
#ifndef ADEPT_STACK_STORAGE_STL
      gradient_(0),
#endif
      most_recent_gap_(gap_list_.end()),
      i_gradient_(0), n_allocated_gradients_(0), max_gradient_(0),
      n_gradients_registered_(0),
      gradients_initialized_(false), 
#ifdef ADEPT_STACK_THREAD_UNSAFE
      is_thread_unsafe_(true),
#else
      is_thread_unsafe_(false),
#endif
      is_recording_(true),
      // Since the library might be compiled with OpenMP support and
      // subsequent programs without, we need to tell the library via
      // the following variable
#ifdef _OPENMP
      have_openmp_(true),
#else
      have_openmp_(false),
#endif
      openmp_manually_disabled_(false)
    { 
      initialize(ADEPT_INITIAL_STACK_LENGTH);
      new_recording();
      if (activate_immediately) {
	activate();
      }
    }
  
    // Destructor
    ~Stack();

    // -------------------------------------------------------------------
    // Stack: 3. Public member functions
    // -------------------------------------------------------------------

    // This function is no longer available
    void start(uIndex n = ADEPT_INITIAL_STACK_LENGTH) {
      throw feature_not_available("The Stack::start() function has been removed since Adept version 1.0: see the documentation about how to use Stack::new_recording()"
				  ADEPT_EXCEPTION_LOCATION);
    }

    // After a sequence of operation pushes, we may append these to
    // the previous statement by calling this function.
    // gradient_index is the index of the gradient on the LHS of the
    // statement: if this does not match the LHS of the previous
    // statement then this is an error and "false" will be returned. A
    // "true" return value indicates success.
    bool update_lhs(const uIndex& gradient_index) {
      if (statement_[n_statements_-1].index != gradient_index) {
	return false;
      }
      else {
	statement_[n_statements_-1].end_plus_one = n_operations_;
	return true;
      }
    }

    // When an aReal object is created it is registered on the stack
    // and keeps a copy of its location, which is returned from this
    // function
    uIndex register_gradient() {
      uIndex return_val;
#ifdef ADEPT_RECORDING_PAUSABLE
      if (is_recording()) {
#endif
	n_gradients_registered_++;
	if (gap_list_.empty()) {
	  // Add to end of gradient vector
	  i_gradient_++;
	  if (i_gradient_ > max_gradient_) {
	    max_gradient_ = i_gradient_;
	  }
	  return_val = i_gradient_-1;
	}
	else {
	  // Insert in a gap
	  Gap& first_gap = gap_list_.front();
	  return_val = first_gap.start;
	  first_gap.start++;
	  if (first_gap.start > first_gap.end) {
	    // Gap has closed: remove it from the list, after checking
	    // if it had been stored as the gap that had most recently
	    // grown
	    if (most_recent_gap_ == gap_list_.begin()) {
	      most_recent_gap_ = gap_list_.end();
	    }
	    gap_list_.pop_front();
	  }
	}
#ifdef ADEPT_RECORDING_PAUSABLE
      }
      else {
	return_val = 0;
      }
#endif
      return return_val;
    }

    // Register n gradients and return the index of the first one
    uIndex register_gradients(const uIndex& n)  {
      uIndex return_val;
#ifdef ADEPT_RECORDING_PAUSABLE
      if (is_recording()) {
#endif
	return_val = do_register_gradients(n);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
      else {
	return_val = 0;
      }
#endif
      return return_val;
    }


    // When an aReal object is destroyed it is unregistered from the
    // stack. If it is at the top of the stack then the stack pointer
    // can be decremented so that the space can be used by another
    // object. A gap can appear in the stack if an active object (or
    // array of active objects) is returned from a function, so we
    // need to keep track of a "gap" appearing in the stack. If the
    // user uses new and delete without any regard for this "last-in
    // first-out" preference then the number of gradients that are
    // allocated in the reverse pass may be larger than needed.
    void unregister_gradient(const uIndex& gradient_index) {
      n_gradients_registered_--;
      if (gradient_index+1 == i_gradient_) {
        // Gradient to be unregistered is at the top of the stack
        i_gradient_--;
	if (!gap_list_.empty()) {
	  Gap& last_gap = gap_list_.back();
	  if (i_gradient_ == last_gap.end+1) {
	    // We have unregistered the elements between the "gap" of
	    // unregistered element and the top of the stack, so can
	    // set the variables indicating the presence of the gap to
	    // zero
	    i_gradient_ = last_gap.start;
	    GapListIterator it = gap_list_.end();
	    it--;
	    if (most_recent_gap_ == it) {
	      most_recent_gap_ = gap_list_.end();
	    }
	    gap_list_.pop_back();
	  }
	}
      }
      else { // Gradient to be unregistered not at top of stack.
	// In the less common situation that the gradient is not at
	// the top of the stack, the task of unregistering is a bit
	// more involved, so we carry it out in a non-inline function
	// to avoid code bloat
	unregister_gradient_not_top(gradient_index);
      }
    }

    // Unregister n gradients starting at gradient_index
    void unregister_gradients(const uIndex& gradient_index,
			      const uIndex& n);


  protected:
    uIndex do_register_gradients(const uIndex& n);

    // Unregister a gradient that is not at the top of the stack
    void unregister_gradient_not_top(const uIndex& gradient_index);
  public:

    // Set the gradients in the list with indices between start and
    // end_plus_one-1 to the values pointed to by "gradient"
    template <typename MyReal>
    typename internal::enable_if<internal::is_floating_point<MyReal>::value,
		       void>::type
    set_gradients(uIndex start, uIndex end_plus_one,
		  const MyReal* gradient) {
      // Need to initialize the gradient list if not already done
      if (!gradients_are_initialized()) {
	initialize_gradients();
      }
      if (end_plus_one > max_gradient_) {
	throw gradient_out_of_range();
      }
      for (uIndex i = start, j = 0; i < end_plus_one; i++, j++) {
	gradient_[i] = gradient[j];
      }
    }

    // Get the gradients in the list with indices between start and
    // end_plus_one-1 and put them in the location pointed to by
    // "gradient"
    template <typename MyReal>
    typename internal::enable_if<internal::is_floating_point<MyReal>::value,
		       void>::type
    get_gradients(uIndex start, uIndex end_plus_one,
		  MyReal* gradient) const {
      if (!gradients_are_initialized()) {
	throw gradients_not_initialized();
      }
      if (end_plus_one > max_gradient_) {
	throw gradient_out_of_range();
      }
      for (uIndex i = start, j = 0; i < end_plus_one; i++, j++) {
	gradient[j] = gradient_[i];
      }
    }
    template <typename MyReal>
    typename internal::enable_if<internal::is_floating_point<MyReal>::value,
		       void>::type
    get_gradients(uIndex start, uIndex end_plus_one,
		  MyReal* gradient, Index src_stride, Index target_stride) const {
      if (!gradients_are_initialized()) {
	throw gradients_not_initialized();
      }
      if (end_plus_one > max_gradient_) {
	throw gradient_out_of_range();
      }
      for (uIndex i = start, j = 0; i < end_plus_one; i+=src_stride, j+=target_stride) {
	gradient[j] = gradient_[i];
      }
    }

    // Run the tangent-linear algorithm on the gradient list; normally
    // this call is preceded calls to set_gradient to load input
    // gradients and followed by calls to get_gradient to extract
    // gradients
    void compute_tangent_linear();
    void forward() { return compute_tangent_linear(); }

    // Run the adjoint algorithm on the gradient list; normally this
    // call is preceded calls to set_gradient to load input gradient
    // and followed by calls to get_gradient to extract gradient
    void compute_adjoint();
    void reverse() { return compute_adjoint(); }

    // Return the number of independent and dependent variables that
    // have been identified
    uIndex n_independent() const { return independent_index_.size(); }
    uIndex n_dependent() const { return dependent_index_.size(); }

    // Compute the Jacobian matrix; note that jacobian_out must be
    // allocated to be of size m*n, where m is the number of dependent
    // variables and n is the number of independents. The independents
    // and dependents must have already been identified with the
    // functions "independent" and "dependent", otherwise this
    // function will throw a
    // "dependents_or_independents_not_identified" exception. In the
    // resulting matrix, the "m" dimension of the matrix varies
    // fastest. This is implemented by calling one of jacobian_forward
    // and jacobian_reverse, whichever would be faster.
    void jacobian(Real* jacobian_out);

    // Compute the Jacobian matrix, but explicitly specify whether
    // this is done with repeated forward or reverse passes.
    void jacobian_forward(Real* jacobian_out);
    void jacobian_reverse(Real* jacobian_out);

    // Return maximum number of OpenMP threads to be used in Jacobian
    // calculation
    int max_jacobian_threads() const;

    // Set the maximum number of threads to be used in Jacobian
    // calculations, if possible. A value of 1 indicates that OpenMP
    // will not be used, while a value of 0 indicates that the number
    // will match the number of available processors. Returns the
    // maximum that will be used, which will be 1 if the Adept library
    // was compiled without OpenMP support. Note that a value of 1
    // will disable the use of OpenMP with Adept, so Adept will then
    // use no OpenMP directives or function calls. Note that if in
    // your program you use OpenMP with each thread performing
    // automatic differentiaion with its own independent Adept stack,
    // then typically only one OpenMP thread is available for each
    // Jacobian calculation, regardless of whether you call this
    // function.
    int set_max_jacobian_threads(int n);

    // In order to compute the jacobian we need to first declare which
    // active variables are independent (x) and which are dependent
    // (y). First, the following two functions declare an individual
    // active variable and an array of active variables to be
    // independent. Note that we use templates here because aReal has
    // not been defined.
    template <class A>
    void independent(const A& x) {
      //      independent_index_.push_back(x.gradient_index());
      x.push_gradient_indices(independent_index_);
    }
    template <class A>
    void independent(const A* x, uIndex n) {
      for (uIndex i = 0; i < n; i++) {
	//	independent_index_.push_back(x[i].gradient_index());
	x[i].push_gradient_indices(independent_index_);
      }
    }

    // Likewise, delcare the dependent variables
    template <class A>
    void dependent(const A& x) {
      //      dependent_index_.push_back(x.gradient_index());
      x.push_gradient_indices(dependent_index_);
    }
    template <class A>
    void dependent(const A* x, uIndex n) {
      for (uIndex i = 0; i < n; i++) {
	//	dependent_index_.push_back(x[i].gradient_index());
	x[i].push_gradient_indices(dependent_index_);
      }
    }

    // Print various bits of information about the Stack to the
    // specified stream (or standard output if not specified). The
    // same behaviour can be obtained by "<<"-ing the Stack to a
    // stream.
    void print_status(std::ostream& os = std::cout) const;

    // Print each derivative statement to the specified stream (or
    // standard output if not specified)
    void print_statements(std::ostream& os = std::cout) const;

    // Print the current gradient list to the specified stream (or
    // standard output if not specified); returns true on success or
    // false if no gradients have been initialized
    bool print_gradients(std::ostream& os = std::cout) const;

    // Print a list of the gaps in the gradient list
    void print_gaps(std::ostream& os = std::cout) const;

    // Clear the gradient list enabling a new adjoint or
    // tangent-linear computation to be performed with the same
    // recording
    void clear_gradients() {
      gradients_initialized_ = false;
    }

    // Clear the list of independent variables, in order that a
    // different Jacobian can be computed from the same recording
    void clear_independents() {
      independent_index_.clear();
    }

    // Clear the list of dependent variables, in order that a
    // different Jacobian can be computed from the same recording
    void clear_dependents() {
      dependent_index_.clear();
    }

    // Function now removed
    void clear() {
      throw feature_not_available("The Stack::clear() function has been removed since Adept version 1.0: see the documentation about how to use Stack::new_recording()"
				  ADEPT_EXCEPTION_LOCATION);
    }
    // Function now removed
    void clear_statements() {
      throw feature_not_available("The Stack::clear_statements() function has been removed since Adept version 1.0: see the documentation about how to use Stack::new_recording()"
				  ADEPT_EXCEPTION_LOCATION);
    }

    // Make this stack "active" by copying its "this" pointer to a
    // global variable; this makes it the stack that aReal objects
    // subsequently interact with when being created and participating
    // in mathematical expressions
    void activate();

    // This stack will stop being the one that aReal objects refer
    // to; this may be useful if the thread needs to use another stack
    // object for the next algorithm
    void deactivate() {
      if (is_active()) {
	ADEPT_ACTIVE_STACK = 0;
      }
    }

    // Return true if the Stack is "active", false otherwise
    bool is_active() const {
      return (ADEPT_ACTIVE_STACK == this);
    }

    // Clear the contents of the various lists ready for a new
    // recording
    void new_recording() {
      clear_stack(); // Defined in the storage class
      clear_independents();
      clear_dependents();
      clear_gradients();

      // i_gradient_ is the maximum index of all currently constructed
      // aReal objects and max_gradient_ is the maximum index of all
      // that were used in a recording.  Thus when deleting the
      // recording we need to set max_gradient_ to i_gradient_ or a
      // little more.
      max_gradient_ = i_gradient_+1;
      // Insert a null statement
      //    std::cerr << "Inserting a null statement; when is this needed?\n";
      push_lhs(-1);
    }

    // Are gradients to be computed?  The default is "true", but if
    // ADEPT_RECORDING_PAUSABLE is defined then this may
    // be false
    bool is_recording() const {
#ifdef ADEPT_RECORDING_PAUSABLE
      return is_recording_;
#else
      return true;
#endif
    }

    // Stop recording gradient information, enabling a piece of active
    // code to be run without the stack information being stored. This
    // only works if ADEPT_RECORDING_PAUSABLE has been defined.
    bool pause_recording() {
#ifdef ADEPT_RECORDING_PAUSABLE
      is_recording_ = false;
      return true;
#else
      return false;
#endif
    }
    // Continue recording gradient information after a previous
    // pause_recording() call. This only works if
    // ADEPT_RECORDING_PAUSABLE has been defined.
    bool continue_recording() {
#ifdef ADEPT_RECORDING_PAUSABLE
      is_recording_ = true;
      return true;
#else
      return false;
#endif
    }

    // For modular codes, some modules may have an existing Jacobian
    // code and possibly be unsuitable for automatic differentiation
    // using Adept (e.g. because they are written in Fortran).  In
    // this case, we can use the following two functions to "wrap" the
    // non-Adept code. These are actually normally called by functions
    // of the same name in the Active, ActiveReference and
    // ActiveConstReference classes.
    void add_derivative_dependence(uIndex lhs_index, uIndex rhs_index,
				   Real multiplier) {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	// Check there is space in the operation stack for 1 entry
	ADEPT_ACTIVE_STACK->check_space(1);
#endif
	if (multiplier != 0.0) {
	  push_rhs(multiplier, rhs_index);
	}
	push_lhs(lhs_index);
#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
    }

    void append_derivative_dependence(uIndex lhs_index, uIndex rhs_index,
				      Real multiplier) {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (ADEPT_ACTIVE_STACK->is_recording()) {
#endif
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	// Check there is space in the operation stack for 1 entry
	ADEPT_ACTIVE_STACK->check_space(1);
#endif
	if (multiplier != 0.0) {
	  push_rhs(multiplier, rhs_index);
	}
	if (!update_lhs(lhs_index)) {
	  throw wrong_gradient("Wrong gradient: append_derivative_dependence called on a different active number from the most recent add_derivative_dependence call"
			       ADEPT_EXCEPTION_LOCATION);
	}
#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
    }

    // To enable the automatic differentiation of matrix
    // multiplication, this function performs a similar role to
    // aReal::add_derivative_dependence.  We add a derivative
    // expression of the form d[lhs_index] =
    // sum(multiplier[i*multiplier_stride]*d[rhs_index+i*index_stride]),
    // where the summation is from i = 0 to n-1. Multiple calls to
    // this function may be carried out but must be followed by
    // push_lhs(lhs_index) to specify the left-hand-side of the
    // statement.
    template <typename Type>
    void push_derivative_dependence(uIndex rhs_index,
				    const Type* multiplier,
				    int n = 1,
				    int index_stride = 1,
				    int multiplier_stride = 1) {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (is_recording()) {
#endif
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	// Check there is space in the operation stack for n entries
	check_space(n);
#endif
	for (int i = 0; i < n; i++, rhs_index += index_stride, 
	       multiplier += multiplier_stride) {
	  push_rhs(*multiplier, rhs_index);
	}
#ifdef ADEPT_RECORDING_PAUSABLE
      }
#endif
    }

    // Have the gradients been initialized?
    bool gradients_are_initialized() const { return gradients_initialized_; }

    // Return the number of statements, operations, and how much
    // memory has been allocated for each
    uIndex n_statements() const { return n_statements_; }
    uIndex n_allocated_statements() const { return n_allocated_statements_; }
    uIndex n_operations() const { return n_operations_; }
    uIndex n_allocated_operations() const { return n_allocated_operations_; }

    // Return the size of the two dimensions of a Jacobian matrix
    uIndex n_independents() const { return independent_index_.size(); }
    uIndex n_dependents() const { return dependent_index_.size(); }

    // Return the maximum number of gradients required to perform
    // adjoint calculation
    uIndex max_gradients() const { return max_gradient_; }

    // Return the index to the current gradient
    uIndex i_gradient() const { return i_gradient_; }

    // Return the number of gradients memory has been allocated for
    uIndex n_allocated_gradients() const { return n_allocated_gradients_; }

    // Return the number of bytes used
    std::size_t memory() const {
      std::size_t mem = n_statements()*sizeof(uIndex)*2
	+ n_operations()*(sizeof(Real)+sizeof(uIndex));
      if (gradients_are_initialized()) {
	mem += max_gradients()*sizeof(Real);
      }
      return mem;
    }

    // Return the number of gradients currently registered
    uIndex n_gradients_registered() const { return n_gradients_registered_; }

    // Return the fraction of multipliers equal to the specified
    // number (usually -1, 0 or 1)
    Real fraction_multipliers_equal_to(Real val) {
      uIndex sum = 0;
      for (uIndex i = 0; i < n_operations_; i++) {
	if (multiplier_[i] == val) {
	  sum++;
	}
      }
      return static_cast<Real>(sum)/static_cast<Real>(n_operations_);
    }


    bool is_thread_unsafe() const { return is_thread_unsafe_; }

    const GapList& gap_list() const { return gap_list_; }

    // Memory to store statements and operations can be preallocated,
    // offering modest performance advantage if you define
    // ADEPT_MANUAL_MEMORY_ALLOCATION and know the maximum number of
    // statements and operations you will need
    void preallocate_statements(uIndex n) {
      if (n_statements_+n+1 >= n_allocated_statements_) {
	grow_statement_stack(n);
      }
    }
    void preallocate_operations(uIndex n) {
      if (n_allocated_operations_ < n_operations_+n+1) {
	grow_operation_stack(n);
      }      
    }

    // -------------------------------------------------------------------
    // Stack: 4. Protected member functions
    // -------------------------------------------------------------------
  protected:
    // Initialize the vector of gradients ready for the adjoint
    // calculation
    void initialize_gradients();

    // Set to zero the gradients required by a Jacobian calculation
    /*
    void zero_gradient_multipass() {
      for (std::size_t i = 0; i < gradient_multipass_.size(); i++) {
	gradient_multipass_[i].zero();
      }
    }
    */

    // OpenMP versions of the forward and reverse Jacobian functions,
    // which are called from the jacobian_forward and jacobian_reverse
    // if OpenMP is enabled
    void jacobian_forward_openmp(Real* jacobian_out) const;
    void jacobian_reverse_openmp(Real* jacobian_out) const;

    // The core code for computing Jacobians, used in both OpenMP and
    // non-OpenMP versions
    void jacobian_forward_kernel(Real* __restrict gradient_multipass_b) const;
    void jacobian_forward_kernel_packet(Real* __restrict gradient_multipass_b) const;
    void jacobian_forward_kernel_extra(Real* __restrict gradient_multipass_b, uIndex) const;
    void jacobian_reverse_kernel(Real* __restrict gradient_multipass_b) const;
    void jacobian_reverse_kernel_packet(Real* __restrict gradient_multipass_b) const;
    void jacobian_reverse_kernel_extra(Real* __restrict gradient_multipass_b, uIndex) const;

    // -------------------------------------------------------------------
    // Stack: 5. Data
    // -------------------------------------------------------------------
  protected:

#ifdef ADEPT_STACK_STORAGE_STL
    // Data are stored using standard template library containers
    //    std::valarray<Real> gradient_;
    std::vector<Real> gradient_;
#else
    // Data are stored as dynamically allocated arrays
    Real* __restrict gradient_;
#endif
    // For Jacobians we process multiple rows/columns at once so need
    // what is essentially a 2D array
    //    std::vector<Block<ADEPT_MULTIPASS_SIZE,Real> > gradient_multipass_;
    // uIndexs of the independent and dependent variables
    std::vector<uIndex> independent_index_;
    std::vector<uIndex> dependent_index_;
    // Keep a record of gaps in the gradient array to ensure that gaps
    // are filled
    GapList gap_list_;
    //    Gap* most_recent_gap_;
    GapListIterator most_recent_gap_;

    uIndex i_gradient_;             // Current number of gradients
    uIndex n_allocated_gradients_;  // Number of allocated gradients
    uIndex max_gradient_;           // Max number of gradients to store
    uIndex n_gradients_registered_; // Number of gradients registered
    bool gradients_initialized_;    // Have the gradients been
				    // initialized?
    bool is_thread_unsafe_;
    bool is_recording_;
    bool have_openmp_;              // true if this header file
				    // compiled with -fopenmp
    bool openmp_manually_disabled_; // true if user called
				    // set_max_jacobian_threads(1)
  }; // End of Stack class


  // -------------------------------------------------------------------
  // Helper functions
  // -------------------------------------------------------------------

  // Sending a Stack object to a stream reports information about the
  // stack
  inline
  std::ostream& operator<<(std::ostream& os, const adept::Stack& stack) {
    stack.print_status(os);
    return os;
  }

  // Memory to store statements and operations can be preallocated,
  // offering modest performance advantage if you define
  // ADEPT_MANUAL_MEMORY_ALLOCATION and know the maximum number of
  // statements and operations you will need. This version is useful
  // in functions that don't have visible access to the currently
  // active Adept stack. 
  inline
  void preallocate_statements(uIndex n) {
    ADEPT_ACTIVE_STACK->preallocate_statements(n);
  }
  inline
  void preallocate_operations(uIndex n) {
    ADEPT_ACTIVE_STACK->preallocate_operations(n);
  }

  // Returns a pointer to the currently active stack (or 0 if there is none)
  inline
  Stack* active_stack() { return ADEPT_ACTIVE_STACK; }

  // Return whether the active stack is stored in a global variable
  // (thread unsafe) rather than a thread-local global variable
  // (thread safe)
#ifdef ADEPT_STACK_THREAD_UNSAFE
  inline bool is_thread_unsafe() { return true; }
#else
  inline bool is_thread_unsafe() { return false; }
#endif 

  // Subsequent code should use adept::active_stack rather than this
  // preprocessor macro
  //#undef ADEPT_ACTIVE_STACK

} // End of namespace adept


#endif
