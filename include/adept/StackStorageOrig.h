/* StackStorageOrig.h -- Original method to store statement & operation stacks

    Copyright (C) 2014-2015 University of Reading

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

   The Stack class inherits from a class providing the storage (and
   interface to the storage) for the derivative statements that are
   accumulated during the execution of an algorithm.  The derivative
   statements are held in two stacks described by Hogan (2014): the
   "statement stack" and the "operation stack".

   This file provides the original storage engine: dynamically
   allocated arrays with the two stacks resulting from an entire
   algorithm being contiguous in memory.  This is not ideal for very
   large algorithms.

*/

#ifndef AdeptStackStorageOrig_H
#define AdeptStackStorageOrig_H 1

#include <adept/base.h>
#include <adept/exception.h>
#include <adept/Statement.h>

namespace adept {
  namespace internal {

    class StackStorageOrig {
    public:
      // Constructor
      StackStorageOrig() : 
	statement_(0), multiplier_(0), index_(0),
	n_statements_(0), n_allocated_statements_(0),
	n_operations_(0), n_allocated_operations_(0) { }
      
      // Destructor
      ~StackStorageOrig();

      // Push an operation (i.e. a multiplier-gradient pair) on to the
      // stack.  We assume here that check_space() as been called before
      // so there is enough space to hold these elements.
      void push_rhs(const Real& multiplier, const uIndex& gradient_index) {
#ifdef ADEPT_REMOVE_NULL_STATEMENTS
	// If multiplier==0 then the resulting statement would have no
	// effect so we can speed up the subsequent adjoint/jacobian
	// calculations (at the expense of making this critical part
	// of the code slower)
	if (multiplier != 0.0) {
#endif
	  multiplier_[n_operations_] = multiplier;
	  index_[n_operations_++] = gradient_index;
	
#ifdef ADEPT_TRACK_NON_FINITE_GRADIENTS
	  if (!std::isfinite(multiplier) || std::isinf(multiplier)) {
	    throw non_finite_gradient();
	  }
#endif
	
#ifdef ADEPT_REMOVE_NULL_STATEMENTS
	}
#endif
      }

      // Push the gradient indices of a vectorized operation on to the
      // stack.  We assume here that check_space() as been called
      // before so there is enough space to hold these elements. The
      // multipliers will be added later.
      template <Index Num, Index Stride>
      void push_rhs_indices(const uIndex& gradient_index) {
	for (Index i = 0; i < Num; ++i) {
	  index_[n_operations_+i*Stride] = gradient_index+i;
	}
	++n_operations_;
      }

      // Push a statement on to the stack: this is done after a
      // sequence of operation pushes; gradient_index is the index of
      // the gradient on the LHS of the expression, while the
      // "end_plus_one" element is simply the current length of the
      // operation list
      void push_lhs(const uIndex& gradient_index) {
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	if (n_statements_ >= n_allocated_statements_) {
	  grow_statement_stack();
	}
#endif
	statement_[n_statements_].index = gradient_index;
	statement_[n_statements_++].end_plus_one = n_operations_;
      }

      // Push n left-hand-sides of differential expressions on to the
      // stack with no corresponding right-hand-side, appropriate if
      // an array of active variables contiguous in memory (or
      // separated by a fixed stride) has been assigned to inactive
      // numbers. Note that the second and third arguments must not be
      // references, since they may be compile-time constants for
      // FixedArray objects.
      void push_lhs_range(const uIndex& first, uIndex n, uIndex stride = 1) {
	uIndex last_plus_1 = first+n*stride;
#ifndef ADEPT_MANUAL_MEMORY_ALLOCATION
	if (n_statements_+n > n_allocated_statements_) {
	  grow_statement_stack(n);
	}
#endif
	for (uIndex i = first; i < last_plus_1; i += stride) {
	  statement_[n_statements_].index = i;
	  statement_[n_statements_++].end_plus_one = n_operations_;
	}
      }

      // Check whether the operation stack contains enough space for n
      // new operations; if not, grow it
      void check_space(uIndex n) {
	if (n_allocated_operations_ < n_operations_+n+1) {
	  grow_operation_stack(n);
	}
      }
      template<uIndex n>
      void check_space_static() {
	check_space(n);
      }

    protected:
      // Called by new_recording()
      void clear_stack() { 
	// Set the recording indices to zero
	n_operations_ = 0;
	n_statements_ = 0;
      }

      // This function is called by the constructor to initialize
      // memory, which can be grown subsequently
      void initialize(uIndex n) {
	multiplier_ = new Real[n];
	index_ = new uIndex[n];
	n_allocated_operations_ = n;
	statement_ = new Statement[n];
	n_allocated_statements_ = n;
      }

      // Grow the capacity of the operation or statement stacks to
      // hold a minimum of "min" elements. If min=0 then the stacks
      // are doubled in size.
      void grow_operation_stack(uIndex min = 0);
      void grow_statement_stack(uIndex min = 0);

    protected:
      // Data are stored as dynamically allocated arrays

      // The "statement stack" is held as a single array
      Statement* __restrict statement_ ;
      // The "operation stack" is held as two arrays
      Real*      __restrict multiplier_;
      uIndex*    __restrict index_;

      uIndex n_statements_;           // Number of statements
      uIndex n_allocated_statements_; // Space allocated for statements
      uIndex n_operations_;           // Number of operations
      uIndex n_allocated_operations_; // Space allocated for statements
    };

  } // End namespace internal
} // End namespace adept

#endif
