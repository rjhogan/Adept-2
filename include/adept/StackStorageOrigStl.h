/* StackStorageOrigStl.h -- Original storage of stacks using STL containers

    Copyright (C) 2014-2015 University of Reading

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

   The Stack class inherits from a class providing the storage (and
   interface to the storage) for the derivative statements that are
   accumulated during the execution of an algorithm.  The derivative
   statements are held in two stacks described by Hogan (2014): the
   "statement stack" and the "operation stack".

   This file provides one of the original storage engine, which used
   std::vector to hold the two stacks. Note that these stacks are
   contiguous in memory, which is not ideal for very large algorithms.

*/

#ifndef AdeptStackStorageOrigStl_H
#define AdeptStackStorageOrigStl_H 1

#include <adept/base.h>
#include <adept/exception.h>
#include <adept/Statement.h>

namespace adept {
  namespace internal {

    class StackStorageOrigStl {
    public:
      // Constructor
      StackStorageOrigStl() :
	n_statements_(0), n_allocated_statements_(0),
	n_operations_(0), n_allocated_operations_(0) { }
      
      // Destructor (does nothing)
      ~StackStorageOrigStl() { };

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
	  multiplier_.push_back(multiplier);
	  index_.push_back(gradient_index);
	  n_operations_++;
	
#ifdef ADEPT_TRACK_NON_FINITE_GRADIENTS
	  if (!std::isfinite(multiplier) || std::isinf(multiplier)) {
	    throw non_finite_gradient();
	  }
#endif
	
#ifdef ADEPT_REMOVE_NULL_STATEMENTS
	}
#endif
      }


      // Push a statement on to the stack: this is done after a
      // sequence of operation pushes; gradient_index is the index of
      // the gradient on the LHS of the expression, while the
      // "end_plus_one" element is simply the current length of the
      // operation list
      void push_lhs(const uIndex& gradient_index) {
	statement_.push_back(Statement(gradient_index, n_operations_));
	n_statements_++;
      }

      // Push n left-hand-sides of differential expressions on to the
      // stack with no corresponding right-hand-side, appropriate if
      // an array of active variables contiguous in memory (or
      // separated by a fixed stride) has been assigned to inactive
      // numbers.
      void push_lhs_range(const uIndex& first, const uIndex& n, 
			  const uIndex& stride = 1) {
	uIndex last_plus_1 = first+n*stride;
	for (uIndex i = first; i < last_plus_1; i += stride) {
	  statement_.push_back(Statement(i, n_operations_));
	}
	n_statements_ += n;
      }

      // Check whether the operation stack contains enough space for n
      // new operations; for STL containers this does nothing
      void check_space(const uIndex& n) { }
      template<uIndex n> void check_space_static() { }

    protected:
      // Called by new_recording()
      void clear_stack() { 
	// If we use STL containers then the clear() function sets their
	// size to zero but leaves the memory allocated
	statement_.clear();
	multiplier_.clear();
	index_.clear();
	// Set the recording indices to zero
	n_operations_ = 0;
	n_statements_ = 0;
      }

      // This function is called by the constructor to initialize
      // memory, which can be grown subsequently
      void initialize(uIndex n) {
	statement_.reserve(n);
	multiplier_.reserve(n);
	index_.reserve(n);
      }

      // Grow the capacity of the operation or statement stacks to
      // hold a minimum of "min" elements. If min=0 then the stacks
      // are doubled in size.
      void grow_operation_stack(uIndex min = 0);
      void grow_statement_stack(uIndex min = 0);

    protected:
      // Data are stored using standard template library containers

      // The "statement stack" is held as a single array
      std::vector<Statement> statement_;
      // The "operation stack" is held as two arrays
      std::vector<Real> multiplier_;
      std::vector<uIndex> index_;

      uIndex n_statements_;           // Number of statements
      uIndex n_allocated_statements_; // Space allocated for statements
      uIndex n_operations_;           // Number of operations
      uIndex n_allocated_operations_; // Space allocated for statements
    };

  } // End namespace internal
} // End namespace adept

#endif
