/* StackStorageOrig.cpp -- Original storage of stacks using STL containers

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

#include <cstring>

#include <adept/StackStorageOrig.h>

namespace adept {
  namespace internal {

    StackStorageOrig::~StackStorageOrig() {
      if (statement_) {
	delete[] statement_;
      }
      if (multiplier_) {
	delete[] multiplier_;
      }
      if (index_) {
	delete[] index_;
      }
    }


    // Double the size of the operation stack, or grow it even more if
    // the requested minimum number of extra entries (min) is greater
    // than this would allow
    void
    StackStorageOrig::grow_operation_stack(uIndex min)
    {
      uIndex new_size = 2*n_allocated_operations_;
      if (min > 0 && new_size < n_allocated_operations_+min) {
	new_size += min;
      }
      Real* new_multiplier = new Real[new_size];
      uIndex* new_index = new uIndex[new_size];
      
      std::memcpy(new_multiplier, multiplier_, n_operations_*sizeof(Real));
      std::memcpy(new_index, index_, n_operations_*sizeof(uIndex));
      
      delete[] multiplier_;
      delete[] index_;
      
      multiplier_ = new_multiplier;
      index_ = new_index;
      
      n_allocated_operations_ = new_size;
    }
    
    // ... likewise for the statement stack
    void
    StackStorageOrig::grow_statement_stack(uIndex min)
    {
      uIndex new_size = 2*n_allocated_statements_;
      if (min > 0 && new_size < n_allocated_statements_+min) {
	new_size += min;
      }
      Statement* new_statement = new Statement[new_size];
      std::memcpy(new_statement, statement_,
		  n_statements_*sizeof(Statement));
      delete[] statement_;
      
      statement_ = new_statement;
      
      n_allocated_statements_ = new_size;
    }

  }
}
