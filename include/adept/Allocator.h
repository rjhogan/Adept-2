/* Allocator.h -- Allocates elements to arrays

    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptAllocator_H
#define AdeptAllocator_H 1

#include <adept/Array.h>
//#include <adept/SpecialMatrix.h>

namespace adept {
  namespace internal {
   
    template <int Rank, class A>
    class Allocator {
    public:
      // Create an allocator object and copy the first object in it
      template <typename F>
      Allocator(A& array, const F& first_arg) 
	: array_(array), size_(array.dimensions()),
	  //	  filled_size_(0), 
	  obj_size_(0), coords_(0),
	  scalar_size_(1) {
	*this << first_arg;
      }

      // Copy a scalar into the array
      template <typename T>
      typename enable_if<is_not_expression<T>::value,Allocator&>::type
      operator<<(const T& x) {
	if (coords_[Rank-1] >= size_[Rank-1]) {
	  // We have reached the end of the array: move to next row
	  complete_row<Rank>();
	  // All dimensions of this object are of length 1
	  obj_size_.set_all(1);
	}
	else if (coords_[Rank-1] == 0) {
	  // At the beginning of a row: set the size of the template
	  // object to that of a scalar
	  obj_size_ = scalar_size_;	  
	}
	else if (obj_size_ != scalar_size_) {
	  // The template object size is not the same as a scalar,
	  // indicating that dissimilar objects have been concatenated
	  // in a row
	  throw index_out_of_bounds("Scalar added to array with \"<<\" when previous objects on row were not scalar" 
				    ADEPT_EXCEPTION_LOCATION);
	}
	// Add the scalar to the array and increment the final index
	array_.get_lvalue(coords_) = x;
	++coords_[Rank-1];
	return *this;
      }


      // Copy an expression into the array
      template <typename T, class E>
      typename enable_if<(E::rank <= Rank), Allocator&>::type
      operator<<(const Expression<T,E>& x) {
	// Evaluate expression and store in an Array of the same rank
	// (if Expression is already an Array then this will make a
	// shallow copy). Ought to check for aliasing.
	const Array<E::rank,T,E::is_active> xx(x.cast());
	ExpressionSize<Rank-1> leading_dim;
	//	leading_dim.copy_dissimilar(xx.dimensions());
	partial_copy(xx.dimensions(), leading_dim);

	if (coords_[Rank-1] >= size_[Rank-1]) {
	  // We have reached the end of the array: move to next row
	  complete_row<Rank>();
	}
	if (coords_[Rank-1] == 0) {
	  partial_copy(xx.dimensions(), obj_size_);
	}
	else if (obj_size_ != leading_dim) {
	  // The template object size is not the same as the current
	  // array, indicating that dissimilar objects have been
	  // concatenated in a row
	  throw index_out_of_bounds("Expression added to array with \"<<\" does not match size of previous objects on row"
				    ADEPT_EXCEPTION_LOCATION);
	}
	// Add the object to the array and increment the final index
	ExpressionSize<Rank> i_lhs(coords_);
	ExpressionSize<E::rank> i_rhs(0);
	int rank;
	do {
	  array_.get_lvalue(i_lhs) = xx.get_rvalue(i_rhs);
	  advance_index(rank, i_lhs, i_rhs, xx.dimensions());
	}
	while (rank >= 0);
	
	coords_[Rank-1] += xx.dimension(E::rank-1);
	return *this;
      }

      template <int RhsRank>
      void advance_index(int& rank, ExpressionSize<Rank>& i_lhs, 
			 ExpressionSize<RhsRank>& i_rhs,
			 const ExpressionSize<RhsRank>& size) const {
	rank = RhsRank;
	while (--rank >= 0) {
	  if (++i_rhs[rank] >= size[rank]) {
	    i_rhs[rank] = 0;
	    i_lhs[rank+(Rank-RhsRank)] -= (size[rank]-1);
	    }
	  else {
	    ++i_lhs[rank+(Rank-RhsRank)];
	    break;
	  }
	}
      }
      
      // Comma operator does the same as "<<" operator
      template <typename T>
      typename enable_if<is_not_expression<T>::value,Allocator&>::type
      operator,(const T& x) {
	return *this << x;
      }
	
    protected:
      // A vector should never complete a row as this indicates it has
      // been overfilled
      template <int MyRank>
      typename enable_if<(MyRank <= 1), void>::type
      complete_row() {
	throw index_out_of_bounds("Row overflow in filling Vector with \"<<\""
				  ADEPT_EXCEPTION_LOCATION);
      }

      // Multi-dimensional arrays: move to next row, checking which
      // dimensions have been filled
      template <int MyRank>
      typename enable_if<(MyRank > 1), void>::type
      complete_row() {
	int next_dim = Rank-2;
	while (next_dim >= 0) {
	  if (coords_[next_dim]+obj_size_[next_dim] < size_[next_dim]) {
	    //	    filled_size_[next_dim] += obj_size_[next_dim];
	    coords_[next_dim] += obj_size_[next_dim];
	    for (int i = next_dim+1; i < Rank; ++i) {
	      coords_[i] = 0;
	    }
	    break;
	  }
	  --next_dim;
	}
	if (next_dim < 0) {
	  throw index_out_of_bounds("Dimension overflow in filling array with \"<<\""
				    ADEPT_EXCEPTION_LOCATION);
	}
	obj_size_.set_all(0);
      }

      template <int MyRank>
      typename enable_if<(MyRank > 1), void>::type
      partial_copy(const ExpressionSize<MyRank>& from,
		   ExpressionSize<Rank-1>& to) const {
	for (int i = 0; i < Rank-MyRank; ++i) {
	  to[i] = 1;
	}
	for (int i = Rank-MyRank; i < Rank-1; ++i) {
	  to[i] = from[i+(MyRank-Rank)];
	}
      }

      template <int MyRank>
      typename enable_if<(MyRank <= 1), void>::type
      partial_copy(const ExpressionSize<MyRank>& from,
		   ExpressionSize<Rank-1>& to) const {
	to.set_all(1);
      }


    protected:
      A& array_;
      const ExpressionSize<Rank> size_;
      //      ExpressionSize<Rank-1> filled_size_;
      ExpressionSize<Rank-1> obj_size_;
      ExpressionSize<Rank> coords_;
      const ExpressionSize<Rank-1> scalar_size_;
    };
    
  }

  // Allow object to be filled with "A << 1, 2, 3";
  template <int Rank, typename T, bool IsActive, typename E>
  internal::Allocator<Rank,Array<Rank,T,IsActive> > 
  operator<<(Array<Rank,T,IsActive>& array, const E& x) {
    if (array.empty()) {
      throw empty_array("Attempt to fill empty array with \"<<\""
			ADEPT_EXCEPTION_LOCATION);
    }
    return internal::Allocator<Rank,Array<Rank,T,IsActive> >(array, x);
  }

}


#endif
