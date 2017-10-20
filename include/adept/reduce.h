/* reduce.h -- "Reduce" functions such as find, all, sum etc.

    Copyright (C) 2015-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   This file implements a number of array functions whose return
   values are reduced in either rank or size compared to their
   arguments.

   The first is the "find" function that takes a rank-1 bool
   Expression, and returns an IntVector of indices to the "true"
   values.  This is modelled on Matlab's "find" function.

   A number of further reduce functions are implemented using the same
   calling style as the equivalent Fortran-90 functions.  They fall
   into two types:
     1. sum, mean, product, minval, maxval, norm2
     2. all, any
   The first take active or inactive Expression arguments of real or
   (sometimes) integer type, while the second only take inactive
   Expressions of bool type.  If called with one Expression argument
   of any rank, a single value is returned containing the result of
   the reduce operation on all the elements of the Expression.  If a
   second integer argument is provided then the operation is carried
   out along that dimension and an Expression of rank one less than
   the first argument is returned. These functions are implemented by
   delegating to a generic "Reduce" function that uses policy classes
   to implement the elemental operations.

*/

#ifndef AdeptReduce_H
#define AdeptReduce_H

#include <algorithm>

#include <adept/Array.h>
#include <adept/Active.h>
#include <adept/SpecialMatrix.h>
#include <adept/array_shortcuts.h>

namespace adept {

  // -------------------------------------------------------------------
  // Section 1. "find"
  // -------------------------------------------------------------------
  // This function takes a rank-1 bool Expression, and returns an
  // IntVector of indices to the "true" values.
  template <class E>
  inline
  typename internal::enable_if<E::rank == 1,IntVector>::type
  find(const Expression<bool, E>& rhs)
  {
    ExpressionSize<1> length;
    // Check the argument of the function is a valid expression
    if (!rhs.get_dimensions(length)) {
      std::string str = "Array size mismatch in "
	+ rhs.expression_string() + ".";
      throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
    }
    // Length of the rank-1 expression
    Index& len = length[0];
    // Allocate a return vector of the same length as the expression
    // in case all values are true
    IntVector ans(len);
    // Keep track of the actual number of true values
    Index true_len = 0;
    // Get location of first value in expression
    ExpressionSize<1> coords(0);
    ExpressionSize<E::n_arrays> loc;
    rhs.set_location(coords, loc);
    // Loop over all values in the expression
    for (int i = 0; i < len; i++) {
      if (rhs.next_value(loc)) {
	ans(true_len++) = i;
      }
    }
    if (true_len == 0) {
      // No values are "true": return an empty vector
      return IntVector();
    }
    else if (true_len < len) {
      // Some values are "true": return the part of the "ans" vector
      // that contains indices to these values.  Note that the
      // following subsetting operation links to the original data
      // rather than copying it.
      return ans(range(0,true_len-1));
    }
    else {
      // All values are "true": return the entire vector.
      return ans;
    }
  }

  namespace internal {

    // -------------------------------------------------------------------
    // 1. Policy classes to enable the generic "reduce" function
    // -------------------------------------------------------------------

    // Sum enables the "sum" function that sums its arguments.
    template <typename T>
    struct Sum {
      // What is the type of the running total?
      typedef T total_type;
      // Do we need to do anything to the final summed value(s)?
      static const bool finish_needed = false;
      // Do we need to do anything to the final summed value(s) in the
      // case that we are doing automatic differentiation?
      static const bool active_finish_needed = true;
      // Used by "expression_string()"
      const char* name() { return "sum"; }
      // Start the accumulation with zero
      T first_value() { return 0; }
      // Accumulation consists of incrementing "total" by the value on
      // the right hand side; note that the arguments are either of
      // type T or type Packet<T>
      template <typename E>
      void accumulate(E& total, const E& rhs) { total += rhs; }
      // When the reduce operation is vectorized, packets of data are
      // accumulated, requiring the ability to horizontally accumulate
      // each element of the packet
      T accumulate_packet(const Packet<T>& ptotal) {
	return hsum(ptotal);
      }
      // In the case of active arguments, the next_value_and_gradient
      // function pushes the right hand side onto the operation stack,
      // but does not push the "total" object onto the statement
      // stack.  This is done right and the end of the summation
      // operations.
      template <class E, int NArrays>
      void accumulate_active(Active<T>& total, const E& rhs, 
			     ExpressionSize<NArrays>& loc) {
	total.lvalue() += rhs.next_value_and_gradient(*ADEPT_ACTIVE_STACK, loc);
      }
      // No need to do anything to the final value
      template <class X>
      void finish(X& total, const Index& n) { }
      // In the active case, the final action is to complete the
      // storage of the differential statement by pushing the left
      // hand side onto the statement stack.
      void finish_active(Active<T>& total, const Index& n) { 
	ADEPT_ACTIVE_STACK->push_lhs(total.gradient_index());
      }
    };

    // Mean enables the "mean" function - the same as "sum" but
    // dividing the final result by the number of elements averaged.
    template <typename T>
    struct Mean {
      typedef T total_type;
      static const bool finish_needed = true;
      static const bool active_finish_needed = true;
      const char* name() { return "mean"; }
      T first_value() { return 0; }
      template <typename E>
      void accumulate(E& total, const E& rhs) { total += rhs; }
      T accumulate_packet(const Packet<T>& ptotal) {
	return hsum(ptotal);
      }
      template <class E, int NArrays>
      void accumulate_active(Active<T>& total, const E& rhs, 
			     ExpressionSize<NArrays>& loc) {
	total.lvalue() += rhs.next_value_and_gradient(*ADEPT_ACTIVE_STACK, loc);
      }
      template <class X>
      // Divide by the total number of elements
      void finish(X& total, const Index& n) { total /= n; }
      void finish_active(Active<T>& total, const Index& n) { 
	ADEPT_ACTIVE_STACK->push_lhs(total.gradient_index());
	total /= n;
      }
    };

    // Product enables the "product" function that multiplies all its
    // arguments together.
    template <typename T>
    struct Product {
      typedef T total_type;
      static const bool finish_needed = false;
      static const bool active_finish_needed = false;
      const char* name() { return "product"; }
      T first_value() { return 1; }
      template <typename E>
      void accumulate(E& total, const E& rhs) { total *= rhs; }
      T accumulate_packet(const Packet<T>& ptotal) {
	return hprod(ptotal);
      }
      template <class E, int NArrays>
      void accumulate_active(Active<T>& total, const E& rhs, 
			     ExpressionSize<NArrays>& loc) {
	// Differentiate t = t*x -> dt = t*dx + x*dt.  First compute
	// x, while passing t as the last argument so that t*dx is put
	// on the operation stack.
	T xval = rhs.next_value_and_gradient_special(*ADEPT_ACTIVE_STACK, loc,
						     total.value());
	// Now treat x as inactive and Active<T> will do the rest
	total *= xval;	
      }
      template <class X>
      void finish(X& total, const Index& n) { }
      void finish_active(Active<T>& total, const Index& n) { }
    };

    // MaxVal enables the "maxval" function that returns the maximum value
    template <typename T>
    struct MaxVal {
      typedef T total_type;
      static const bool finish_needed = false;
      static const bool active_finish_needed = false;
      const char* name() { return "maxval"; }
      // Initiate the total with the minimum possible value
      T first_value() { return std::numeric_limits<T>::min(); }
#ifdef ADEPT_CXX11_FEATURES
      void accumulate(T& total, const T& rhs) { total = std::fmax(total,rhs); }
#else
      void accumulate(T& total, const T& rhs) { total = std::max(total,rhs); }
#endif
      void accumulate(Packet<T>& total, const Packet<T>& rhs) { total = fmax(total,rhs); }
      T accumulate_packet(const Packet<T>& ptotal) {
	return hmax(ptotal);
      }
      template <class E, int NArrays>
      void accumulate_active(Active<T>& total, const E& rhs, 
			     ExpressionSize<NArrays>& loc) {
	// The following is not optimal since if a maximum is found
	// then the value is evaluated twice. Better would be to
	// locate the maximum in the entire array, then do the active
	// stuff just for that element.
	if (rhs.value_at_location(loc) > total.value()) {
	  // The right hand side puts itself on the operation stack,
	  // while operator= puts the left hand side on the statement
	  // stack.
	  total = rhs.next_value_and_gradient(*ADEPT_ACTIVE_STACK, loc);
	}
      }
      template <class X>
      void finish(X& total, const Index& n) { }
      void finish_active(Active<T>& total, const Index& n) { }
    };

    // MinVal enables the "minval" function that returns the minimum value
    template <typename T>
    struct MinVal {
      typedef T total_type;
      static const bool finish_needed = false;
      static const bool active_finish_needed = false;
      const char* name() { return "minval"; }
      T first_value() { return std::numeric_limits<T>::max(); }
#ifdef ADEPT_CXX11_FEATURES
      void accumulate(T& total, const T& rhs) { total = std::fmin(total,rhs); }
#else
      void accumulate(T& total, const T& rhs) { total = std::min(total,rhs); }
#endif
      void accumulate(Packet<T>& total, const Packet<T>& rhs) { total = fmin(total,rhs); }
      T accumulate_packet(const Packet<T>& ptotal) {
	return hmin(ptotal);
      }
      template <class E, int NArrays>
      void accumulate_active(Active<T>& total, const E& rhs, 
			     ExpressionSize<NArrays>& loc) {
	// The following is not optimal since if a maximum is found
	// then the value is evaluated twice
	if (rhs.value_at_location(loc) < total.value()) {
	  // The right hand side puts itself on the operation stack,
	  // while operator= puts the left hand side on the statement
	  // stack.
	  total = rhs.next_value_and_gradient(*ADEPT_ACTIVE_STACK, loc);
	}
      }
      template <class X>
      void finish(X& total, const Index& n) { }
      void finish_active(Active<T>& total, const Index& n) { }
    };
  
    // Norm2 enables the "norm2" function that returns the L-2 norm of
    // its arguments, equal to sqrt(sum(rhs*rhs))
    template <typename T>
    struct Norm2 {
      typedef T total_type;
      static const bool finish_needed = true;
      static const bool active_finish_needed = true;
      const char* name() { return "norm2"; }
      T first_value() { return 0; }
      template <typename E>
      void accumulate(E& total, const E& rhs) { total += rhs*rhs; }
      T accumulate_packet(const Packet<T>& ptotal) {
	return hsum(ptotal);
      }
      template <class E, int NArrays>
      void accumulate_active(Active<T>& total, const E& rhs, 
			     ExpressionSize<NArrays>& loc) {
	// Differentiate t += x*x -> dt += 2*x*dx.  Use the "special"
	// version of the following function, where multiplier*x*dx is
	// put on the operation stack.
	T xval = rhs.next_value_and_gradient_special(*ADEPT_ACTIVE_STACK, loc,
						   2.0);
	// Now do a purely inactive operation since we will put
	// "total" on the statement stack only right at the end
	total.lvalue() += xval*xval;
      }
      template <class X>
      void finish(X& total, const Index& n) {
	using std::sqrt;
	total = noalias(sqrt(total));
      }
      void finish_active(Active<T>& total, const Index& n) {
	using std::sqrt;
	// The operation stack now contains the derivatives of all the
	// squared elements on the right hand side.  Here we complete
	// the differential statement by pushing the left hand side
	// onto the statement stack.
	ADEPT_ACTIVE_STACK->push_lhs(total.gradient_index());
	// Since total is active it will do the right thing in the
	// final operation.
	total = noalias(sqrt(total));
      }
    };

    // All enables the "all" function that returns "true" only if all
    // the bool elements of the right hand side are true.  It would be
    // faster if it could quit after finding the first "false".
    struct All {
      typedef bool total_type;
      static const bool finish_needed = false;
      const char* name() { return "all"; }
      bool first_value() { return true; }
      void accumulate(bool& total, const bool& rhs)
      { total = total && rhs; }
      template <class X>
      void finish(X& total, const Index& n) { }
    };

    // Any enables the "any" function that returns "true" if any of
    // the bool elements of the right hand side are true. It would be
    // faster if it could quite after finding the first "true".
    struct Any {
      typedef bool total_type;
      static const bool finish_needed = false;
      const char* name() { return "any"; }
      bool first_value() { return false; }
      void accumulate(bool& total, const bool& rhs)
      { total = total || rhs; }
      template <class X>
      void finish(X& total, const Index& n) { }
    };

    // Count enables the "count" function that returns the number of
    // "true" elements in a bool array.
    struct Count {
      typedef Index total_type;
      static const bool finish_needed = false;
      const char* name() { return "count"; }
      Index first_value() { return 0; }
      void accumulate(Index& total, const bool& rhs)
      { total += static_cast<Index>(rhs); } // true=1, false=0
      template <class X>
      void finish(X& total, const Index& n) { }
    };

    // -------------------------------------------------------------------
    // Section 2. Various versions of the "reduce" function
    // -------------------------------------------------------------------

    // Reduce an entire inactive array, unvectorized
    template <class Func, typename Type, class E>
    inline
    typename enable_if<!(E::is_vectorizable
			 &&Packet<Type>::is_vectorized
			 &&is_same<Type,typename Func::total_type>::value),
		       typename Func::total_type>::type
    reduce_inactive(const Expression<Type, E>& rhs) {
      typename Func::total_type total;
      Func f;
      ExpressionSize<E::rank> dims;
      // Check right hand side is a valid expression
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (dims[0] == 0) {
	// Return zero if any of these functions applied to an empty
	// array
	total = 0;
      }
      else {
	total = f.first_value();
	Index n = dims.size();
	ExpressionSize<E::rank> i(0);
	ExpressionSize<E::n_arrays> loc(0);
	int my_rank;
	static const int last = E::rank-1;
	do {
	  i[last] = 0;
	  rhs.set_location(i, loc);
	  // Innermost loop
	  for ( ; i[last] < dims[last]; ++i[last]) {
	    f.accumulate(total, rhs.next_value(loc));
	  }
	  my_rank = E::rank-1;
	  while (--my_rank >= 0) {
	    if (++i[my_rank] >= dims[my_rank]) {
	      i[my_rank] = 0;
	    }
	    else {
	      break;
	    }
	  }
	} while (my_rank >= 0);
	f.finish(total, n);
      }
      return total;
    }

    // Reduce an entire inactive array, vectorized
    template <class Func, typename Type, class E>
    inline
    typename enable_if<E::is_vectorizable
                       &&Packet<Type>::is_vectorized
                       &&is_same<Type,typename Func::total_type>::value,
		       typename Func::total_type>::type
    reduce_inactive(const Expression<Type, E>& rhs) {
      typename Func::total_type total;
      Func f;
      ExpressionSize<E::rank> dims;
      // Check right hand side is a valid expression
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (dims[0] == 0) {
	// Return zero if any of these functions applied to an empty
	// array
	total = 0;
      }
      else if (dims[E::rank-1] > Packet<Type>::size*2
	       && rhs.all_arrays_contiguous()) {
	// Vectorization is possible
	Packet<Type> ptotal(f.first_value());
	Index n = dims.size();
	ExpressionSize<E::rank> i(0);
	ExpressionSize<E::n_arrays> loc(0);
	int my_rank;
	static const int last = E::rank-1;
	int iendvec;
	int istartvec = rhs.alignment_offset();
	total = f.first_value();
	if (istartvec < 0) {
	  istartvec = iendvec = 0;
	}
	else {
	  iendvec = (dims[last]-istartvec);
	  iendvec -= (iendvec % Packet<Type>::size);
	  iendvec += istartvec;
	}
	do {
	  i[last] = 0;
	  rhs.set_location(i, loc);
	  // Innermost loop
	  for ( ; i[last] < istartvec; ++i[last]) {
	    f.accumulate(total, rhs.next_value_contiguous(loc));
	  }
	  for ( ; i[last] < iendvec; i[last] += Packet<Type>::size) {
	    f.accumulate(ptotal, rhs.next_packet(loc));
	  }
	  for ( ; i[last] < dims[last]; ++i[last]) {
	    f.accumulate(total, rhs.next_value_contiguous(loc));
	  }
	  my_rank = E::rank-1;
	  while (--my_rank >= 0) {
	    if (++i[my_rank] >= dims[my_rank]) {
	      i[my_rank] = 0;
	    }
	    else {
	      break;
	    }
	  }
	} while (my_rank >= 0);
	f.accumulate(total, f.accumulate_packet(ptotal));
	f.finish(total, n);
      }
      else {
	// Back to unvectorized version
	total = f.first_value();
	Index n = dims.size();
	ExpressionSize<E::rank> i(0);
	ExpressionSize<E::n_arrays> loc(0);
	int my_rank;
	static const int last = E::rank-1;
	do {
	  i[last] = 0;
	  rhs.set_location(i, loc);
	  // Innermost loop
	  for ( ; i[last] < dims[last]; ++i[last]) {
	    f.accumulate(total, rhs.next_value(loc));
	  }
	  my_rank = E::rank-1;
	  while (--my_rank >= 0) {
	    if (++i[my_rank] >= dims[my_rank]) {
	      i[my_rank] = 0;
	    }
	    else {
	      break;
	    }
	  }
	} while (my_rank >= 0);
	f.finish(total, n);
      }
      return total;
    }


    // Reduce the specified dimension of an inactive array of rank > 1
    template <class Func, typename Type, class E>
    inline
    void reduce_dimension(const Expression<Type, E>& rhs, int reduce_dim,
		    Array<E::rank-1,typename Func::total_type,false>& total) {
      Func f;
      ExpressionSize<E::rank> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (dims[0] == 0) {
	// Return empty array if any of these functions applied to an
	// empty array
	total.clear();
      }
      else if (reduce_dim >= E::rank) {
	std::stringstream s;
	s << "In " << f.name() << "(Expression<rank="
	  << E::rank << ">,dim=" << reduce_dim 
	  << "), dim must be less than rank.";
	throw invalid_dimension(s.str() ADEPT_EXCEPTION_LOCATION);
      }
      else {
	// New array has the same dimensions as the input but with one
	// of the dimensions removed
	ExpressionSize<E::rank-1> new_dims;
	int jnew = 0;
	for (int j = 0; j < E::rank; ++j) {
	  if (j != reduce_dim) {
	    new_dims[jnew++] = dims[j];
	  }
	}
	total.resize(new_dims);
	total = f.first_value();
	ExpressionSize<E::rank> i(0);
	ExpressionSize<E::rank-1> inew(0);
	ExpressionSize<E::n_arrays> loc(0);
	int my_rank;
	static const int last = E::rank-1;
	do {
	  i[last] = 0;
	  rhs.set_location(i, loc);
	  // Innermost loop. Note that indexing of total with inew is
	  // not very efficient for high-rank arrays since the
	  // location must be computed from all dimensions each time.
	  if (reduce_dim == last) {
	    for ( ; i[last] < dims[last]; ++i[last]) {
	      f.accumulate(total.get_lvalue(inew), rhs.next_value(loc));
	    }
	  }
	  else {
	    for ( inew[last-1] = 0; i[last] < dims[last]; 
		 ++i[last], ++inew[last-1]) {
	      f.accumulate(total.get_lvalue(inew), rhs.next_value(loc));
	    }
	  }
	  // Advancing to next innermost loop is somewhat involved
	  // since we have to do something different when we reach the
	  // dimension that is being reduced
	  my_rank = E::rank-1;
	  while (--my_rank >= 0) {
	    ++i[my_rank];
	    if (my_rank < reduce_dim) {
	      ++inew[my_rank];
	      if (i[my_rank] >= dims[my_rank]) {
		i[my_rank] = 0;
		inew[my_rank] = 0;
	      }
	      else {
		break;
	      }   
	    }
	    else if (my_rank == reduce_dim) {
	      if (i[my_rank] >= dims[my_rank]) {
		i[my_rank] = 0;
	      }
	      else {
		break;
	      }   
	    }
	    else {
	      ++inew[my_rank-1];
	      if (i[my_rank] >= dims[my_rank]) {
		i[my_rank] = 0;
		inew[my_rank-1] = 0;
	      }
	      else {
		break;
	      }
	    }
	  }
	} while (my_rank >= 0);
	
	if (f.finish_needed) {
	  f.finish(total, dims[reduce_dim]);
	}
      }
    }
  
    // Reduce the entirety of an active array
    template <class Func, typename Type, class E>
    inline
    void reduce_active(const Expression<Type, E>& rhs, Active<Type>& total) {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (!ADEPT_ACTIVE_STACK->is_recording()) {
	total.lvalue() = reduce_inactive<Func>(rhs);
	return;
      }
#endif

      Func f;
      ExpressionSize<E::rank> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (dims[0] == 0) {
	// Return zero if any of these functions applied to an empty
	// array
	total = 0;
      }
      else {
	total.set_value(f.first_value());
	Index n = dims.size();
	ExpressionSize<E::rank> i(0);
	ExpressionSize<E::n_arrays> loc(0);
	int my_rank;
	static const int last = E::rank-1;
	ADEPT_ACTIVE_STACK->check_space(E::n_active * n); // FIX!
	do {
	  i[last] = 0;
	  rhs.set_location(i, loc);
	  // Innermost loop
	  for ( ; i[last] < dims[last]; ++i[last]) {
	    f.accumulate_active(total, rhs, loc);
	  }
	  my_rank = E::rank-1;
	  while (--my_rank >= 0) {
	    if (++i[my_rank] >= dims[my_rank]) {
	      i[my_rank] = 0;
	    }
	    else {
	      break;
	    }
	  }
	} while (my_rank >= 0);
	if (f.active_finish_needed) {
	  f.finish_active(total, n);
	}
      }
    }

    // Reduce the specified dimension of an active array of rank > 1
    template <class Func, typename Type, class E>
    inline
    void reduce_dimension(const Expression<Type, E>& rhs, int reduce_dim,
		Array<E::rank-1,Type,true>& result) {
#ifdef ADEPT_RECORDING_PAUSABLE
      if (!ADEPT_ACTIVE_STACK->is_recording()) {
	// This solution requires more shallow copies than are really
	// needed; could be made more efficient if Array had a member
	// function to link an pre-constructed active Array to
	// inactive data.
	Array<E::rank-1,Type,false> result_inactive;
	reduce_dimension<Func>(rhs, reduce_dim, result_inactive);
	Array<E::rank-1,Type,true> result_active(result_inactive.data(),
						 result_inactive.storage(),
						 result_inactive.dimensions(),
						 result_inactive.offset());
	result >>= result_active;
	return;
      }
#endif

      Func f;
      ExpressionSize<E::rank> dims;
      if (!rhs.get_dimensions(dims)) {
	std::string str = "Array size mismatch in "
	  + rhs.expression_string() + ".";
	throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
      }
      else if (dims[0] == 0) {
	// Return empty array if any of these functions applied to an
	// empty array
	result.clear();
      }
      else if (reduce_dim >= E::rank) {
	std::stringstream s;
	s << "In " << f.name() << "(Expression<rank="
	  << E::rank << ">,dim=" << reduce_dim 
	  << "), dim must be less than rank.";
	throw invalid_dimension(s.str() ADEPT_EXCEPTION_LOCATION);
      }
      else {
	// New array has the same dimensions as the input but with one
	// of the dimensions removed
	ExpressionSize<E::rank-1> new_dims;
	int jnew = 0;
	for (int j = 0; j < E::rank; ++j) {
	  if (j != reduce_dim) {
	    new_dims[jnew++] = dims[j];
	  }
	}
	result.resize(new_dims);
	ExpressionSize<E::rank> i(0);
	ExpressionSize<E::rank-1> inew(0);
	ExpressionSize<E::n_arrays> loc(0);
	int my_rank;
	Active<Type> total;
	do {
	  i[reduce_dim] = 0;
	  //	  total.set_value(f.first_value());
	  total = f.first_value();

	  // Innermost loop. Note that indexing of total with inew is
	  // not very efficient for high-rank arrays since the
	  // location must be computed from all dimensions each time.
	  for ( ; i[reduce_dim] < dims[reduce_dim]; ++i[reduce_dim]) {
	    rhs.set_location(i, loc);
	    f.accumulate_active(total, rhs, loc);
	  }
	  if (f.active_finish_needed) {
	    f.finish_active(total, dims[reduce_dim]);
	  }
	  result.get_lvalue(inew) = total;
	  my_rank = E::rank;
	  while (--my_rank >= 0) {
	    if (my_rank == reduce_dim) {
	      continue;
	    }
	    ++i[my_rank];
	    if (my_rank < reduce_dim) {
	      ++inew[my_rank];
	      if (i[my_rank] >= dims[my_rank]) {
		i[my_rank] = 0;
		inew[my_rank] = 0;
	      }
	      else {
		break;
	      }   
	    }
	    else if (my_rank == reduce_dim) {
	      if (i[my_rank] >= dims[my_rank]) {
		i[my_rank] = 0;
	      }
	      else {
		break;
	      }   
	    }
	    else {
	      ++inew[my_rank-1];
	      if (i[my_rank] >= dims[my_rank]) {
		i[my_rank] = 0;
		inew[my_rank-1] = 0;
	      }
	      else {
		break;
	      }
	    }
	  }
	} while (my_rank >= 0);
      }
    }

  }


  // -------------------------------------------------------------------
  // Section 3. Implement the functions
  // -------------------------------------------------------------------

  // Implement sum(x), sum(x,dim), mean(x), mean(x,dim) etc.
  // Different versions of the "reduce" function are called depending
  // on whether "x" is active and whether "dim" is present.

#define DEFINE_REDUCE_FUNCTION(NAME, CLASSNAME)		\
  /* function(inactive) */				\
  template <typename Type, class E>			\
  inline						\
  typename enable_if<!E::is_active && E::rank != 0,	\
		     Type>::type			\
  NAME(const Expression<Type, E>& rhs) {		\
    return reduce_inactive<CLASSNAME<Type> >(rhs);	\
  }							\
  							\
  /* function(active) */				\
  template <typename Type, class E>			\
  inline						\
  typename enable_if<E::is_active && E::rank != 0,	\
		     Active<Type> >::type		\
  NAME(const Expression<Type, E>& rhs) {		\
    Active<Type> result;				\
    reduce_active<CLASSNAME<Type> >(rhs, result);		\
    return result;					\
  }							\
							\
  /* function(active[rank=1], dim) */			\
  template <typename Type, class E>			\
  inline						\
  typename enable_if<!E::is_active && E::rank == 1,	\
				     Type>::type	\
  NAME(const Expression<Type, E>& rhs, int dim) {	\
    if (dim != 0) {					\
      throw invalid_dimension("Two-argument reduce function applied to vector must have zero as second argument" \
			      ADEPT_EXCEPTION_LOCATION);		\
    }							\
    return reduce_inactive<CLASSNAME<Type> >(rhs);	\
  }							\
  							\
  /* function(active[rank=1], dim) */			\
  template <typename Type, class E>			\
  inline						\
  typename enable_if<E::is_active && E::rank == 1,	\
		     Active<Type> >::type		\
  NAME(const Expression<Type, E>& rhs, int dim) {	\
    if (dim != 0) {					\
      throw invalid_dimension("Two-argument reduce function applied to vector must have zero as second argument" \
			    ADEPT_EXCEPTION_LOCATION);			\
    }							\
    Active<Type> result;				\
    reduce_active<CLASSNAME<Type> >(rhs, result);		\
    return result;					\
  }							\
							\
  /* function(inactive[rank>1], dim) */			\
  /* function(active[rank>1], dim) */			\
  template <typename Type, class E>			\
  inline						\
  typename enable_if<(E::rank > 1),			\
	     Array<E::rank-1,Type,E::is_active> >::type	\
  NAME(const Expression<Type, E>& rhs, int dim) {	\
    Array<E::rank-1,Type,E::is_active> result;		\
    reduce_dimension<CLASSNAME<Type> >(rhs, dim, result);		\
    return result;					\
  }

  DEFINE_REDUCE_FUNCTION(sum, Sum);
  DEFINE_REDUCE_FUNCTION(mean, Mean);
  DEFINE_REDUCE_FUNCTION(product, Product);
  DEFINE_REDUCE_FUNCTION(maxval, MaxVal);
  DEFINE_REDUCE_FUNCTION(minval, MinVal);
  DEFINE_REDUCE_FUNCTION(norm2, Norm2);

#undef DEFINE_REDUCE_FUNCTION


  // Implement all(x), all(x,dim), any(x) and any(x,dim).  Fewer
  // possibilities this time as no active versions.

#define DEFINE_BOOL_REDUCE_FUNCTION(NAME, CLASSNAME)	 \
  template <class E>					 \
  inline bool NAME(const Expression<bool, E>& rhs)	 \
  { return reduce_inactive<CLASSNAME>(rhs); }		 \
  							 \
  template <class E>					 \
  inline						 \
  Array<E::rank-1,bool,false>				 \
  NAME(const Expression<bool, E>& rhs, int dim) {	 \
    Array<E::rank-1,bool,false> result;			 \
    reduce_dimension<CLASSNAME>(rhs, dim, result);		 \
    return result;					 \
  }

  DEFINE_BOOL_REDUCE_FUNCTION(all, All);
  DEFINE_BOOL_REDUCE_FUNCTION(any, Any);
#undef DEFINE_BOOL_REDUCE_FUNCTION

  // count(x) and count(x,dim) is slightly different as it returns
  // Index
  template <class E>
  inline Index count(const Expression<bool, E>& rhs)
  { return reduce_inactive<Count>(rhs); }

  template <class E>
  inline Array<E::rank-1,Index,false>
  count(const Expression<bool, E>& rhs, int dim) {
    Array<E::rank-1,Index,false> result;
    reduce_dimension<Count>(rhs, dim, result);
    return result;
  }


  // -------------------------------------------------------------------
  // Section 4. diag_vector
  // -------------------------------------------------------------------

  // diag_vector(A,offdiag), where A is a 2D array, returns the
  // diagonal indexed by "offdiag" as a 1D array pointing to the
  // original data, or the main diagonal if offidag is missing. Can be
  // used as an lvalue.
  template <typename Type, bool IsActive>
  Array<1,Type,IsActive>
  diag_vector(Array<2,Type,IsActive>& A, Index offdiag = 0) {
    ExpressionSize<2> dims = A.dimensions();
    ExpressionSize<2> offset = A.offset();
    ExpressionSize<1> new_dim, new_offset;
    new_offset[0] = offset[0]+offset[1];
    if (offdiag >= 0) {
      new_dim[0] = std::min(dims[0], dims[1]-offdiag);
      return Array<1,Type,IsActive>(A.data()+offdiag*offset[1],
				    A.storage(), new_dim, new_offset);
    }
    else {
      new_dim[0] = std::min(dims[0]+offdiag, dims[1]);
      return Array<1,Type,IsActive>(A.data()-offdiag*offset[0],
				    A.storage(), new_dim, new_offset);
    }
  }

  // diag_vector(A,offdiag), where A is a 2D expression, returns the
  // diagonal indexed by "offdiag" as a 1D array, or the main diagonal
  // if offidag is missing. Cannot be used as an lvalue.
  template <typename Type, class E>
  typename internal::enable_if<E::rank == 2 && !E::is_active,
			       Array<1,Type,E::is_active> >::type
  diag_vector(const Expression<Type,E>& arg, Index offdiag = 0) {
    ExpressionSize<2> dims;
    if (!arg.get_dimensions(dims)) {
      std::string str;
      str += "Array size mismatch in ";
      str += arg.expression_string();
      throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
    }

    ExpressionSize<2> i;
    ExpressionSize<E::n_arrays> ind;
    if (offdiag >= 0) {
      Index new_dim = std::min(dims[0], dims[1]-offdiag);
      Array<1,Type,E::is_active> v(new_dim);
      for (int j = 0; j < new_dim; ++j) {
	i[0] = j;
	i[1] = j+offdiag;
	arg.set_location(i, ind);
	v(j) = arg.next_value(ind);
      }
      return v;
    }
    else {
      Index new_dim = std::min(dims[0]+offdiag, dims[1]);
      Array<1,Type,E::is_active> v(new_dim);
      for (int j = 0; j < new_dim; ++j) {
	i[0] = j;
	i[1] = j+offdiag;
	arg.set_location(i, ind);
	v(j) = arg.next_value(ind);
      }
      return v;
    }
  }
  template <typename Type, class E>
  typename internal::enable_if<E::rank == 2 && E::is_active,
			       Array<1,Type,E::is_active> >::type
  diag_vector(const Expression<Type,E>& arg, Index offdiag = 0) {
    ExpressionSize<2> dims;
    if (!arg.get_dimensions(dims)) {
      std::string str;
      str += "Array size mismatch in ";
      str += arg.expression_string();
      throw size_mismatch(str ADEPT_EXCEPTION_LOCATION);
    }

    ExpressionSize<2> i;
    ExpressionSize<E::n_arrays> ind;
    if (offdiag >= 0) {
      Index new_dim = std::min(dims[0], dims[1]-offdiag);
      Array<1,Type,E::is_active> v(new_dim);
      for (int j = 0; j < new_dim; ++j) {
	i[0] = j;
	i[1] = j+offdiag;
	arg.set_location(i, ind);
	v.data()[j] = arg.next_value_and_gradient(*ADEPT_ACTIVE_STACK,ind);
	ADEPT_ACTIVE_STACK->push_lhs(v.gradient_index()+j);
      }
      return v;
    }
    else {
      Index new_dim = std::min(dims[0]+offdiag, dims[1]);
      Array<1,Type,E::is_active> v(new_dim);
      for (int j = 0; j < new_dim; ++j) {
	i[0] = j;
	i[1] = j+offdiag;
	arg.set_location(i, ind);
	v.data()[j] = arg.next_value_and_gradient(*ADEPT_ACTIVE_STACK,ind);
	ADEPT_ACTIVE_STACK->push_lhs(v.gradient_index()+j);
      }
      return v;
    }
  }

  // diag_matrix(v,offdiag), where v is a 1D expression, returns a
  // DiagMatrix whose diagonal is a copy of v. Cannot be used as an
  // lvalue.
  template <typename Type, class E>
  typename internal::enable_if<E::rank == 1,
       SpecialMatrix<Type, internal::BandEngine<internal::ROW_MAJOR,0,0>,
		    E::is_active> >::type
  diag_matrix(const Expression<Type,E>& arg) {
    Array<1,Type,E::is_active> v = arg;
    return v.diag_matrix();
  }

  // -------------------------------------------------------------------
  // dot_product
  // -------------------------------------------------------------------
  template <typename LType, typename RType, class L, class R>
  typename enable_if<L::rank == 1 && R::rank == 1,
	     typename internal::active_scalar<typename internal::promote<LType,RType>::type,
				     L::is_active || R::is_active>::type>::type
  dot_product(const Expression<LType,L>& l,
	      const Expression<RType,R>& r) {
    return sum(l*r);
  }

} // End namespace adept

#endif
