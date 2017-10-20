/* Expression.h -- Base class for arrays and active objects

    Copyright (C) 2014-2015 European Centre for Medium-Range Weather Forecasts

    Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptExpression_H
#define AdeptExpression_H

#include <sstream>
#include <cmath>

#include <adept/ExpressionSize.h>
#include <adept/traits.h>
#include <adept/exception.h>
#include <adept/Stack.h>
#include <adept/ScratchVector.h>
#include <adept/Packet.h>

#define ADEPT_GNU_COMPILER (defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER))

namespace adept {

  using internal::Packet;

  // ---------------------------------------------------------------------
  // SECTION 0: Forward declarations 
  // ---------------------------------------------------------------------
  
  template <int Rank, typename Type, bool IsActive> class Array;

  // ---------------------------------------------------------------------
  // SECTION 1: Definition of Expression type
  // ---------------------------------------------------------------------

  // All types of expression derive from Expression.  "A" is the
  // actual type of the expression (a use of the Curiously Recurring
  // Template Pattern).
  template <typename Type, class A>
  struct Expression {

    // Static information about the expression
  public:
    typedef Type type;

    // Expression is used as the base class for the curiously
    // recurring template pattern (CRTP), and it needs to be able to
    // access the static members of any derived classes. Unfortunately
    // this has to be done in different ways on different compilers
    // :-(
#if ADEPT_GNU_COMPILER
    // In the case of g++, we must define the static constants
    // immediately; if they are defined outside the class, they are
    // not treated as static constants and so cannot be used in
    // compile time expressions.

    // Rank of the array
    static const int  rank = A::rank_;

    // Number of active variables in the expression (where each array
    // counts as 1), used to work out how much space must be reserved
    // on the operation stack
    static const int  n_active = A::n_active_;

    // Number of scratch floating-point variables needed in the
    // expression, for example to store the result of a calculation
    // when it is needed again to compult the equivalent differential
    // statement
    static const int  n_scratch = A::n_scratch_;

    // Number of arrays in the expression, needed as each array uses a
    // scratch Index variable to store its current memory location,
    // otherwise when looping over all the elements in a
    // multidimensional expression this is expensive to recompute
    static const int  n_arrays = A::n_arrays_;
    static const bool is_active = A::is_active_;

    // An expression is currently vectorizable only if it is of
    // floating point type, all arrays have the same type, and if the
    // only mathematical operators and functions can be treated by
    // hardware vector operations (+-*/sqrt)
    static const bool is_vectorizable
      = A::is_vectorizable_ && Packet<Type>::is_vectorized;
#else
    // ...but for other compilers, defining the static constants
    // in-class means that derived classes using CRTP fail to compile,
    // so the definitions need to be provided after the class
    // definition.

    static const int  rank; // = A::rank_;
    static const int  n_active; // = A::n_active_;
    static const int  n_scratch; // = A::n_scratch_;
    static const int  n_arrays; // = A::n_arrays_;
    static const bool is_active; // = A::is_active_;
    static const bool is_vectorizable;
    //      = A::is_vectorizable_ && Packet<Type>::is_vectorized;
#endif

    // Fall-back position is that an expression is not vectorizable:
    // only those that are need to define is_vectorizable_.
    static const bool is_vectorizable_ = false;

    // Classes derived from this one that do not define how many
    // scratch variables, active variables or arrays they contain are
    // assumed to need zero
    static const int  n_scratch_ = 0;
    static const int  n_active_ = 0;
    //static const int  n_arrays_ = 0;
    static const bool is_active_ = false;

    // Expressions cannot be lvalues by default, but override this
    // bool if they are
    static const bool is_lvalue = false;

    // The presence of _adept_expression_flag is used to define the
    // adept::is_not_expression trait
    typedef bool _adept_expression_flag;

    // Cast the expression to its true type, given by the template
    // argument
    const A& cast() const { return static_cast<const A&>(*this); }
    
    // Return the dimensions of the expression
    template <int Rank>
    bool get_dimensions(ExpressionSize<Rank>& dim) const {
      return cast().get_dimensions_(dim);
    }

    // Return the dimension of a rank-1 expression to be used as an
    // index to an array, where "len" is the length of the dimension
    // being indexed
//     Index get_dimension_with_len(Index len) const {
//       ADEPT_STATIC_ASSERT(rank == 1,
// 		  GET_DIMENSION_WITH_LEN_ONLY_APPLICABLE_TO_ARRAYS_OF_RANK_1);
//       return cast().get_dimension_with_len_(len);
//     }

    // Return a string representation of the expression
    std::string expression_string() const {
      return cast().expression_string_();
    }
    
    // Get a copy of the value at the specified location
    /*
    template <int Rank>
    Type value_(const ExpressionSize<Rank>& j) const {
      return cast().value_(j);
    }
    */
    /*
    // Get a reference to the value at the specified location
    template <int Rank>
    Type& get_lvalue(const ExpressionSize<Rank>& j) {
      throw array_exception("Attempt to use an Expression as an l-value");
    }
    */

    // Get the value at the specified location, but also supplying a
    // length so that if the expression contains "end", it will
    // resolve to length-1
    /*
    template <int Rank>
    Type value_with_len(const ExpressionSize<Rank>& j, Index len) const {
      return cast().value_with_len_(j, len);
    }
    */
    //    template <int reqRank>
    Type value_with_len(Index j, Index len) const {
      ADEPT_STATIC_ASSERT(A::rank<=1,
		  VALUE_WITH_LEN_ONLY_APPLICABLE_TO_ARRAYS_OF_RANK_0_OR_1);
      return cast().value_with_len_(j, len);
    }

    // These functions are for rank-0 expressions where there is no
    // indexing required
    Type scalar_value() const { 
      //      ADEPT_STATIC_ASSERT(rank > 0,
      //	  SCALAR_VALUE_ONLY_APPLICABLE_TO_EXPRESSIONS_OF_RANK_0);
      //      return cast().scalar_value_(); 
      ExpressionSize<0> dummy_index;
      return cast().template value_at_location_<0>(dummy_index);
    }
    //    Type get_scalar_with_len(Index len) const { return cast().get_scalar_with_len(len); }

    // Return true if any memory in the expression lies between mem1
    // and mem2: used to test for aliasing when doing assignment.
    bool is_aliased(const Type* mem1, const Type* mem2) const {
      return cast().is_aliased_(mem1, mem2);
    }

    // Return true if the fastest varying dimension of all the arrays
    // in the expression are contiguous and increasing.  If so, we can
    // more simply increment their indices.
    bool all_arrays_contiguous() const {
      return cast().all_arrays_contiguous_();
    }

    // By default, arrays are contiguous (this fall-back used for
    // objects that aren't arrays)
    bool all_arrays_contiguous_() const { return true; }

    // In order to perform optimal vectorization, the first memory
    // addresses of each inner dimension must be aligned
    // appropriately, or they should all have the same offset so that
    // this number of scalar operations can be performed at the start
    // before begining on vector instructions.  This function returns
    // the offset of the data in any arrays in the expression, or -1 if
    // there is a clash in offsets.
    int alignment_offset() const {
      int val = cast().template alignment_offset_<Packet<Type>::size>();
      if (val < Packet<Type>::size) {
	return val;
      }
      else {
	return 0;
      }
    }
    
    // Fall-back position is that alignment doesn't matter for this
    // object, which is encoded by returning n
    template <int n>
    int alignment_offset_() const { return n; }

    // If the sub-expression is of a different type from that
    // requested then we assume there must be no aliasing.
    template <typename MyType>
    typename internal::enable_if<!internal::is_same<MyType,Type>::value, bool>::type
    is_aliased(const MyType* mem1, const MyType* mem2) const {
      return false;
    }

    // Calculate the gradient of the function and pass the necessary
    // information to the Stack
    /*
    template <int Rank>
    void calc_gradient(Stack& stack, const ExpressionSize<Rank>& i) const {
      cast().calc_gradient(stack, i);
    }
    */
    
    // As the previous but multiplying the gradient by "multiplier"
    /*
    template <int Rank>
    void calc_gradient(Stack& stack, Type& multiplier,
		       const ExpressionSize<Rank>& i) const {
      cast().calc_gradient(stack, multiplier, i);
    }
    */
  
    Type 
    scalar_value_and_gradient(Stack& stack) const {
      internal::ScratchVector<n_scratch> scratch;
      //      Type val = cast().scalar_value_store_<0>(scratch);
      //      cast().calc_scalar_gradient_(*ADEPT_ACTIVE_STACK, scratch);
      ExpressionSize<0> dummy_index;
      Type val = cast().template value_at_location_store_<0,0>(dummy_index, scratch);
      cast().template calc_gradient_<0,0>(*ADEPT_ACTIVE_STACK, dummy_index, scratch);
      return val;
    }
 
    /*
    template <int Rank>
    Type value_and_gradient(Stack& stack, 
			    const ExpressionSize<Rank>& i) const {
      Type val = cast().get(i);
      cast().calc_gradient(stack, i);
      return val;
    }
    */

    // For each array in the expression use location "i" to return the
    // memory index
    template <int Rank, int NArrays>
    void
    set_location(const ExpressionSize<Rank>& i, 
		 ExpressionSize<NArrays>& index) const {
      cast().template set_location_<0>(i, index);
    }

    // Get the value at the specified location and move to the next
    // location
    template <int NArrays>
    Type next_value(ExpressionSize<NArrays>& index) const {
      Type val = cast().template value_at_location_<0>(index);
      cast().template advance_location_<0>(index);
      return val;
    }
    // If all arrays are have an inner dimension that is contiguous
    // and increasing then their indices may be incremented all
    // together, which is more efficient
    template <int NArrays>
    Type next_value_contiguous(ExpressionSize<NArrays>& index) const {
      Type val = cast().template value_at_location_<0>(index);
      ++index;
      return val;
    }

    template <int NArrays>
    Packet<Type> next_packet(ExpressionSize<NArrays>& index) const {
      asm("# %%% ADEPT PACKET CALCULATION");
      Packet<Type> val
      	= cast().template packet_at_location_<0>(index);
      asm("# %%% ADEPT END PACKET CALCULATION");
      index += Packet<Type>::size;
      return val;
    }

    template <int NArrays>
    Type value_at_location(ExpressionSize<NArrays>& index) const {
      return cast().template value_at_location_<0>(index);
    }
    template <int NArrays>
    void advance_location(ExpressionSize<NArrays>& index) const {
      cast().template advance_location_<0>(index);
    }

    // Get the value at the specified location, calculate the gradient
    // and move to the next location
    template <int NArrays>
    Type next_value_and_gradient(Stack& stack,
				 ExpressionSize<NArrays>& index) const {
      internal::ScratchVector<n_scratch> scratch;
      Type val = cast().template value_at_location_store_<0,0>(index, scratch);
      cast().template calc_gradient_<0,0>(stack, index, scratch);
      cast().template advance_location_<0>(index);
      //++index;
      return val;
    }
    template <int NArrays>
    Type next_value_and_gradient_contiguous(Stack& stack,
				 ExpressionSize<NArrays>& index) const {
      internal::ScratchVector<n_scratch> scratch;
      Type val = cast().template value_at_location_store_<0,0>(index, scratch);
      cast().template calc_gradient_<0,0>(stack, index, scratch);
      //cast().template advance_location_<0>(index);
      ++index;
      return val;
    }

    /*
    template <int NArrays, typename MyType>
    Type next_value_and_gradient(Stack& stack,
				 ExpressionSize<NArrays>& index,
				 const MyType& multiplier) const {
      internal::ScratchVector<n_scratch> scratch;
      Type val = cast().template value_at_location_store_<0,0>(index, scratch);
      cast().template calc_gradient_<0,0>(stack, index, scratch, multiplier);
      cast().template advance_location_<0>(index);
      return val;
    }
    */

    // This is used in norm2()
    template <int NArrays, typename MyType>
    Type next_value_and_gradient_special(Stack& stack,
				 ExpressionSize<NArrays>& index,
				 const MyType& multiplier) const {
      internal::ScratchVector<n_scratch> scratch;
      Type val = cast().template value_at_location_store_<0,0>(index, scratch);
      cast().template calc_gradient_<0,0>(stack, index, scratch, multiplier*val);
      cast().template advance_location_<0>(index);
      return val;
    }

    // Inaccessible methods
    //  private:
    //    Expression(const Expression&) { }

  }; // End struct Expression

#if !ADEPT_GNU_COMPILER
  // Non-GNU compilers have problems with static members of Expression
  // depending on its template argument when used in combination with
  // the Curiously Recurring Template Pattern. This can be solved by
  // putting the definition of each immediately after the Expression
  // class definition.
  template <typename Type, class A>
  const int Expression<Type,A>::rank = A::rank_;
  template <typename Type, class A>
  const int Expression<Type,A>::n_active = A::n_active_;
  template <typename Type, class A>
  const int Expression<Type,A>::n_scratch = A::n_scratch_;
  template <typename Type, class A>
  const int Expression<Type,A>::n_arrays = A::n_arrays_;
  template <typename Type, class A>
  const bool Expression<Type,A>::is_active = A::is_active_;
  template <typename Type, class A>
  const bool Expression<Type,A>::is_vectorizable
    = A::is_vectorizable_ && Packet<Type>::is_vectorized;
#endif
#undef ADEPT_GNU_COMPILER

  // ---------------------------------------------------------------------
  // SECTION 2: Definition of Scalar type
  // ---------------------------------------------------------------------

  // Specific types of operation are in the adept::internal namespace
  namespace internal {

    // SCALAR

    template <typename Type>
    struct Scalar : public Expression<Type, Scalar<Type> > {
      static const int  rank_      = 0;
      static const int  n_scratch_ = 0;
      static const int  n_active_ = 0;
      static const int  n_arrays_ = 0;
      static const bool is_active_ = false;
      static const bool is_vectorizable_ = true;

      Scalar(const Type& value) : val_(value) { }

      bool get_dimensions_(ExpressionSize<0>& dim) const { return true; }

      std::string expression_string_() const {
	std::stringstream s;
	s << val_;
	return s.str();
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const { return false; }

      Type value_with_len_(const Index& j, const Index& len) const
      { return val_; }

      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const { } 

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const
      { return val_; }

      template <int MyArrayNum, int NArrays>
      Packet<Type>
      packet_at_location_(const ExpressionSize<NArrays>& loc) const
      { return Packet<Type>(val_); }

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
			  const ScratchVector<NScratch>& scratch) const {}

      template <int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch, typename MyType>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  const MyType& multiplier) const {}

      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {}

    protected:
      Type val_;
      
    };
  }
}

#endif // AdeptExpression_H
