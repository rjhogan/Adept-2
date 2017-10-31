/* Expression.h -- Base class for arrays and active objects

    Copyright (C) 2014-2017 European Centre for Medium-Range Weather Forecasts

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

    // There are several "static const" members in the derived
    // classes, some of which require fall-back values, defined here:

    // By default an expression is not vectorizable.
    static const bool is_vectorizable = false;

    // Classes derived from this one that do not define how many
    // scratch variables, active variables or arrays they contain are
    // assumed to need zero
    static const int  n_scratch = 0;

    // Number of active variables in the expression (where each array
    // counts as 1), used to work out how much space must be reserved
    // on the operation stack
    static const int  n_active = 0;

    // Is this an active expression?
    static const bool is_active = false;

    // Expressions cannot be lvalues by default
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

    // Return a string representation of the expression
    std::string expression_string() const {
      return cast().expression_string_();
    }
    
    Type value_with_len(Index j, Index len) const {
      ADEPT_STATIC_ASSERT(A::rank<=1,
		  VALUE_WITH_LEN_ONLY_APPLICABLE_TO_ARRAYS_OF_RANK_0_OR_1);
      return cast().value_with_len_(j, len);
    }

    // These functions are for rank-0 expressions where there is no
    // indexing required
    Type scalar_value() const { 
      ExpressionSize<0> dummy_index;
      return cast().template value_at_location_<0>(dummy_index);
    }

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

    // Are all the arrays in the expression aligned to a Packet<Type>
    // boundary?
    bool is_aligned() const {
      return cast().is_aligned();
    }

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
  
    Type 
    scalar_value_and_gradient(Stack& stack) const {
      internal::ScratchVector<A::n_scratch> scratch;
      ExpressionSize<0> dummy_index;
      Type val = cast().template value_at_location_store_<0,0>(dummy_index, scratch);
      cast().template calc_gradient_<0,0>(*ADEPT_ACTIVE_STACK, dummy_index, scratch);
      return val;
    }
 
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
      Packet<Type> val
      	= cast().template packet_at_location_<0>(index);
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
      internal::ScratchVector<A::n_scratch> scratch;
      Type val = cast().template value_at_location_store_<0,0>(index, scratch);
      cast().template calc_gradient_<0,0>(stack, index, scratch);
      cast().template advance_location_<0>(index);
      //++index;
      return val;
    }
    template <int NArrays>
    Type next_value_and_gradient_contiguous(Stack& stack,
				 ExpressionSize<NArrays>& index) const {
      internal::ScratchVector<A::n_scratch> scratch;
      Type val = cast().template value_at_location_store_<0,0>(index, scratch);
      cast().template calc_gradient_<0,0>(stack, index, scratch);
      //cast().template advance_location_<0>(index);
      ++index;
      return val;
    }

    // This is used in norm2()
    template <int NArrays, typename MyType>
    Type next_value_and_gradient_special(Stack& stack,
				 ExpressionSize<NArrays>& index,
				 const MyType& multiplier) const {
      internal::ScratchVector<A::n_scratch> scratch;
      Type val = cast().template value_at_location_store_<0,0>(index, scratch);
      cast().template calc_gradient_<0,0>(stack, index, scratch, multiplier*val);
      cast().template advance_location_<0>(index);
      return val;
    }

    // Inaccessible methods
    //  private:
    //    Expression(const Expression&) { }

  }; // End struct Expression


  // ---------------------------------------------------------------------
  // SECTION 2: Definition of Scalar type
  // ---------------------------------------------------------------------

  // Specific types of operation are in the adept::internal namespace
  namespace internal {

    // SCALAR

    template <typename Type>
    struct Scalar : public Expression<Type, Scalar<Type> > {
      static const int  rank       = 0;
      static const int  n_scratch  = 0;
      static const int  n_active   = 0;
      static const int  n_arrays   = 0;
      static const bool is_active  = false;
      static const bool is_vectorizable = true;

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

      template <bool IsAligned,	int MyArrayNum, typename PacketType,
	int NArrays>
      PacketType values_at_location_(const ExpressionSize<NArrays>& loc) const {
	return PacketType(val_);
      }

      template <bool UseStored, bool IsAligned,	int MyArrayNum, int MyScratchNum,
		typename PacketType, int NArrays, int NScratch>
      PacketType values_at_location_store_(const ExpressionSize<NArrays>& loc,
		   ScratchVector<NScratch,PacketType>& scratch) const {
	return PacketType(val_);
      }

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

      template <bool IsAligned, int MyArrayNum, int MyScratchNum, int MyActiveNum,
		int NArrays, int NScratch, int NActive>
      void calc_gradient_packet_(Stack& stack, 
				 const ExpressionSize<NArrays>& loc,
				 const ScratchVector<NScratch,Packet<Real> >& scratch,
				 ScratchVector<NActive,Packet<Real> >& gradients) const {}

      template <bool IsAligned, int MyArrayNum, int MyScratchNum, int MyActiveNum,
		int NArrays, int NScratch, int NActive, typename MyType>
      void calc_gradient_packet_(Stack& stack, 
				 const ExpressionSize<NArrays>& loc,
				 const ScratchVector<NScratch,Packet<Real> >& scratch,
				 ScratchVector<NActive,Packet<Real> >& gradients,
				 const MyType& multiplier) const {}

      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {}

    protected:
      Type val_;
      
    };



    // ---------------------------------------------------------------------
    // SECTION 3. "expr_cast" helper 
    // ---------------------------------------------------------------------

    // The following enables one of the static consts only in a
    // derived class of Expression to be extracted, and is useful when
    // you don't know whether a template argument to a function is an
    // Expression or a class derived from it.  Thus
    // expr_cast<Expression<double,Array> >::is_vectorizable and
    // expr_cast<Array>::is_vectorizable would both return
    // Array::is_vectorizable.

    template <class E>
    struct expr_cast {
      // Rank of the array
      static const int  rank = E::rank;
      // Number of scratch floating-point variables needed in the
      // expression, for example to store the result of a calculation
      // when it is needed again to compult the equivalent differential
      // statement
      static const int  n_scratch = E::n_scratch;
      // Number of arrays within the expression; more specifically,
      // the number of indices required to store the location of an
      // element of the array
      static const int  n_arrays = E::n_arrays;
      // Number of active terms in the expression
      static const int  n_active = E::n_active;
      // Is this an array expression?
      static const bool is_array = (E::rank > 0);
      // Is this an array expression with dimension of 2 or more?
      static const bool is_multidimensional = (E::rank > 1);
      // Is this an active expression?
      static const bool is_active = E::is_active;
      // Is this expression actually an lvalue such as Array or
      // FixedArray?
      static const bool is_lvalue = E::is_lvalue;
      // Is this expression vectorizable (conditional on a few extra
      // run-time checks)?
      static const bool is_vectorizable = E::is_vectorizable;  
    };

    template <typename T, class E>
    struct expr_cast<Expression<T,E> > {
      static const int  rank = E::rank;
      static const int  n_scratch = E::n_scratch;
      static const int  n_arrays = E::n_arrays;
      static const int  n_active = E::n_active;
      static const bool is_array = (E::rank > 0);
      static const bool is_multidimensional = (E::rank > 1);
      static const bool is_active = E::is_active;
      static const bool is_lvalue = E::is_lvalue;
      static const bool is_vectorizable = E::is_vectorizable;
    };

  }
}

#endif // AdeptExpression_H
