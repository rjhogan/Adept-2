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
#include <adept/VectorOrientation.h>

namespace adept {

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
    static const bool is_array  = (rank > 0);
    static const bool is_active = A::is_active_;
    static const bool is_multidimensional = (rank > 1);
    static const bool is_lvalue = A::is_lvalue_;
    static const VectorOrientation vector_orientation = A::vector_orientation_;

    // Classes derived from this one that do not define how many
    // scratch variables, active variables or arrays they contain are
    // assumed to need zero
    static const int  n_scratch_ = 0;
    static const int  n_active_ = 0;
    //static const int  n_arrays_ = 0;
    static const bool is_active_ = false;

    // Expressions cannot be lvalues by default, but override this
    // bool if they are
    static const bool is_lvalue_ = false;

    static const VectorOrientation vector_orientation_ = UNSPECIFIED_VECTOR_ORIENTATION;

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
      ADEPT_STATIC_ASSERT(!is_multidimensional,
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
      //      ++index;
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


    // ---------------------------------------------------------------------
    // SECTION 3.1: Unary operations: define UnaryOperation type
    // ---------------------------------------------------------------------

    // Unary operations derive from this class, where Op is a policy
    // class defining how to implement the operation, and R is the
    // type of the argument of the operation
    template <typename Type, template<class> class Op, class R>
    struct UnaryOperation
      : public Expression<Type, UnaryOperation<Type, Op, R> >,
	protected Op<Type> {
      
      static const int  rank_      = R::rank;
      static const bool is_active_ = R::is_active && !is_same<Type,bool>::value;
      static const int  n_active_ = R::n_active;
      // FIX! Only store if active and if needed
      static const int  n_scratch_ = 1 + R::n_scratch;
      static const int  n_arrays_ = R::n_arrays;
      static const VectorOrientation vector_orientation_ = R::vector_orientation;

      using Op<Type>::operation;
      using Op<Type>::operation_string;
      using Op<Type>::derivative;
      
      const R& arg;

      UnaryOperation(const Expression<Type, R>& arg_)
	: arg(arg_.cast()) { }
      
      template <int Rank>
      bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return arg.get_dimensions(dim);
      }

//       Index get_dimension_with_len(Index len) const {
// 	return arg.get_dimension_with_len_(len);
//       }

      std::string expression_string_() const {
	std::string str;
	str = operation_string();
	str += "(" + static_cast<const R*>(&arg)->expression_string() + ")";
	return str;
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return arg.is_aliased(mem1, mem2);
      }

      /*
      template <int Rank>
      Type get(const ExpressionSize<Rank>& i) const {
	return operation(arg.get(i));
      }
      */

      template <int Rank>
      Type value_with_len_(Index i, Index len) const {
	return operation(arg.value_with_len(i, len));
      }
      /*
      template <int Rank>
      Type get_scalar() const {
	return operation(arg.get_scalar());
      }
      template <int Rank>
      Type get_scalar_with_len() const {
	return operation(arg.get_scalar_with_len());
      }
      */
      
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	arg.advance_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(arg.value_at_location_<MyArrayNum>(loc));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] 
	  = operation(arg.value_at_location_store_<MyArrayNum,MyScratchNum+1>(loc, scratch));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	arg.template calc_gradient_<MyArrayNum, MyScratchNum+1>(stack, loc, scratch,
		derivative(arg.value_stored_<MyArrayNum,MyScratchNum+1>(loc, scratch),
			   scratch[MyScratchNum]));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch,
		typename MyType>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
	arg.template calc_gradient_<MyArrayNum, MyScratchNum+1>(stack, loc, scratch,
		multiplier*derivative(arg.value_stored_<MyArrayNum,MyScratchNum+1>(loc, scratch), 
				      scratch[MyScratchNum]));
      }

      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {
	arg.set_location_<MyArrayNum>(i, index);
      }

    }; // End UnaryOperation type
  
  } // End namespace internal

} // End namespace adept


// ---------------------------------------------------------------------
// SECTION 3.2: Unary operations: define specific operations
// ---------------------------------------------------------------------

// Overloads of mathematical functions only works if done in the
// global namespace
#define ADEPT_DEF_UNARY_FUNC(NAME, FUNC, RAWFUNC, STRING, DERIVATIVE)	\
  namespace adept{							\
    namespace internal {						\
      template <typename Type>						\
      struct NAME  {							\
	static const bool is_operator = false;				\
	const char* operation_string() const { return STRING; }		\
	Type operation(const Type& val) const {				\
	  return RAWFUNC(val);						\
	}								\
	Type derivative(const Type& val, const Type& result) const {	\
	  return DERIVATIVE;						\
	}								\
	Type fast_sqr(Type val) { return val*val; }			\
      };								\
    } /* End namespace internal */					\
  } /* End namespace adept */						\
  template <class Type, class R>					\
  inline								\
  adept::internal::UnaryOperation<Type, adept::internal::NAME, R>	\
  FUNC(const adept::Expression<Type, R>& r)	{			\
    return adept::internal::UnaryOperation<Type,			\
      adept::internal::NAME, R>(r.cast());				\
  }

// Functions y(x) whose derivative depends on the argument of the
// function, i.e. dy(x)/dx = f(x)
ADEPT_DEF_UNARY_FUNC(Log,   log,   log,   "log",   1.0/val)
ADEPT_DEF_UNARY_FUNC(Log10, log10, log10, "log10", 0.43429448190325182765/val)
ADEPT_DEF_UNARY_FUNC(Log2,  log2,  log2,  "log2",  1.44269504088896340737/val)
ADEPT_DEF_UNARY_FUNC(Sin,   sin,   sin,   "sin",   cos(val))
ADEPT_DEF_UNARY_FUNC(Cos,   cos,   cos,   "cos",   -sin(val))
ADEPT_DEF_UNARY_FUNC(Tan,   tan,   tan,   "tan",   1.0/fast_sqr(cos(val)))
ADEPT_DEF_UNARY_FUNC(Asin,  asin,  asin,  "asin",  1.0/sqrt(1.0-val*val))
ADEPT_DEF_UNARY_FUNC(Acos,  acos,  acos,  "acos",  -1.0/sqrt(1.0-val*val))
ADEPT_DEF_UNARY_FUNC(Atan,  atan,  atan,  "atan",  1.0/(1.0+val*val))
ADEPT_DEF_UNARY_FUNC(Sinh,  sinh,  sinh,  "sinh",  cosh(val))
ADEPT_DEF_UNARY_FUNC(Cosh,  cosh,  cosh,  "cosh",  sinh(val))
ADEPT_DEF_UNARY_FUNC(Abs,   abs,   std::abs, "abs", ((val>0.0)-(val<0.0)))
ADEPT_DEF_UNARY_FUNC(Fabs,  fabs,  std::abs, "fabs", ((val>0.0)-(val<0.0)))
ADEPT_DEF_UNARY_FUNC(Expm1, expm1, expm1, "expm1", exp(val))
ADEPT_DEF_UNARY_FUNC(Exp2,  exp2,  exp2,  "exp2",  0.6931471805599453094172321214581766*exp2(val))
ADEPT_DEF_UNARY_FUNC(Log1p, log1p, log1p, "log1p", 1.0/(1.0+val))
ADEPT_DEF_UNARY_FUNC(Asinh, asinh, asinh, "asinh", 1.0/sqrt(val*val+1.0))
ADEPT_DEF_UNARY_FUNC(Acosh, acosh, acosh, "acosh", 1.0/sqrt(val*val-1.0))
ADEPT_DEF_UNARY_FUNC(Atanh, atanh, atanh, "atanh", 1.0/(1.0-val*val))
ADEPT_DEF_UNARY_FUNC(Erf,   erf,   erf,   "erf",   1.12837916709551*exp(-val*val))
ADEPT_DEF_UNARY_FUNC(Erfc,  erfc,  erfc,  "erfc",  -1.12837916709551*exp(-val*val))

// Functions y(x) whose derivative depends on the result of the
// function, i.e. dy(x)/dx = f(y)
ADEPT_DEF_UNARY_FUNC(Exp,   exp,   exp,   "exp",   result)
ADEPT_DEF_UNARY_FUNC(Sqrt,  sqrt,  sqrt,  "sqrt",  0.5/result)
ADEPT_DEF_UNARY_FUNC(Cbrt,  cbrt,  cbrt,  "cbrt",  (1.0/3.0)/(result*result))
ADEPT_DEF_UNARY_FUNC(Tanh,  tanh,  tanh,  "tanh",  1.0 - result*result)

// Functions with zero derivative
ADEPT_DEF_UNARY_FUNC(Round, round, round, "round", 0.0)
ADEPT_DEF_UNARY_FUNC(Ceil,  ceil,  ceil,  "ceil",  0.0)
ADEPT_DEF_UNARY_FUNC(Floor, floor, floor, "floor", 0.0)
ADEPT_DEF_UNARY_FUNC(Trunc, trunc, trunc, "trunc", 0.0)
ADEPT_DEF_UNARY_FUNC(Rint,  rint,  rint,  "rint",  0.0)
ADEPT_DEF_UNARY_FUNC(Nearbyint,nearbyint,nearbyint,"nearbyint",0.0)

// Operators
ADEPT_DEF_UNARY_FUNC(UnaryPlus,  operator+, +, "+", 1.0)
ADEPT_DEF_UNARY_FUNC(UnaryMinus, operator-, -, "-", -1.0)
ADEPT_DEF_UNARY_FUNC(Not,        operator!, !, "!", 0.0)

//#undef ADEPT_DEF_UNARY_FUNC


// ---------------------------------------------------------------------
// SECTION 3.3: Unary operations: define noalias function
// ---------------------------------------------------------------------

namespace adept {
  namespace internal {
    // No-alias wrapper for enabling noalias()
    template <typename Type, class R>
    struct NoAlias
      : public Expression<Type, NoAlias<Type, R> > 
    {
      static const int  rank_      = R::rank;
      static const bool is_active_ = R::is_active;
      static const int  n_active_  = R::n_active;
      static const int  n_scratch_ = R::n_scratch;
      static const int  n_arrays_  = R::n_arrays;
      
      const R& arg;

      NoAlias(const Expression<Type, R>& arg_)
	: arg(arg_.cast()) { }
      
      template <int Rank>
	bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return arg.get_dimensions(dim);
      }

//       Index get_dimension_with_len(Index len) const {
// 	return arg.get_dimension_with_len_(len);
//       }

      std::string expression_string_() const {
	std::string str = "noalias(";
	str += static_cast<const R*>(&arg)->expression_string() + ")";
	return str;
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return false;
      }

      template <int Rank>
      Type value_with_len_(Index i, Index len) const {
	return operation(arg.value_with_len(i, len));
      }
      
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	arg.advance_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return arg.value_at_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return arg.value_at_location_store_<MyArrayNum,MyScratchNum>(loc, 
								     scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	arg.template calc_gradient_<MyArrayNum, MyScratchNum>(stack, loc, 
							      scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch,
		typename MyType>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
	arg.template calc_gradient_<MyArrayNum, MyScratchNum+1>(stack, loc, 
								scratch,
								multiplier);
      }

      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {
	arg.set_location_<MyArrayNum>(i, index);
      }

    }; // End struct NoAlias

  }

  template <typename Type, class R>
  inline
  adept::internal::NoAlias<Type, R>
  noalias(const Expression<Type, R>& r) {
    return adept::internal::NoAlias<Type, R>(r.cast());
  }

  template <typename Type>
  inline
  typename internal::enable_if<internal::is_not_expression<Type>::value, Type>::type
  noalias(const Type& r) {
    return r;
  }


  // ---------------------------------------------------------------------
  // SECTION 3.4: Unary operations: transpose function
  // ---------------------------------------------------------------------

  namespace internal {
    /*
    template <typename Type, class R>
    struct Transpose
      : public Expression<Type, Transpose<Type, R> > 
    {
      static const int  rank_      = 2;
      static const bool is_active_ = R::is_active;
      static const int  n_active_  = R::n_active;
      static const int  n_scratch_ = R::n_scratch;
      static const int  n_arrays_  = R::n_arrays;

      const R& arg;

      Transpose(const Expression<Type, R>& arg_)
	: arg(arg_.cast()) { }
      
      template <int Rank>
	bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return arg.get_dimensions(dim);
      }

      std::string expression_string_() const {
	std::string str = "transpose(";
	str += static_cast<const R*>(&arg)->expression_string() + ")";
	return str;
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return arg.is_aliased(mem1,mem2);
      }

      template <int Rank>
      Type value_with_len_(Index i, Index len) const {
	return operation(arg.value_with_len(i, len));
      }
      
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	arg.advance_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return arg.value_at_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return arg.value_at_location_store_<MyArrayNum,MyScratchNum>(loc, 
								     scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	arg.template calc_gradient_<MyArrayNum, MyScratchNum>(stack, loc, 
							      scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch,
		typename MyType>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
	arg.template calc_gradient_<MyArrayNum, MyScratchNum+1>(stack, loc, 
								scratch,
								multiplier);
      }

      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {
	arg.set_location_<MyArrayNum>(i, index);
      }

    }; // End struct Transpose
    */

  }


  // ---------------------------------------------------------------------
  // SECTION 3.5: Unary operations: returning boolean expression
  // ---------------------------------------------------------------------
  namespace internal {

    // Unary operations returning bool derive from this class, where
    // Op is a policy class defining how to implement the operation,
    // and R is the type of the argument of the operation
    template <typename Type, template<class> class Op, class R>
    struct UnaryBoolOperation
      : public Expression<bool, UnaryBoolOperation<Type, Op, R> >,
	protected Op<Type> {
      
      static const int  rank_      = R::rank;
      static const bool is_active_ = false;
      static const int  n_active_ = 0;
      static const int  n_scratch_ = 0;
      static const int  n_arrays_ = R::n_arrays;
      
      using Op<Type>::operation;
      using Op<Type>::operation_string;
      
      const R& arg;

      UnaryBoolOperation(const Expression<Type, R>& arg_)
	: arg(arg_.cast()) { }
      
      template <int Rank>
      bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return arg.get_dimensions(dim);
      }

//       Index get_dimension_with_len(Index len) const {
// 	return arg.get_dimension_with_len_(len);
//       }

      std::string expression_string_() const {
	std::string str;
	str = operation_string();
	str += "(" + static_cast<const R*>(&arg)->expression_string() + ")";
	return str;
      }

      bool is_aliased_(const bool* mem1, const bool* mem2) const {
	return false;
      }

      template <int Rank>
      Type value_with_len_(Index i, Index len) const {
	return operation(arg.value_with_len(i, len));
      }
      
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	arg.advance_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(arg.value_at_location_<MyArrayNum>(loc));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] 
	  = operation(arg.value_at_location_store_<MyArrayNum,MyScratchNum+1>(loc, scratch));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const { }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch,
		typename MyType>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const { }

      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {
	arg.set_location_<MyArrayNum>(i, index);
      }

    };
  
  } // End namespace internal
} // End namespace adept

// Overloads of mathematical functions only works if done in the
// global namespace
#define ADEPT_DEF_UNARY_BOOL_FUNC(NAME, FUNC, RAWFUNC)		\
  namespace adept {						\
    namespace internal {					\
      template <typename Type>					\
      struct NAME  {						\
	const char* operation_string() const { return #FUNC; }	\
	bool operation(const Type& val) const {			\
	  return RAWFUNC(val);					\
	}							\
      };							\
    } /* End namespace internal */				\
  } /* End namespace adept */					\
  template <class Type, class R>					\
  inline								\
  adept::internal::UnaryBoolOperation<Type, adept::internal::NAME, R>	\
  FUNC(const adept::Expression<Type, R>& r){				\
    return adept::internal::UnaryBoolOperation<Type,			\
      adept::internal::NAME, R>(r.cast());				\
  }

ADEPT_DEF_UNARY_BOOL_FUNC(IsNan,    isnan,    std::isnan)
ADEPT_DEF_UNARY_BOOL_FUNC(IsInf,    isinf,    std::isinf)
ADEPT_DEF_UNARY_BOOL_FUNC(IsFinite, isfinite, std::isfinite)

//#undef ADEPT_DEF_UNARY_BOOL_FUNC

  


namespace adept {
  namespace internal {

    // ---------------------------------------------------------------------
    // SECTION 4.1: Binary operations: define BinaryOperation type
    // ---------------------------------------------------------------------

    // Binary operations derive from this class, where Op is a policy
    // class defining how to implement the operation and L and R are
    // the arguments to the operation
    template <class Type, class L, class Op, class R>
    struct BinaryOperation
      : public Expression<Type, BinaryOperation<Type, L, Op, R> >,
	protected Op {

      // Static data
      static const int  rank_ = (L::rank > R::rank ? L::rank : R::rank);
      static const bool is_active_ = (L::is_active || R::is_active) 
	&& !is_same<Type, bool>::value;
      static const int  store_result = is_active_ * Op::store_result;
      static const int  n_active_ = L::n_active + R::n_active;
      // Assume the only local scratch variable is the result of the
      // binary expression
      static const int  n_local_scratch = store_result; 
      //	+ Op::n_scratch<L::is_active,R::is_active>::value
      static const int  n_scratch_ 
        = n_local_scratch + L::n_scratch + R::n_scratch;
      static const int  n_arrays_ = L::n_arrays + R::n_arrays;
      static const VectorOrientation vector_orientation_ 
	= combined_orientation<L::vector_orientation,R::vector_orientation>::value;

      using Op::is_operator;
      using Op::operation;
      using Op::operation_string;
      
      // DATA
      const L& left;
      const R& right;

      BinaryOperation(const Expression<typename L::type, L>& left_,
		      const Expression<typename R::type, R>& right_)
	: left(left_.cast()), right(right_.cast()) { 
      }
      
      template <int Rank>
      bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return my_get_dimensions<L::rank != 0, R::rank != 0>(dim);
      }

//       Index get_dimension_with_len(Index len) const {
// 	Index ldim = left.get_dimension_with_len_(len);
// 	Index rdim = right.get_dimension_with_len_(len);
// 	if (ldim == rdim) {
// 	  // Dimensions match
// 	  return ldim;
// 	}
// 	else if ((ldim < 1 || rdim < 1)
// 		 || (ldim > 1 && rdim > 1)) {
// 	  // Dimensions don't match or there has been 
// 	  return -1;
// 	}
// 	else {
// 	  return ldim > rdim ? ldim : rdim;
// 	}
//       }

    protected:

      template <bool LIsArray, bool RIsArray, int Rank>
      typename enable_if<LIsArray && RIsArray, bool>::type
      my_get_dimensions(ExpressionSize<Rank>& dim) const {
	if (left.get_dimensions(dim)) {
	  ExpressionSize<Rank> right_dim;
	  if (right.get_dimensions(right_dim)
	      && compatible(dim, right_dim)) {
	    return true;
	  }
	}
	return false;
      }

      template <bool LIsArray, bool RIsArray, int Rank>
      typename enable_if<LIsArray && !RIsArray, bool>::type
      my_get_dimensions(ExpressionSize<Rank>& dim) const {
	return left.get_dimensions(dim);
      }

      template <bool LIsArray, bool RIsArray, int Rank>
      typename enable_if<!LIsArray && RIsArray, bool>::type
      my_get_dimensions(ExpressionSize<Rank>& dim) const {
	return right.get_dimensions(dim);
      }

      template <bool LIsArray, bool RIsArray, int Rank>
      typename enable_if<!LIsArray && !RIsArray, bool>::type
      my_get_dimensions(ExpressionSize<Rank>& dim) const {
	return true;
      }

    public:

      std::string expression_string_() const {
	std::string str;
	if (is_operator) {
	  str = "(" + left.expression_string()
	    + operation_string()
	    + right.expression_string() + ")";
	}
	else {
	  str = operation_string();
	  str += "(" + static_cast<const L*>(&left)->expression_string()
	    + "," + static_cast<const R*>(&right)->expression_string() + ")";
	}
	return str;
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return left.is_aliased(mem1, mem2) || right.is_aliased(mem1, mem2);
      }

      Type value_with_len_(const Index& j, const Index& len) const {
	return operation(left.value_with_len(j,len), 
			right.value_with_len(j,len));
      }

      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	left .advance_location_<MyArrayNum>(loc);
	right.advance_location_<MyArrayNum+L::n_arrays>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(left .value_at_location_<MyArrayNum>(loc),
			 right.value_at_location_<MyArrayNum+L::n_arrays>(loc));
      }

      // Adept-1.x did not store for addition and subtraction!
      // Moreover, we should ideally not ask inactive arguments to
      // store their result.
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return my_value_at_location_store_<store_result,MyArrayNum,MyScratchNum>(loc, scratch);
      }

      // Adept-1.x did not store for addition and subtraction!
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return my_value_stored_<store_result,MyArrayNum,MyScratchNum>(loc, scratch);
      }

    protected:
      template <int StoreResult, int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch>
      typename enable_if<StoreResult==1, Type>::type
      my_value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] 
	  = operation(left .value_at_location_store_<MyArrayNum,MyScratchNum+n_local_scratch>(loc, scratch),
		      right.value_at_location_store_<MyArrayNum+L::n_arrays,
						     MyScratchNum+L::n_scratch+n_local_scratch>(loc, scratch));
      }
      // In differentiating "a/b", it helps to store "1/b";
      // "operation_store" is only provided by Divide
      template <int StoreResult, int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch>
      typename enable_if<StoreResult==2, Type>::type
      my_value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] 
	  = Op::operation_store(left .value_at_location_store_<MyArrayNum,MyScratchNum+n_local_scratch>(loc, scratch),
			    right.value_at_location_store_<MyArrayNum+L::n_arrays,
			    MyScratchNum+L::n_scratch+n_local_scratch>(loc, scratch),
			    scratch[MyScratchNum+1]);
      }

      // Adept-1.x did not store for addition and subtraction!
      template <int StoreResult, int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      typename enable_if<(StoreResult > 0), Type>::type
      my_value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch>
      typename enable_if<StoreResult==0, Type>::type
      my_value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	/*
	return operation(left .value_at_location_<MyArrayNum>(loc),
			 right.value_at_location_<MyArrayNum+L::n_arrays>(loc));
	*/
	return operation(left .value_at_location_store_<MyArrayNum,MyScratchNum+n_local_scratch>(loc, scratch),
			 right.value_at_location_store_<MyArrayNum+L::n_arrays,
			 MyScratchNum+L::n_scratch+n_local_scratch>(loc, scratch));
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      typename enable_if<StoreResult==0, Type>::type
      my_value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
	return operation(left .value_at_location_<MyArrayNum>(loc),
			 right.value_at_location_<MyArrayNum+L::n_arrays>(loc));
      }
    

    public:

      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {
	left. set_location_<MyArrayNum>(i, index);
	right.set_location_<MyArrayNum+L::n_arrays>(i, index);
      }


      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
        calc_left_ <MyArrayNum, MyScratchNum>(stack, left,  loc, scratch);
        calc_right_<MyArrayNum, MyScratchNum>(stack, right, loc, scratch);
      }
      // As the previous but multiplying the gradient by "multiplier"
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, typename MyType>
      void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
        calc_left_ <MyArrayNum, MyScratchNum>(stack, left,  loc, scratch, multiplier);
        calc_right_<MyArrayNum, MyScratchNum>(stack, right, loc, scratch, multiplier);
      }
    
    protected:
      // Only calculate gradients for left and right arguments if they
      // are active; otherwise do nothing
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType>
      typename enable_if<LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	Op::template calc_left<MyArrayNum, MyScratchNum>(stack, left, right, loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType>
      typename enable_if<!LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const { }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType>
      typename enable_if<RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	Op::template calc_right<MyArrayNum, MyScratchNum>(stack, left, right, loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType>
      typename enable_if<!RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const { }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType, typename MyType>
      typename enable_if<LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	Op::template calc_left<MyArrayNum, MyScratchNum>(stack, left, right, loc, scratch, multiplier);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType, typename MyType>
      typename enable_if<!LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const { }


      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType, typename MyType>
      typename enable_if<RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	Op::template calc_right<MyArrayNum, MyScratchNum>(stack, left, right, loc, scratch, multiplier);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType, typename MyType>
      typename enable_if<!RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const { }
    };
  

    // ---------------------------------------------------------------------
    // SECTION 4.2: policy classes for BinaryOperation: with scalars
    // ---------------------------------------------------------------------

    // Binary operations with a non-Expression on the left-hand-side
    template <class Type, typename L, class Op, class R>
    struct BinaryOpScalarLeft
      : public Expression<Type, BinaryOpScalarLeft<Type, L, Op, R> >,
	protected Op {

      // Static data
      static const int rank_ = R::rank;
      static const bool is_active_ = R::is_active && !is_same<Type, bool>::value;
      static const int  store_result = is_active_ * Op::store_result;
      static const int n_active_ = R::n_active;
      // Assume the only local scratch variable is the result of the
      // binary expression
      static const int  n_local_scratch = store_result; 
      //	+ Op::n_scratch<L::is_active,R::is_active>::value
      static const int  n_scratch_ 
        = n_local_scratch + R::n_scratch;
      static const int  n_arrays_ = R::n_arrays;

      using Op::is_operator;
      using Op::operation;
      using Op::operation_string;
      
      // DATA
      L left;
      const R& right;

      BinaryOpScalarLeft(L left_,  const Expression<typename R::type, R>& right_)
	: left(left_), right(right_.cast()) { 
      }
      
      template <int Rank>
      bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return right.get_dimensions(dim);
      }

      std::string expression_string_() const {
	std::stringstream s;
	if (is_operator) {
	  s << "(" << left << operation_string()
	    << right.expression_string() << ")";
	}
	else {
	  s << operation_string() << "(" << left << ","
	    << static_cast<const R*>(&right)->expression_string() << ")";
	}
	return s.str();
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return right.is_aliased(mem1, mem2);
      }

      Type value_with_len_(const Index& j, const Index& len) const {
	return operation(left, right.value_with_len(j,len));
      }

      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	right.advance_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(left, right.value_at_location_<MyArrayNum>(loc));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return my_value_at_location_store_<store_result,MyArrayNum,MyScratchNum>(loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return my_value_stored_<store_result,MyArrayNum,MyScratchNum>(loc, scratch);
      }

    protected:
      template <int StoreResult, int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch>
      typename enable_if<StoreResult == 1, Type>::type
      my_value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] = operation(left,
		      right.value_at_location_store_<MyArrayNum, MyScratchNum+n_local_scratch>(loc, scratch));
      }
      template <int StoreResult, int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch>
      typename enable_if<StoreResult == 2, Type>::type
      my_value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] = Op::operation_store(left,
	       right.value_at_location_store_<MyArrayNum, MyScratchNum+n_local_scratch>(loc, scratch),
	       scratch[MyScratchNum+1]);
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      typename enable_if<(StoreResult > 0), Type>::type
      my_value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch>
      typename enable_if<StoreResult == 0, Type>::type
      my_value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	return operation(left,
	     right.value_at_location_store_<MyArrayNum,MyScratchNum+n_local_scratch>(loc, scratch));
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      typename enable_if<StoreResult == 0, Type>::type
      my_value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
	return operation(left,right.value_at_location_<MyArrayNum>(loc));
      }
    

    public:

      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {
	right.set_location_<MyArrayNum>(i, index);
      }


      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
        calc_right_<MyArrayNum, MyScratchNum>(stack, right, loc, scratch);
      }
      // As the previous but multiplying the gradient by "multiplier"
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, typename MyType>
      void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
        calc_right_<MyArrayNum, MyScratchNum>(stack, right, loc, scratch, multiplier);
      }
    
    protected:
      // Only calculate gradients arguments if they are active;
      // otherwise do nothing
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType>
      typename enable_if<RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	Op::template calc_right<MyArrayNum, MyScratchNum>(stack, Scalar<L>(left), right, loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType>
      typename enable_if<!RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const { }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType, typename MyType>
      typename enable_if<RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	Op::template calc_right<MyArrayNum, MyScratchNum>(stack, Scalar<L>(left), right, loc, scratch, multiplier);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType, typename MyType>
      typename enable_if<!RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const { }
    };




    // Binary operations with a non-Expression on the right-hand-side
    template <class Type, typename L, class Op, class R>
    struct BinaryOpScalarRight
      : public Expression<Type, BinaryOpScalarRight<Type, L, Op, R> >,
	protected Op {

      // Static data
      static const int rank_ = L::rank;
      static const bool is_active_ = L::is_active && !is_same<Type,bool>::value;
      static const int  store_result = is_active_ * Op::store_result;
      static const int n_active_ = L::n_active;
      // Assume the only local scratch variable is the result of the
      // binary expression
      static const int  n_local_scratch = store_result; 
      //	+ Op::n_scratch<L::is_active,R::is_active>::value
      static const int  n_scratch_ 
        = n_local_scratch + L::n_scratch;
      static const int  n_arrays_ = L::n_arrays;

      using Op::is_operator;
      using Op::operation;
      using Op::operation_string;
      
      // DATA
      const L& left;
      R right;

      BinaryOpScalarRight(const Expression<typename L::type, L>& left_, R right_)
	: left(left_.cast()), right(right_) { 
      }
      
      template <int Rank>
      bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return left.get_dimensions(dim);
      }

      std::string expression_string_() const {
	std::stringstream s;
	if (is_operator) {
	  s << "(" << left.expression_string() << operation_string()
	    << right << ")";
	}
	else {
	  s << operation_string() << "("
	    << static_cast<const L*>(&left)->expression_string() << ","
	    << right << ")";
	}
	return s.str();
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return left.is_aliased(mem1, mem2);
      }

      Type value_with_len_(const Index& j, const Index& len) const {
	return operation(left.value_with_len(j,len), right);
      }

      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	left.advance_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(left.value_at_location_<MyArrayNum>(loc), right);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return my_value_at_location_store_<store_result,MyArrayNum,MyScratchNum>(loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return my_value_stored_<store_result,MyArrayNum,MyScratchNum>(loc, scratch);
      }

    protected:
      template <int StoreResult, int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch>
      typename enable_if<(StoreResult > 0), Type>::type
      my_value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] = operation(
	   left.value_at_location_store_<MyArrayNum, MyScratchNum+n_local_scratch>(loc, scratch), right);
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      typename enable_if<(StoreResult > 0), Type>::type
      my_value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch>
      typename enable_if<StoreResult == 0, Type>::type
      my_value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	return operation(left.value_at_location_store_<MyArrayNum,MyScratchNum+n_local_scratch>(loc, scratch), 
			 right);
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      typename enable_if<StoreResult == 0, Type>::type
      my_value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
	return operation(left.value_at_location_<MyArrayNum>(loc), right);
      }
    

    public:

      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {
	left.set_location_<MyArrayNum>(i, index);
      }


      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
        calc_left_<MyArrayNum, MyScratchNum>(stack, left, loc, scratch);
      }
      // As the previous but multiplying the gradient by "multiplier"
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, typename MyType>
      void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
        calc_left_<MyArrayNum, MyScratchNum>(stack, left, loc, scratch, multiplier);
      }
    
    protected:
      // Only calculate gradients arguments if they are active;
      // otherwise do nothing
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType>
      typename enable_if<LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	Op::template calc_left<MyArrayNum, MyScratchNum>(stack, left, Scalar<R>(right), loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType>
      typename enable_if<!LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const { }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType, typename MyType>
      typename enable_if<LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	Op::template calc_left<MyArrayNum, MyScratchNum>(stack, left, Scalar<R>(right), loc, scratch, multiplier);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType, typename MyType>
      typename enable_if<!LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const { }
    };
 
  } // End namespace internal




  namespace internal {

    // ---------------------------------------------------------------------
    // SECTION 4.3: policy classes for BinaryOperation: standard operators
    // ---------------------------------------------------------------------

    // Policy class implementing operator+
    struct Add {
      static const bool is_operator  = true;  // Operator or function for expression_string()
      static const int  store_result = 0;     // Do we need any scratch space?

      const char* operation_string() const { return "+"; } // For expression_string()
      
      // Implement the basic operation
      template <class LType, class RType>
      typename promote<LType, RType>::type
      operation(const LType& left, const RType& right) const { return left + right; }
      
      // Calculate the gradient of the left-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch);
      }

      // Calculate the gradient of the right-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
        right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch);
      }

      // Calculate the gradient of the left-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, multiplier);
      }

      // Calculate the gradient of the right-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
        right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, multiplier);
      }
    };

    // Policy class implementing operator-
    struct Subtract {
      static const bool is_operator  = true;  // Operator or function for expression_string()
      static const int  store_result = 1;     // Do we need any scratch space?

      const char* operation_string() const { return "-"; } // For expression_string()
      
      // Implement the basic operation
      template <class LType, class RType>
      typename promote<LType, RType>::type
      operation(const LType& left, const RType& right) const { return left - right; }
      
      // Calculate the gradient of the left-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch);
      }

      // Calculate the gradient of the right-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
        right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, -1.0);
      }

      // Calculate the gradient of the left-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, multiplier);
      }

      // Calculate the gradient of the right-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
        right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, -multiplier);
      }
    };


    // Policy class implementing operator*
    struct Multiply {
      static const bool is_operator  = true; // Operator or function for expression_string()
      static const int  store_result = 1;    // Do we need any scratch space? (this can be 0 or 1)

      const char* operation_string() const { return "*"; } // For expression_string()
      
      // Implement the basic operation
      template <class LType, class RType>
      typename promote<LType, RType>::type
      operation(const LType& left, const RType& right) const { return left * right; }
      
      // Calculate the gradient of the left-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, 
	    right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch));
      }

      // Calculate the gradient of the right-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
        right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, 
				   left.template value_stored_<MyArrayNum,MyScratchNum+store_result>(loc, scratch));
      }

      // Calculate the gradient of the left-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, multiplier
	    *right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch));
      }

      // Calculate the gradient of the right-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
        right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, 
		   multiplier*left.template value_stored_<MyArrayNum,MyScratchNum+store_result>(loc, scratch));
      }
    };

    // Policy class implementing operator/
    struct Divide {
      static const bool is_operator  = true; // Operator or function for expression_string()
      static const int  store_result = 2;    // Do we need any scratch space? (this can be 1 or 2)

      const char* operation_string() const { return "/"; } // For expression_string()
      
      // Implement the basic operation
      template <class LType, class RType>
      typename promote<LType, RType>::type
      operation(const LType& left, const RType& right) const { return left / right; }

      template <class LType, class RType>
      typename promote<LType, RType>::type
      operation_store(const LType& left, const RType& right, Real& one_over_right) const { 
	one_over_right = 1.0 / right;
	return left * one_over_right; 
      }
      
      // Calculate the gradient of the left-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
	// If f(a,b) = a/b then df/da = 1/b
	// If store_result==1 then do this:
        //left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, 
	//    1.0 / right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch));
	// If store_result==2 then do this:
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, 
									    scratch[MyScratchNum+1]);
      }

      // Calculate the gradient of the right-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
	// If f(a,b) = a/b then df/db = -a/(b*b) = -f/b
	// If store_result==1 then do this:
        //right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, 
	//      -scratch[MyScratchNum] / right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch));
	// If store_result==2 then do this:
	right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, 
								      -scratch[MyScratchNum] * scratch[MyScratchNum+1]);
      }

      // Calculate the gradient of the left-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	// If f(a,b) = a/b then w*df/da = w/b
	// If store_result==1 then do this:
        //left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, multiplier
	//    / right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch));
	// If store_result==2 then do this:
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, 
									    multiplier*scratch[MyScratchNum+1]);
      }

      // Calculate the gradient of the right-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	// If f(a,b) = a/b then w*df/db = -w*a/(b*b) = -w*f/b
	// If store_result==1 then do this:
        //right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, 
	//		  -multiplier * scratch[MyScratchNum] 
	//	      / right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch));
	// If store_result==2 then do this:
	right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, 
						      -multiplier * scratch[MyScratchNum] * scratch[MyScratchNum+1]);
      }
    };

    // Policy class implementing function pow
    struct Pow {
      static const bool is_operator  = false; // Operator or function for expression_string()
      static const int  store_result = 1;     // Do we need any scratch space? (this CANNOT be changed)

      const char* operation_string() const { return "pow"; } // For expression_string()
      
      // Implement the basic operation
      template <class LType, class RType>
      typename promote<LType, RType>::type
      operation(const LType& left, const RType& right) const { return pow(left, right); }
      
      // Calculate the gradient of the left-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, 
	   right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch)
	    *pow(left.template value_stored_<MyArrayNum, MyScratchNum+store_result>(loc, scratch),
		 right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch) - 1.0));
      }

      // Calculate the gradient of the right-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
        right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, 
	  scratch[MyScratchNum] * log(left.template value_stored_<MyArrayNum,MyScratchNum+store_result>(loc, scratch)));
      }

      // Calculate the gradient of the left-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
        left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, multiplier
	    *right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch)
	    *pow(left.template value_stored_<MyArrayNum, MyScratchNum+store_result>(loc, scratch),
		 right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch) - 1.0));
      }

      // Calculate the gradient of the right-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
        right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, 
		   multiplier * scratch[MyScratchNum] 
		  * log(left.template value_stored_<MyArrayNum,MyScratchNum+store_result>(loc, scratch)));
      }
    };

    // Policy class implementing function max
    struct Max {
      static const bool is_operator  = false; // Operator or function for expression_string()
      static const int  store_result = 0;    // Do we need any scratch space? (this can be 0 or 1)

      const char* operation_string() const { return "max"; } // For expression_string()
      
      // Implement the basic operation
      template <class LType, class RType>
      typename promote<LType, RType>::type
      operation(const LType& left, const RType& right) const { return left < right ? right : left; }
      
      // Calculate the gradient of the left-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
	if (is_left<MyArrayNum,MyScratchNum>(left,right,loc,scratch)) {
	  left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch);
	}
      }

      // Calculate the gradient of the right-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
	if (!is_left<MyArrayNum,MyScratchNum>(left,right,loc,scratch)) {
	  right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch);
	}
      }

      // Calculate the gradient of the left-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	if (is_left<MyArrayNum,MyScratchNum>(left,right,loc,scratch)) {
	  left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, multiplier);
	}
      }

      // Calculate the gradient of the right-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	if (!is_left<MyArrayNum,MyScratchNum>(left,right,loc,scratch)) {
	  right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, multiplier);
	}
      }

    private:
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      bool is_left(const L& left, const R& right, const ExpressionSize<NArrays>& loc,
		   const ScratchVector<NScratch>& scratch) const {
	return left.template value_stored_<MyArrayNum,MyScratchNum+store_result>(loc, scratch)
	  > right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch);
      }
    };


    // Policy class implementing function min
    struct Min {
      static const bool is_operator  = false; // Operator or function for expression_string()
      static const int  store_result = 0;    // Do we need any scratch space? (this can be 0 or 1)

      const char* operation_string() const { return "min"; } // For expression_string()
      
      // Implement the basic operation
      template <class LType, class RType>
      typename promote<LType, RType>::type
      operation(const LType& left, const RType& right) const { return left < right ? left : right; }
      
      // Calculate the gradient of the left-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
	if (is_left<MyArrayNum,MyScratchNum>(left,right,loc,scratch)) {
	  left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch);
	}
      }

      // Calculate the gradient of the right-hand argument
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch) const {
	if (!is_left<MyArrayNum,MyScratchNum>(left,right,loc,scratch)) {
	  right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch);
	}
      }

      // Calculate the gradient of the left-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_left(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	if (is_left<MyArrayNum,MyScratchNum>(left,right,loc,scratch)) {
	  left.template calc_gradient_<MyArrayNum, MyScratchNum+store_result>(stack, loc, scratch, multiplier);
	}
      }

      // Calculate the gradient of the right-hand argument with a multiplier
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R, typename MyType>
      void calc_right(Stack& stack, const L& left, const R& right, const ExpressionSize<NArrays>& loc,
			       const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	if (!is_left<MyArrayNum,MyScratchNum>(left,right,loc,scratch)) {
	  right.template calc_gradient_<MyArrayNum+L::n_arrays, MyScratchNum+L::n_scratch+store_result>(stack, loc, scratch, multiplier);
	}
      }

    private:
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class L, class R>
      bool is_left(const L& left, const R& right, const ExpressionSize<NArrays>& loc,
		   const ScratchVector<NScratch>& scratch) const {
	return left.template value_stored_<MyArrayNum,MyScratchNum+store_result>(loc, scratch)
	  <= right.template value_stored_<MyArrayNum+L::n_arrays,MyScratchNum+L::n_scratch+store_result>(loc, scratch);
      }
    };


  } // End namespace internal


#define ADEPT_DEFINE_OPERATION(NAME, OPERATOR)				\
  template<class L, class R>						\
  inline								\
  typename internal::enable_if<internal::rank_compatible<L::rank, R::rank>::value, \
			       internal::BinaryOperation<typename internal::promote<typename L::type, \
										    typename R::type>::type, \
							 L, internal:: NAME, R> >::type	\
  OPERATOR(const Expression<typename L::type, L>& l,			\
	   const Expression<typename R::type, R>& r)	{		\
    using namespace adept::internal;					\
    return BinaryOperation<typename promote<typename L::type,		\
					    typename R::type>::type,	\
			   L, NAME, R>(l.cast(), r.cast());		\
  };									\
									\
  template<typename LType, class R>					\
  inline								\
  typename internal::enable_if<internal::is_not_expression<LType>::value, \
			       internal::BinaryOpScalarLeft<typename internal::promote<LType, \
										       typename R::type>::type, \
							    LType, internal:: NAME, R> >::type \
  OPERATOR(const LType& l, const Expression<typename R::type, R>& r)	{ \
    using namespace adept::internal;					\
    return BinaryOpScalarLeft<typename promote<LType, typename R::type>::type, \
      LType, NAME, R>(l, r.cast());					\
  };

#define ADEPT_DEFINE_SCALAR_RHS_OPERATION(NAME, OPERATOR)		\
  template<class L, typename RType>					\
  inline								\
  typename internal::enable_if<internal::is_not_expression<RType>::value, \
			       internal::BinaryOpScalarRight<typename internal::promote<typename L::type, \
											RType>::type, \
							     L, internal:: NAME, RType> >::type \
  OPERATOR(const Expression<typename L::type, L>& l, const RType& r) {	\
    using namespace adept::internal;					\
    return BinaryOpScalarRight<typename promote<typename L::type, RType>::type, \
      L, NAME, RType>(l.cast(), r);		\
  };


  ADEPT_DEFINE_OPERATION(Add, operator+);
  ADEPT_DEFINE_OPERATION(Subtract, operator-);
  ADEPT_DEFINE_OPERATION(Multiply, operator*);
  ADEPT_DEFINE_OPERATION(Divide, operator/);
  ADEPT_DEFINE_OPERATION(Pow, pow);
  ADEPT_DEFINE_OPERATION(Max, max);
  ADEPT_DEFINE_OPERATION(Min, min);

  ADEPT_DEFINE_SCALAR_RHS_OPERATION(Add, operator+);
  ADEPT_DEFINE_SCALAR_RHS_OPERATION(Subtract, operator-);
  ADEPT_DEFINE_SCALAR_RHS_OPERATION(Multiply, operator*);
  ADEPT_DEFINE_SCALAR_RHS_OPERATION(Pow, pow);
  ADEPT_DEFINE_SCALAR_RHS_OPERATION(Max, max);
  ADEPT_DEFINE_SCALAR_RHS_OPERATION(Min, min);

#undef ADEPT_DEFINE_OPERATION
#undef ADEPT_DEFINE_SCALAR_RHS_OPERATION

  // Treat expression divided by scalar differently since this can be
  // changed to a more efficient multiplication
  template<class L, typename RType>
  inline
  typename internal::enable_if<internal::is_not_expression<RType>::value,
			       internal::BinaryOpScalarRight<typename internal::promote<typename L::type,
											RType>::type,
							     L, internal::Multiply, 
							     typename internal::promote<typename L::type,
											RType>::type> >::type
  operator/(const Expression<typename L::type, L>& l, const RType& r) {
    using namespace adept::internal;
    typedef typename promote<typename L::type, RType>::type PType;
    return BinaryOpScalarRight<PType, L, Multiply, PType>(l.cast(), 1.0/static_cast<PType>(r));
  };

// Now the operators returning boolean results

#define ADEPT_DEFINE_OPERATOR(NAME, OPERATOR, OPSYMBOL, OPSTRING)	\
  namespace internal {							\
    struct NAME {							\
      static const bool is_operator  = true;				\
      static const int  store_result = 0;	                        \
      const char* operation_string() const { return OPSTRING; }		\
      									\
      template <class LType, class RType>				\
      bool operation(const LType& left, const RType& right) const	\
      { return left OPSYMBOL right; }					\
    };									\
  }									\
									\
  template<class L, class R>						\
  inline								\
  typename internal::enable_if<internal::rank_compatible<L::rank, R::rank>::value \
			       && (L::rank > 0 || R::rank > 0) ,	\
	    internal::BinaryOperation<bool,L,internal:: NAME, R> >::type \
  OPERATOR(const Expression<typename L::type, L>& l,			\
	   const Expression<typename R::type, R>& r)	{		\
    using namespace adept::internal;					\
    return BinaryOperation<bool, L, NAME, R>(l.cast(), r.cast());	\
  };									\
  									\
  template<typename LType, class R>					\
  inline								\
  typename internal::enable_if<internal::is_not_expression<LType>::value \
			       && (R::rank > 0) ,			\
			       internal::BinaryOpScalarLeft<bool,LType,internal:: NAME, R> >::type \
  OPERATOR(const LType& l, const Expression<typename R::type, R>& r) {	\
    using namespace adept::internal;					\
    return BinaryOpScalarLeft<bool, LType, NAME, R>(l, r.cast());	\
  };									\
  									\
  template<class L, typename RType>					\
  inline								\
  typename internal::enable_if<internal::is_not_expression<RType>::value \
		       && (L::rank > 0),			\
       internal::BinaryOpScalarRight<bool, L, internal:: NAME, RType> >::type \
  OPERATOR(const Expression<typename L::type, L>& l, const RType& r) {	\
    using namespace adept::internal;					\
    return BinaryOpScalarRight<bool, L, NAME, RType>(l.cast(), r);	\
  };									\
									\
  template<class L, class R>						\
  inline								\
  typename internal::enable_if<L::rank == 0 && R::rank == 0,		\
			       bool>::type				\
  OPERATOR(const Expression<typename L::type, L>& l,			\
	   const Expression<typename R::type, R>& r) {			\
    return l.scalar_value() OPSYMBOL r.scalar_value();			\
  };									\
  									\
  template<typename LType, class R>					\
  inline								\
  typename internal::enable_if<internal::is_not_expression<LType>::value \
			       && R::rank == 0, bool>::type		\
  OPERATOR(const LType& l, const Expression<typename R::type, R>& r) {	\
    return l OPSYMBOL r.scalar_value();					\
  };									\
  									\
  template<class L, typename RType>					\
  inline								\
  typename internal::enable_if<internal::is_not_expression<RType>::value \
			       && L::rank == 0, bool>::type		\
  OPERATOR(const Expression<typename L::type, L>& l, const RType& r) {	\
    return l.scalar_value() OPSYMBOL r;					\
  };


// These return bool expressions when applied to expressions of rank
// greater than zero
ADEPT_DEFINE_OPERATOR(GreaterThan, operator>, >, " > ")
ADEPT_DEFINE_OPERATOR(LessThan, operator<, <, " < ")
ADEPT_DEFINE_OPERATOR(GreaterThanEqualTo, operator>=, >=, " >= ")
ADEPT_DEFINE_OPERATOR(LessThanEqualTo, operator<=, <=, " <= ")
ADEPT_DEFINE_OPERATOR(EqualTo, operator==, ==, " == ")
ADEPT_DEFINE_OPERATOR(NotEqualTo, operator!=, !=, " != ")

// These should only work on bool expressions
ADEPT_DEFINE_OPERATOR(Or, operator||, ||, " || ")
ADEPT_DEFINE_OPERATOR(And, operator&&, &&, " && ")

#undef ADEPT_DEFINE_OPERATOR

  /*
  // Conditional operators should behave exactly the same as with
  // non-active arguments so in each of the cases below the value()
  // function is called to extract the value of the expression
#define ADEPT_DEFINE_CONDITIONAL(OPERATOR, OP)			\
  template <class A, class B>					\
  inline							\
  bool OPERATOR(const Expression<A>& a,				\
		const Expression<B>& b) {			\
    return a.value() OP b.value();				\
  }								\
								\
  template <class A>						\
  inline							\
  bool OPERATOR(const Expression<A>& a, const Real& b) {	\
    return a.value() OP b;					\
  }								\
  								\
  template <class B>						\
  inline							\
  bool OPERATOR(const Real& a, const Expression<B>& b) {	\
    return a OP b.value();					\
  }

  ADEPT_DEFINE_CONDITIONAL(operator==, ==)
  ADEPT_DEFINE_CONDITIONAL(operator!=, !=)
  ADEPT_DEFINE_CONDITIONAL(operator>, >)
  ADEPT_DEFINE_CONDITIONAL(operator<, <)
  ADEPT_DEFINE_CONDITIONAL(operator>=, >=)
  ADEPT_DEFINE_CONDITIONAL(operator<=, <=)
  
#undef ADEPT_DEFINE_CONDITIONAL
  */


// 

  template <typename Type, class R>
  inline
  typename internal::enable_if<R::rank == 0,Type>::type
  value(const Expression<Type, R>& r) {
    return r.scalar_value();
  }




} // End namespace adept

#endif // AdeptExpression_H
