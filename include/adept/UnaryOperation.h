/* UnaryOperation.h -- Unary operations on Adept expressions

    Copyright (C) 2014-2017 European Centre for Medium-Range Weather Forecasts

    Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptUnaryOperation_H
#define AdeptUnaryOperation_H

#include <adept/Expression.h>

#include <adept/ArrayWrapper.h>

namespace adept {

  namespace internal {

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
      
      static const int  rank       = R::rank;
      static const bool is_active  = R::is_active && !is_same<Type,bool>::value;
      static const int  n_active   = R::n_active;
      // FIX! Only store if active and if needed
      static const int  n_scratch  = 1 + R::n_scratch;
      static const int  n_arrays   = R::n_arrays;
      // Will need to modify this for sqrt:
      static const bool is_vectorizable
	= Op<Type>::is_vectorized && R::is_vectorizable;

      using Op<Type>::operation;
      using Op<Type>::operation_string;
      using Op<Type>::derivative;
      
      //const R& arg;
      typename nested_expression<R>::type arg;

      UnaryOperation(const Expression<Type, R>& arg_)
	: arg(arg_.cast()) { }
      
      template <int Rank>
      bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return arg.get_dimensions(dim);
      }

      std::string expression_string_() const {
	std::string str;
	str = operation_string();
	str += "(" + arg.expression_string() + ")";
	return str;
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return arg.is_aliased(mem1, mem2);
      }
      bool all_arrays_contiguous_() const {
	return arg.all_arrays_contiguous_();
      }
       bool is_aligned_() const {
	return arg.is_aligned_();
      }
      template <int n>
      int alignment_offset_() const { return arg.template alignment_offset_<n>(); }

      template <int Rank>
      Type value_with_len_(Index i, Index len) const {
	return operation(arg.value_with_len(i, len));
      }
      
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	arg.template advance_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(arg.template value_at_location_<MyArrayNum>(loc));
      }

      template <int MyArrayNum, int NArrays>
      Packet<Type> packet_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(arg.template packet_at_location_<MyArrayNum>(loc));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] 
	  = operation(arg.template value_at_location_store_<MyArrayNum,MyScratchNum+1>(loc, scratch));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <bool IsAligned,	int MyArrayNum, typename PacketType,
	int NArrays>
      PacketType values_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(arg.template values_at_location_<IsAligned,MyArrayNum,PacketType>(loc));
      }

      template <bool UseStored, bool IsAligned,	int MyArrayNum, int MyScratchNum,
		typename PacketType, int NArrays, int NScratch>
      typename enable_if<!UseStored,PacketType>::type
      values_at_location_store_(const ExpressionSize<NArrays>& loc,
				ScratchVector<NScratch,PacketType>& scratch) const {
	return scratch[MyScratchNum]
	  = operation(arg.template values_at_location_store_<UseStored,IsAligned,
		      MyArrayNum,MyScratchNum+1>(loc, scratch));
      }
      template <bool UseStored, bool IsAligned,	int MyArrayNum, int MyScratchNum,
		typename PacketType, int NArrays, int NScratch>
      typename enable_if<UseStored,PacketType>::type
      values_at_location_store_(const ExpressionSize<NArrays>& loc,
				ScratchVector<NScratch,PacketType>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	arg.template calc_gradient_<MyArrayNum, MyScratchNum+1>(stack, loc, scratch,
		derivative(arg.template value_stored_<MyArrayNum,MyScratchNum+1>(loc, scratch),
			   scratch[MyScratchNum]));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch,
		typename MyType>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
	arg.template calc_gradient_<MyArrayNum, MyScratchNum+1>(stack, loc, scratch,
		multiplier*derivative(arg.template value_stored_<MyArrayNum,MyScratchNum+1>(loc, scratch), 
				      scratch[MyScratchNum]));
      }

      template <bool IsAligned, int MyArrayNum, int MyScratchNum, int MyActiveNum,
		int NArrays, int NScratch, int NActive>
      void calc_gradient_packet_(Stack& stack, 
				 const ExpressionSize<NArrays>& loc,
				 const ScratchVector<NScratch,Packet<Real> >& scratch,
				 ScratchVector<NActive,Packet<Real> >& gradients) const {
	arg.template calc_gradient_packet_<IsAligned,MyArrayNum,MyScratchNum+1,
					   MyActiveNum>(stack, loc, scratch, gradients,
		derivative(arg.template values_at_location_store_<true,IsAligned,MyArrayNum,MyScratchNum+1,
			   MyActiveNum>(loc, scratch), scratch[MyScratchNum]));
      }

      template <bool IsAligned, int MyArrayNum, int MyScratchNum, int MyActiveNum,
		int NArrays, int NScratch, int NActive, typename MyType>
      void calc_gradient_packet_(Stack& stack, 
				 const ExpressionSize<NArrays>& loc,
				 const ScratchVector<NScratch,Packet<Real> >& scratch,
				 ScratchVector<NActive,Packet<Real> >& gradients,
				 const MyType& multiplier) const {
	arg.template calc_gradient_packet_<IsAligned,MyArrayNum,MyScratchNum+1,
					   MyActiveNum>(stack, loc, scratch, gradients,
		multiplier*derivative(arg.template values_at_location_store_<true,IsAligned,MyArrayNum,MyScratchNum+1,
				      MyActiveNum>(loc, scratch), scratch[MyScratchNum]));
      }


      template <int MyArrayNum, int Rank, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {
	arg.template set_location_<MyArrayNum>(i, index);
      }

    }; // End UnaryOperation type
  
  } // End namespace internal

  // ---------------------------------------------------------------------
  // SECTION 3.2: Unary operations: define specific operations
  // ---------------------------------------------------------------------

  // We may place the overloaded mathematical functions in the global
  // namespace provided that a using declaration enables the std::
  // version of the function to be located
#define ADEPT_DEF_UNARY_FUNC(NAME, FUNC, RAWFUNC, STRING, DERIVATIVE,	\
			     ISVEC)					\
  namespace internal {							\
    template <typename Type>						\
    struct NAME  {							\
      static const bool is_operator = false;				\
      static const bool is_vectorized = ISVEC;				\
      const char* operation_string() const { return STRING; }		\
      template <typename T>						\
      T operation(const T& val) const {					\
	using RAWFUNC;							\
	return FUNC(val);						\
      }									\
      Type derivative(const Type& val, const Type& result) const {	\
	using std::sin;							\
	using std::cos;							\
	using std::sqrt;						\
	using std::cosh;						\
	using std::sinh;						\
	using std::exp;							\
	return DERIVATIVE;						\
      }									\
      Type fast_sqr(Type val) const { return val*val; }			\
    };									\
  } /* End namespace internal */					\
  template <class Type, class R>					\
  inline								\
  adept::internal::UnaryOperation<Type, adept::internal::NAME, R>	\
  FUNC(const adept::Expression<Type, R>& r)	{			\
    return adept::internal::UnaryOperation<Type,			\
				   adept::internal::NAME, R>(r.cast()); \
  }

  // Functions y(x) whose derivative depends on the argument of the
  // function, i.e. dy(x)/dx = f(x)
  ADEPT_DEF_UNARY_FUNC(Log,   log,   std::log,   "log",   1.0/val, false)
  ADEPT_DEF_UNARY_FUNC(Log10, log10, std::log10, "log10", 0.43429448190325182765/val, false)
  ADEPT_DEF_UNARY_FUNC(Sin,   sin,   std::sin,   "sin",   cos(val), false)
  ADEPT_DEF_UNARY_FUNC(Cos,   cos,   std::cos,   "cos",   -sin(val), false)
  ADEPT_DEF_UNARY_FUNC(Tan,   tan,   std::tan,   "tan",   1.0/fast_sqr(cos(val)), false)
  ADEPT_DEF_UNARY_FUNC(Asin,  asin,  std::asin,  "asin",  1.0/sqrt(1.0-val*val), false)
  ADEPT_DEF_UNARY_FUNC(Acos,  acos,  std::acos,  "acos",  -1.0/sqrt(1.0-val*val), false)
  ADEPT_DEF_UNARY_FUNC(Atan,  atan,  std::atan,  "atan",  1.0/(1.0+val*val), false)
  ADEPT_DEF_UNARY_FUNC(Sinh,  sinh,  std::sinh,  "sinh",  cosh(val), false)
  ADEPT_DEF_UNARY_FUNC(Cosh,  cosh,  std::cosh,  "cosh",  sinh(val), false)
  ADEPT_DEF_UNARY_FUNC(Abs,   abs,   std::abs, "abs", ((val>0.0)-(val<0.0)), false)
  ADEPT_DEF_UNARY_FUNC(Fabs,  fabs,  std::fabs, "fabs", ((val>0.0)-(val<0.0)), false)

  // Functions y(x) whose derivative depends on the result of the
  // function, i.e. dy(x)/dx = f(y)
  ADEPT_DEF_UNARY_FUNC(Exp,   exp,   std::exp,   "exp",   result, false)
  ADEPT_DEF_UNARY_FUNC(Sqrt,  sqrt,  std::sqrt,  "sqrt",  0.5/result, true)
  ADEPT_DEF_UNARY_FUNC(Tanh,  tanh,  std::tanh,  "tanh",  1.0 - result*result, false)

  // Functions with zero derivative
  ADEPT_DEF_UNARY_FUNC(Ceil,  ceil,  std::ceil,  "ceil",  0.0, false)
  ADEPT_DEF_UNARY_FUNC(Floor, floor, std::floor, "floor", 0.0, false)
  
  // Functions defined in the std namespace in C++11 but only in the
  // global namespace before that
#ifdef ADEPT_CXX11_FEATURES
  ADEPT_DEF_UNARY_FUNC(Log2,  log2,  std::log2,  "log2",  1.44269504088896340737/val, false)
  ADEPT_DEF_UNARY_FUNC(Expm1, expm1, std::expm1, "expm1", exp(val), false)
  ADEPT_DEF_UNARY_FUNC(Exp2,  exp2,  std::exp2,  "exp2",  0.6931471805599453094172321214581766*result, false)
  ADEPT_DEF_UNARY_FUNC(Log1p, log1p, std::log1p, "log1p", 1.0/(1.0+val), false)
  ADEPT_DEF_UNARY_FUNC(Asinh, asinh, std::asinh, "asinh", 1.0/sqrt(val*val+1.0), false)
  ADEPT_DEF_UNARY_FUNC(Acosh, acosh, std::acosh, "acosh", 1.0/sqrt(val*val-1.0), false)
  ADEPT_DEF_UNARY_FUNC(Atanh, atanh, std::atanh, "atanh", 1.0/(1.0-val*val), false)
  ADEPT_DEF_UNARY_FUNC(Erf,   erf,   std::erf,   "erf",   1.12837916709551*exp(-val*val), false)
  ADEPT_DEF_UNARY_FUNC(Erfc,  erfc,  std::erfc,  "erfc",  -1.12837916709551*exp(-val*val), false)
  ADEPT_DEF_UNARY_FUNC(Cbrt,  cbrt,  std::cbrt,  "cbrt",  (1.0/3.0)/(result*result), false)
  ADEPT_DEF_UNARY_FUNC(Round, round, std::round, "round", 0.0, false)
  ADEPT_DEF_UNARY_FUNC(Trunc, trunc, std::trunc, "trunc", 0.0, false)
  ADEPT_DEF_UNARY_FUNC(Rint,  rint,  std::rint,  "rint",  0.0, false)
  ADEPT_DEF_UNARY_FUNC(Nearbyint,nearbyint,std::nearbyint,"nearbyint",0.0, false)
#else
  ADEPT_DEF_UNARY_FUNC(Log2,  log2,  ::log2,  "log2",  1.44269504088896340737/val, false)
  ADEPT_DEF_UNARY_FUNC(Expm1, expm1, ::expm1, "expm1", exp(val), false)
  ADEPT_DEF_UNARY_FUNC(Exp2,  exp2,  ::exp2,  "exp2",  0.6931471805599453094172321214581766*result, false)
  ADEPT_DEF_UNARY_FUNC(Log1p, log1p, ::log1p, "log1p", 1.0/(1.0+val), false)
  ADEPT_DEF_UNARY_FUNC(Asinh, asinh, ::asinh, "asinh", 1.0/sqrt(val*val+1.0), false)
  ADEPT_DEF_UNARY_FUNC(Acosh, acosh, ::acosh, "acosh", 1.0/sqrt(val*val-1.0), false)
  ADEPT_DEF_UNARY_FUNC(Atanh, atanh, ::atanh, "atanh", 1.0/(1.0-val*val), false)
  ADEPT_DEF_UNARY_FUNC(Erf,   erf,   ::erf,   "erf",   1.12837916709551*exp(-val*val), false)
  ADEPT_DEF_UNARY_FUNC(Erfc,  erfc,  ::erfc,  "erfc",  -1.12837916709551*exp(-val*val), false)
  ADEPT_DEF_UNARY_FUNC(Cbrt,  cbrt,  ::cbrt,  "cbrt",  (1.0/3.0)/(result*result), false)
  ADEPT_DEF_UNARY_FUNC(Round, round, ::round, "round", 0.0, false)
  ADEPT_DEF_UNARY_FUNC(Trunc, trunc, ::trunc, "trunc", 0.0, false)
  ADEPT_DEF_UNARY_FUNC(Rint,  rint,  ::rint,  "rint",  0.0, false)
  ADEPT_DEF_UNARY_FUNC(Nearbyint,nearbyint,::nearbyint,"nearbyint",0.0, false)
#endif

  //#undef ADEPT_DEF_UNARY_FUNC

#define ADEPT_DEF_UNARY_OP(NAME, FUNC, RAWFUNC, STRING, DERIVATIVE,	\
			   ISVEC)					\
  namespace internal {							\
    template <typename Type>						\
    struct NAME  {							\
      static const bool is_operator = false;				\
      static const bool is_vectorized = ISVEC;				\
      const char* operation_string() const { return STRING; }		\
      template <typename T>						\
      T operation(const T& val) const {					\
	return RAWFUNC(val);						\
      }									\
      Type derivative(const Type& val, const Type& result) const {	\
	return DERIVATIVE;						\
      }									\
      Type fast_sqr(Type val) { return val*val; }			\
    };									\
  } /* End namespace internal */					\
  template <class Type, class R>					\
  inline								\
  adept::internal::UnaryOperation<Type, adept::internal::NAME, R>	\
  FUNC(const adept::Expression<Type, R>& r)	{			\
    return adept::internal::UnaryOperation<Type,			\
				   adept::internal::NAME, R>(r.cast()); \
  }
  
  // Operators
  ADEPT_DEF_UNARY_OP(UnaryPlus,  operator+, +, "+", 1.0, true)
  ADEPT_DEF_UNARY_OP(UnaryMinus, operator-, -, "-", -1.0, true)
  ADEPT_DEF_UNARY_OP(Not,        operator!, !, "!", 0.0, false)


  // ---------------------------------------------------------------------
  // SECTION 3.4: Unary operations: transpose function [DELETED]
  // ---------------------------------------------------------------------

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
      
      static const int  rank       = R::rank;
      static const bool is_active  = false;
      static const int  n_active   = 0;
      static const int  n_scratch  = 0;
      static const int  n_arrays   = R::n_arrays;
      
      using Op<Type>::operation;
      using Op<Type>::operation_string;
      
      const R& arg;

      UnaryBoolOperation(const Expression<Type, R>& arg_)
	: arg(arg_.cast()) { }
      
      template <int Rank>
      bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return arg.get_dimensions(dim);
      }

      std::string expression_string_() const {
	std::string str;
	str = operation_string();
	str += "(" + static_cast<const R*>(&arg)->expression_string() + ")";
	return str;
      }

      bool is_aliased_(const bool* mem1, const bool* mem2) const {
	return false;
      }
      bool all_arrays_contiguous_() const {
	return arg.all_arrays_contiguous_(); 
      }
      template <int n>
      int alignment_offset_() const { return arg.template alignment_offset_<n>(); }

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
	return operation(arg.template value_at_location_<MyArrayNum>(loc));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] 
	  = operation(arg.template value_at_location_store_<MyArrayNum,MyScratchNum+1>(loc, scratch));
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }

      template <bool IsAligned,	int MyArrayNum, typename PacketType,
	int NArrays>
      PacketType values_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(arg.template values_at_location_<IsAligned,MyArrayNum,PacketType>(loc));
      }

      template <bool UseStored, bool IsAligned,	int MyArrayNum, int MyScratchNum,
		typename PacketType, int NArrays, int NScratch>
      PacketType values_at_location_store_(const ExpressionSize<NArrays>& loc,
		   ScratchVector<NScratch,PacketType>& scratch) const {
	return operation(arg.template values_at_location_<IsAligned,MyArrayNum,PacketType>(loc));
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
			 ExpressionSize<NArrays>& index) const {
	arg.set_location_<MyArrayNum>(i, index);
      }

    };
  
  } // End namespace internal

#define ADEPT_DEF_UNARY_BOOL_FUNC(NAME, FUNC, RAWFUNC)		\
  namespace internal {						\
    template <typename Type>					\
    struct NAME  {						\
      const char* operation_string() const { return #FUNC; }	\
      bool operation(const Type& val) const {			\
	using RAWFUNC;						\
	return FUNC(val); /* RAWFUNC(val); */			\
      }								\
    };								\
  } /* End namespace internal */					\
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

} /* End namespace adept */



#endif
