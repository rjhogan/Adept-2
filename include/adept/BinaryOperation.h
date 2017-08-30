/* BinaryOperation.h -- Binary operations on Adept expressions

    Copyright (C) 2014-2016 European Centre for Medium-Range Weather Forecasts

    Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptBinaryOperation_H
#define AdeptBinaryOperation_H

#include <adept/Expression.h>

#include <adept/ArrayWrapper.h>

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
      //      static const VectorOrientation vector_orientation_ 
      //	= combined_orientation<L::vector_orientation,R::vector_orientation>::value;

      static const bool is_vectorizable_
	= L::is_vectorizable && R::is_vectorizable && Op::is_vectorized
	&& is_same<typename L::type,typename R::type>::value
	&& Op::is_operator;

      using Op::is_operator;
      using Op::operation;
      using Op::operation_string;
      
      // DATA
      //const L& left;
      //const R& right;
      const typename nested_expression<L>::type left;
      const typename nested_expression<R>::type right;

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
	ExpressionSize<Rank> right_dim;
	return left.get_dimensions(dim)
	  && right.get_dimensions(right_dim)
	  && compatible(dim, right_dim);
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
	  //	  str += "(" + static_cast<const L*>(&left)->expression_string()
	  //	    + "," + static_cast<const R*>(&right)->expression_string() + ")";
	  str += "(" + left.expression_string()
	    + "," + right.expression_string() + ")";
	}
	return str;
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return left.is_aliased(mem1, mem2) || right.is_aliased(mem1, mem2);
      }
      bool all_arrays_contiguous_() const { 
	return left.all_arrays_contiguous_()
	  &&  right.all_arrays_contiguous_();
      }
      template <int n>
      int alignment_offset_() const {
	int l = left.alignment_offset_<n>();
	int r = right.alignment_offset_<n>();
	if (l == r) {
	  return l;
	}
	else if (l == n) {
	  return r;
	} else if (r == n) {
	  return l;
	}
	else {
	  return -1;
	}
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
      template <int MyArrayNum, int NArrays>
      Packet<Type> packet_at_location_(const ExpressionSize<NArrays>& loc) const {
	/*
	std::cout << left .packet_at_location_<MyArrayNum>(loc)
		  << operation_string()
		  << right.packet_at_location_<MyArrayNum+L::n_arrays>(loc)
		  << "=" << operation(left .packet_at_location_<MyArrayNum>(loc),
				      right.packet_at_location_<MyArrayNum+L::n_arrays>(loc))
		  << "\n";
	*/
	return operation(left .packet_at_location_<MyArrayNum>(loc),
			 right.packet_at_location_<MyArrayNum+L::n_arrays>(loc));
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
      static const bool is_vectorizable_ = R::is_vectorizable && Op::is_vectorized;

      using Op::is_operator;
      using Op::operation;
      using Op::operation_string;
      
      // DATA
      Packet<L> left;
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
	  s << "(" << left.value << operation_string()
	    << right.expression_string() << ")";
	}
	else {
	  s << operation_string() << "(" << left.value << ","
	    << static_cast<const R*>(&right)->expression_string() << ")";
	}
	return s.str();
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return right.is_aliased(mem1, mem2);
      }
      bool all_arrays_contiguous_() const {
	return right.all_arrays_contiguous_(); 
      }
      template <int n>
      int alignment_offset_() const { return right.alignment_offset_<n>(); }

      Type value_with_len_(const Index& j, const Index& len) const {
	return operation(left.value, right.value_with_len(j,len));
      }

      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	right.advance_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(left.value, right.value_at_location_<MyArrayNum>(loc));
      }
      template <int MyArrayNum, int NArrays>
      Packet<Type> packet_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(left, 
			 right.packet_at_location_<MyArrayNum>(loc));
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
	return scratch[MyScratchNum] = operation(left.value,
		      right.value_at_location_store_<MyArrayNum, MyScratchNum+n_local_scratch>(loc, scratch));
      }
      template <int StoreResult, int MyArrayNum, int MyScratchNum, 
		int NArrays, int NScratch>
      typename enable_if<StoreResult == 2, Type>::type
      my_value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] = Op::operation_store(left.value,
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
	return operation(left.value,
	     right.value_at_location_store_<MyArrayNum,MyScratchNum+n_local_scratch>(loc, scratch));
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      typename enable_if<StoreResult == 0, Type>::type
      my_value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
	return operation(left.value,right.value_at_location_<MyArrayNum>(loc));
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
	Op::template calc_right<MyArrayNum, MyScratchNum>(stack, Scalar<L>(left.value), right, loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType>
      typename enable_if<!RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const { }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class RType, typename MyType>
      typename enable_if<RType::is_active,void>::type
      calc_right_(Stack& stack, const RType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	Op::template calc_right<MyArrayNum, MyScratchNum>(stack, Scalar<L>(left.value), right, loc, scratch, multiplier);
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
      static const bool is_vectorizable_ = L::is_vectorizable && Op::is_vectorized;

      using Op::is_operator;
      using Op::operation;
      using Op::operation_string;
      
      // DATA
      const L& left;
      Packet<R> right;

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
	    << right.value << ")";
	}
	else {
	  s << operation_string() << "("
	    << static_cast<const L*>(&left)->expression_string() << ","
	    << right.value << ")";
	}
	return s.str();
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return left.is_aliased(mem1, mem2);
      }
      bool all_arrays_contiguous_() const {
	return left.all_arrays_contiguous_(); 
      }
      template <int n>
      int alignment_offset_() const { return left.alignment_offset_<n>(); }

      Type value_with_len_(const Index& j, const Index& len) const {
	return operation(left.value_with_len(j,len), right.value);
      }

      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	left.advance_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return operation(left.value_at_location_<MyArrayNum>(loc), right.value);
      }
      template <int MyArrayNum, int NArrays>
      Packet<Type> packet_at_location_(const ExpressionSize<NArrays>& loc) const {
	/*
	std::cout << left.packet_at_location_<MyArrayNum>(loc)
		  << operation_string()
		  << Packet<Type>(right)
		  << "=" << operation(left.packet_at_location_<MyArrayNum>(loc),
			 Packet<Type>(right))
		  << "\n";
	*/
	return operation(left.packet_at_location_<MyArrayNum>(loc),
			 right);
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
	   left.value_at_location_store_<MyArrayNum, MyScratchNum+n_local_scratch>(loc, scratch), right.value);
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
			 right.value);
      }

      template <int StoreResult, int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      typename enable_if<StoreResult == 0, Type>::type
      my_value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
	return operation(left.value_at_location_<MyArrayNum>(loc), right.value);
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
	Op::template calc_left<MyArrayNum, MyScratchNum>(stack, left, Scalar<R>(right.value), loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType>
      typename enable_if<!LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const { }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class LType, typename MyType>
      typename enable_if<LType::is_active,void>::type
      calc_left_(Stack& stack, const LType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	Op::template calc_left<MyArrayNum, MyScratchNum>(stack, left, Scalar<R>(right.value), loc, scratch, multiplier);
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
      static const bool is_vectorized = true;

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
      static const bool is_vectorized = true;

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
      static const bool is_vectorized = true;

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
      static const bool is_vectorized = true;

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
      static const bool is_vectorized = false;

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
      static const bool is_vectorized = false;

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
      static const bool is_vectorized = false;

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
      static const bool is_vectorized = false;				\
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


#endif