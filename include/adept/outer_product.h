/* outer_product.h -- Compute the outer product of two vectors

    Copyright (C) 2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/
                             
#ifndef AdeptOuterProduct_H
#define AdeptOuterProduct_H

#include <adept/BinaryOperation.h>
#include <adept/Array.h>

namespace adept {

  namespace internal {

    // Expression representing an outer product
    template <typename Type, typename LType, class L, typename RType, class R>
    class OuterProduct
      : public Expression<Type, OuterProduct<Type,LType,L,RType,R> > {

      typedef Array<1,LType,L::is_active> LArray;
      typedef Array<1,RType,R::is_active> RArray;

    public:
      // Static data
      static const int rank  = 2;
      static const bool is_active  = L::is_active || R::is_active;
      static const int  store_result = is_active;
      static const int  n_active  = LArray::n_active + RArray::n_active;
      static const int  n_local_scratch = store_result; 
      static const int  n_scratch 
        = n_local_scratch + LArray::n_scratch + RArray::n_scratch;
      static const int  n_arrays  = LArray::n_arrays + RArray::n_arrays;
      // Currently not vectorizable because the current design always
      // has the array index increasing
      //      static const bool is_vectorizable = is_same<LType,RType>::value;
      static const bool is_vectorizable = false;//is_same<LType,RType>::value;

    protected:

      // DATA: need to store actual arrays to avoid temporaries going
      // out of scope before they're used; note that if an array is
      // passed in then a shallow copy is made.
      const LArray left;
      const RArray right;
 
    public:

      OuterProduct(const Expression<LType,L>& left_,
		   const Expression<RType,R>& right_) 
	: left(left_.cast()), right(right_.cast()) { }

      bool get_dimensions_(ExpressionSize<2>& dim) const {
	dim[0] = left.size();
	dim[1] = right.size();

	return dim[0] > 0 && dim[1] > 0;
      }

      std::string expression_string_() const {
	return "outer_product(" + left.expression_string() + ","
	  + right.expression_string() + ")";
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return false;
      }

      bool all_arrays_contiguous_() const {
	return right.all_arrays_contiguous_();
      }
 
      bool is_aligned_() const {
	return right.is_aligned_();
      }
      
      template <int n>
      int alignment_offset_() const {
	return right.template alignment_offset_<n>();
      }

      // Do not implement value_with_len_

      // Advance the row only, so the left vector is not advanced
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	right.template advance_location_<MyArrayNum+LArray::n_arrays>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return left.template value_at_location_<MyArrayNum>(loc)
  	    * right.template value_at_location_<MyArrayNum+LArray::n_arrays>(loc);
      }

      // This does not work because the array index is always
      // increased which it shouldn't be for the left vector. For this
      // reason, vectorization is turned off (see is_vectorizable
      // above)
      template <int MyArrayNum, int NArrays>
      Packet<Type> packet_at_location_(const ExpressionSize<NArrays>& loc) const {
	// The LHS of the following multiplication returns a packet
	// containing repeated values of the left vector at one
	// location
	return Packet<Type>(left.template value_at_location_<MyArrayNum>(loc)) // <- fix!
	  * right.template packet_at_location_<MyArrayNum+LArray::n_arrays>(loc);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum] = 
	  left.template value_at_location_store_<MyArrayNum,MyScratchNum+n_local_scratch>(loc, scratch)
	  * right.template value_at_location_store_<MyArrayNum+LArray::n_arrays,
					   MyScratchNum+LArray::n_scratch+n_local_scratch>(loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return scratch[MyScratchNum];
      }
      
      template <int MyArrayNum, int NArrays>
      void set_location_(const ExpressionSize<2>& i, 
			 ExpressionSize<NArrays>& index) const {
	left.template  set_location_<MyArrayNum>(ExpressionSize<1>(i[0]), index);
	right.template set_location_<MyArrayNum+LArray::n_arrays>(ExpressionSize<1>(i[1]), index);
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
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class MyLType>
      typename enable_if<MyLType::is_active,void>::type
      calc_left_(Stack& stack, const MyLType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	Multiply::template calc_left<MyArrayNum, MyScratchNum>(stack, left, right, loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class MyLType>
      typename enable_if<!MyLType::is_active,void>::type
      calc_left_(Stack& stack, const MyLType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const { }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class MyRType>
      typename enable_if<MyRType::is_active,void>::type
      calc_right_(Stack& stack, const MyRType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	Multiply::template calc_right<MyArrayNum, MyScratchNum>(stack, left, right, loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class MyRType>
      typename enable_if<!MyRType::is_active,void>::type
      calc_right_(Stack& stack, const MyRType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const { }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class MyLType, typename MyType>
      typename enable_if<MyLType::is_active,void>::type
      calc_left_(Stack& stack, const MyLType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	Multiply::template calc_left<MyArrayNum, MyScratchNum>(stack, left, right, loc, scratch, multiplier);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class MyLType, typename MyType>
      typename enable_if<!MyLType::is_active,void>::type
      calc_left_(Stack& stack, const MyLType& left, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const { }


      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class MyRType, typename MyType>
      typename enable_if<MyRType::is_active,void>::type
      calc_right_(Stack& stack, const MyRType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const {
	Multiply::template calc_right<MyArrayNum, MyScratchNum>(stack, left, right, loc, scratch, multiplier);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, class MyRType, typename MyType>
      typename enable_if<!MyRType::is_active,void>::type
      calc_right_(Stack& stack, const MyRType& right, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch, MyType multiplier) const { }
    };
   
  }

  // Define outer_product function
  template <typename LType, class L, typename RType, class R>
  internal::OuterProduct<typename promote<LType,RType>::type,LType,L,RType,R>
  outer_product(const Expression<LType,L>& l, const Expression<RType,R>& r) {
    return internal::OuterProduct<typename promote<LType,RType>::type,
				  LType,L,RType,R>(l,r);
  }

}


#endif
