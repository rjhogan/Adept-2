/* ArrayWrapper.h -- Make Arrays work faster in expressions

    Copyright (C) 2016-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptArrayWrapper_H
#define AdeptArrayWrapper_H 1

#include <adept/Array.h>

namespace adept {
  namespace internal {

    template<int Rank, typename Type, bool IsActive>
    struct ArrayWrapper : public Expression<Type,ArrayWrapper<Rank,Type,IsActive> > {

      typedef Array<Rank,Type,IsActive> MyArray;

      // Static definitions to enable the properties of this type of
      // expression to be discerned at compile time
      static const bool is_active  = IsActive;
      static const bool is_lvalue  = true;
      static const int  rank       = Rank;
      static const int  n_active   = IsActive * (1 + is_complex<Type>::value);
      static const int  n_scratch  = 0;
      static const int  n_arrays   = 1;
      static const bool is_vectorizable = MyArray::is_vectorizable;
      
      ArrayWrapper(const MyArray& a) : data(a.const_data()), array(a) { }
      
      bool get_dimensions_(ExpressionSize<Rank>& dim) const {
	return array.get_dimensions_(dim);
      }
      
      std::string expression_string_() const {
	return std::string("wrapped") + array.expression_string_();
      }
      
      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return array.is_aliased(mem1, mem2);
      }
      
      bool all_arrays_contiguous_() const { 
	return array.all_arrays_contiguous_();
      }
      
      bool is_aligned_() const {
	return array.is_aligned_();
      }
      
      template <int n>
      int alignment_offset_() const {
	return array.template alignment_offset_<n>();
      }
      
      Type value_with_len_(const Index& j, const Index& len) const {
	return array.value_with_len_(j,len);
      }
      
      // Optimize by storing the offset of the fastest-varying dimension?
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	array.template advance_location_<MyArrayNum>(loc);
      }
      
      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return data[loc[MyArrayNum]];
      }
      
      template <int MyArrayNum, int NArrays>
      Packet<Type> packet_at_location_(const ExpressionSize<NArrays>& loc) const {
	return Packet<Type>(data+loc[MyArrayNum]);
      }
      
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				       ScratchVector<NScratch>& scratch) const {
	return data[loc[MyArrayNum]];
      }
      
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
		       const ScratchVector<NScratch>& scratch) const {
	return data[loc[MyArrayNum]];
      }
      
      template <int MyArrayNum, int NArrays>
      void set_location_(const ExpressionSize<Rank>& i, 
			 ExpressionSize<NArrays>& index) const {
	array.template set_location_<MyArrayNum>(i, index);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	array.template calc_gradient_<MyArrayNum,MyScratchNum>(stack, loc, scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch, typename MyType>
      void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
	array.template calc_gradient_<MyArrayNum,MyScratchNum>(stack, loc, scratch, multiplier);
      }
         
      
    protected:
      //      typedef Type __attribute__((aligned(32))) aligned_type;
      Type const * const __restrict data;
      //aligned_type const * const __restrict data;
      const MyArray& __restrict array;
    };
    
    // Unary and binary operations normally contain constant
    // references to their arguments, but if that reference is an
    // Array then the compiler represents this reference as a pointer
    // that must be dereferenced every time a value is extracted from
    // the Array. To speed this up, nested_expression<ExprType>::type
    // is used to obtain the constant reference to ExprType, but for
    // passive Arrays an ArrayWrapper object is returned instead that
    // is faster.
    template <class T>
    struct nested_expression {
      typedef const T& __restrict type;
    };

    template <int Rank, typename Type, bool IsActive>
    struct nested_expression<Array<Rank,Type,IsActive> > {
      typedef const ArrayWrapper<Rank,Type,IsActive> type;
    };

    template <class Type, template<class> class Op, class R>
    struct UnaryOperation;
    template <class Type, class L, class Op, class R>
    struct BinaryOperation;

    // Should we check that rank is > 1?
    template <class Type, template<class> class Op, class R>
    struct nested_expression<UnaryOperation<Type,Op,R> > {
      typedef UnaryOperation<Type,Op,R> type;
    };
    template <class Type, class L, class Op, class R>
    struct nested_expression<BinaryOperation<Type,L,Op,R> > {
      typedef BinaryOperation<Type,L,Op,R> type;
    };
    
  }
}


#endif
