/* noalias.h -- Wrap an expression so that alias checking is not performed

    Copyright (C) 2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptNoalias_H
#define AdeptNoalias_H

#include <adept/Expression.h>

namespace adept {

  namespace internal {

    // No-alias wrapper for enabling noalias()
    template <typename Type, class R>
    struct NoAlias
      : public Expression<Type, NoAlias<Type, R> > 
    {
      static const int  rank       = R::rank;
      static const bool is_active  = R::is_active;
      static const int  n_active   = R::n_active;
      static const int  n_scratch  = R::n_scratch;
      static const int  n_arrays   = R::n_arrays;
      static const bool is_vectorizable = R::is_vectorizable;

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
      bool all_arrays_contiguous_() const {
	return arg.all_arrays_contiguous_(); 
      }
 
      bool is_aligned_() const {
	return arg.is_aligned_();
      } 
     
      template <int n>
      int alignment_offset_() const { 
        return arg.template alignment_offset_<n>();
      }

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
	return arg.template value_at_location_<MyArrayNum>(loc);
      }
      template <int MyArrayNum, int NArrays>
      Packet<Type> packet_at_location_(const ExpressionSize<NArrays>& loc) const {
	return arg.template packet_at_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return arg.template value_at_location_store_<MyArrayNum,MyScratchNum>(loc, 
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
	arg.template set_location_<MyArrayNum>(i, index);
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

}

#endif
