/* spread.h -- Spread an array into an additional dimension

    Copyright (C) 2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/
                   
#ifndef AdeptSpread_H
#define AdeptSpread_H

#include <adept/Array.h>

namespace adept {

  namespace internal {
    
    // Expression representing the spread of an array into an
    // additional dimension
    template <int SpreadDim, typename Type, class E>
    class Spread : public Expression<Type, Spread<SpreadDim,Type,E> > {
      typedef Array<E::rank,Type,E::is_active> ArrayType;

    public:
      // Static data
      static const int rank_ = E::rank+1;
      static const bool is_active_ = E::is_active;
      static const int  n_active_ = ArrayType::n_active;
      static const int  n_scratch_ = 0;
      static const int  n_arrays_ = ArrayType::n_arrays;
      // Currently not vectorizable if the final dimension is the
      // spread dimension because the current design always has the
      // array index increasing
      static const bool is_vectorizable_ = (SpreadDim != E::rank);

    protected:
      const ArrayType array;
      ExpressionSize<rank_> dims;
      Index n;

    public:
      Spread(const Expression<Type,E>& e, Index n_)
	: array(e.cast()), n(n_) {
	for (int i = 0; i < SpreadDim; ++i) {
	  dims[i] = array.dimension(i);
	}
	dims[SpreadDim] = n_;
	for (int i = SpreadDim+1; i < rank_; ++i) {
	  dims[i] = array.dimension(i-1);
	}
	// Communicate empty array if n == 0
	if (n_ == 0) {
	  dims[0] = 0;
	}
      }

      bool get_dimensions_(ExpressionSize<rank_>& dim) const {
	dim = dims;
	return true;
      }

      std::string expression_string_() const {
	std::stringstream s;
	s << "spread<" << SpreadDim << ">(" << array.expression_string()
	  << "," << n << ")";
	return s.str();
      }

      bool is_aliased_(const Type* mem1, const Type* mem2) const {
	return false;
      }

      bool all_arrays_contiguous_() const {
	return array.all_arrays_contiguous_();
      }

      template <int n>
      int alignment_offset_() const {
	return array.alignment_offset_<n>();
      }

      // Do not implement value_with_len_

      // Advance only if the spread dimension is not the last
      template <int MyArrayNum, int NArrays>
      void advance_location_(ExpressionSize<NArrays>& loc) const {
	// If false this if statement should be optimized away
	if (SpreadDim < rank_-1) {
	  array.advance_location_<MyArrayNum>(loc);
	}
      }

      template <int MyArrayNum, int NArrays>
      Type value_at_location_(const ExpressionSize<NArrays>& loc) const {
	return array.value_at_location_<MyArrayNum>(loc);
      }
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_at_location_store_(const ExpressionSize<NArrays>& loc,
				    ScratchVector<NScratch>& scratch) const {
	return array.value_at_location_<MyArrayNum>(loc);
      }
      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      Type value_stored_(const ExpressionSize<NArrays>& loc,
			 const ScratchVector<NScratch>& scratch) const {
	return array.value_at_location_<MyArrayNum>(loc);
      }

      template <int MyArrayNum, int NArrays>
      Packet<Type> 
      packet_at_location_(const ExpressionSize<NArrays>& loc) const {
	return packet_at_location_local_<SpreadDim==rank_-1,MyArrayNum>(loc);

      }

    protected:

      // Specializing for the case when the final dimension is the
      // final dimension of the wrapped array
      template <bool IsDuplicate, int MyArrayNum, int NArrays>
      typename enable_if<!IsDuplicate, Packet<Type> >::type
      packet_at_location_local_(const ExpressionSize<NArrays>& loc) const {
	return array.packet_at_location_<MyArrayNum>(loc);
      }

      // Specializing for the case when the final dimension is to be
      // "spread".  The following does not work because the array
      // location is incremented for packets when we really want it to
      // always point to the start of a row.  It is deactivated by
      // is_vectorizable_ (above).
      template <bool IsDuplicate, int MyArrayNum, int NArrays>
      typename enable_if<IsDuplicate, Packet<Type> >::type
      packet_at_location_local_(const ExpressionSize<NArrays>& loc) const {
	return Packet<Type>(array.value_at_location_<MyArrayNum>(loc));
      }
      
    public:

      template <int MyArrayNum, int NArrays>
      void set_location_(const ExpressionSize<rank_>& i, 
			 ExpressionSize<NArrays>& index) const {
	ExpressionSize<rank_-1> i_array(0);
	int j = 0;
	for ( ; j < SpreadDim; ++j) {
	  i_array[j] = i[j];
	}
	for ( ; j < rank_-1; ++j) {
	  i_array[j] = i[j+1];
	}
	array.set_location_<MyArrayNum>(i_array, index);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch>
      void calc_gradient_(Stack& stack, const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch) const {
	array.calc_gradient_<MyArrayNum,MyScratchNum>(stack,loc,scratch);
      }

      template <int MyArrayNum, int MyScratchNum, int NArrays, int NScratch,
		typename MyType>
      void calc_gradient_(Stack& stack, 
			  const ExpressionSize<NArrays>& loc,
			  const ScratchVector<NScratch>& scratch,
			  MyType multiplier) const {
	array.calc_gradient_<MyArrayNum,MyScratchNum>(stack,loc,
						      scratch,multiplier);
      }


    };
    
      
  }

  // Define spread function
  template <int SpreadDim, typename Type, class E>
  typename internal::enable_if<(SpreadDim >= 0 && SpreadDim <= E::rank),
	       internal::Spread<SpreadDim,Type,E> >::type
  spread(const Expression<Type,E>& e, Index n) {
    return internal::Spread<SpreadDim,Type,E>(e,n);
  }

}


#endif
