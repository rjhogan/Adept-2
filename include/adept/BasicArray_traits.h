/* BasicArray_traits.h -- traits for the BasicArray class

    Copyright (C) 2020- European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/


#ifndef AdeptBasicArrayTraits_H
#define AdeptBasicArrayTraits_H 1

#include <array>
#include <cstddef>
#include <type_traits>

#include <adept/Expression.h>
#include <adept/traits.h>

namespace adept {

  namespace internal {

    typedef int rank_type;
    typedef unsigned int options_type;
    //typedef std::size_t size_type;
    //typedef std::ptrdiff_t size_type;
    //typedef std::ptrdiff_t difference_type;
    typedef int size_type;
    typedef int difference_type;
    typedef Index gradient_index_type;

    // narray has a template argument treated as a bitmask
    static const options_type ARRAY_IS_REF            = (1 << 0);
    static const options_type ARRAY_IS_PARTIALLY_CONTIGUOUS = (1 << 1);
    static const options_type ARRAY_IS_ALL_CONTIGUOUS = (1 << 2);
    static const options_type ARRAY_IS_COLUMN_MAJOR   = (1 << 3);
    static const options_type ARRAY_IS_ACTIVE         = (1 << 7);
    
    constexpr bool opt_is_ref(options_type options)
    { return options & ARRAY_IS_REF; }
    constexpr bool opt_is_partially_contiguous(options_type options)
    { return options & ARRAY_IS_PARTIALLY_CONTIGUOUS; }
    constexpr bool opt_is_all_contiguous(options_type options)
    { return options & ARRAY_IS_ALL_CONTIGUOUS; }
    constexpr bool opt_is_column_major(options_type options)
    { return options & ARRAY_IS_COLUMN_MAJOR; }
    constexpr bool opt_is_active(options_type options)
    { return options & ARRAY_IS_ACTIVE; }


    template <typename... T> struct all_scalar_indices {
      static const bool value = (is_scalar_int<T>::value && ...);
    };
    /*
    template <typename T0, typename... T> struct all_scalar_indices {
      static const bool value = is_scalar_int<T0>::value & all_scalar_indices<T>::value;
    };
    template <typename T0> struct all_scalar_indices {
      static const bool value = is_scalar_int<T0>::value;
    };
    */
    
  };

};


namespace adept {
  
  template <typename T, std::size_t Len>
  std::ostream& operator<<(std::ostream& os,
			   const std::array<T,Len>& vec) {
      if constexpr (Len > 0) {
        os << "{" << vec[0];
	for (std::size_t i = 1; i < Len; ++i) {
	  os << "," << vec[i];
	}
	return os << "}";
      }
      else {
	return os;
      }
    }
    

  
};

#endif
