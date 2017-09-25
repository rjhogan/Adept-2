/* eval.h -- Convert expression to array to avoid aliasing issues

    Copyright (C) 2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptEval_H
#define AdeptEval_H

#include <adept/Array.h>

namespace adept {

  // Copy an expression to an Array of the same rank, type and
  // activeness
  template <typename EType, class E>
  typename enable_if<(E::rank > 0), Array<E::rank,EType,E::is_active> >::type
  eval(const Expression<EType,E>& e) {
    Array<E::rank,EType,E::is_active> a;
    a = e.cast();
    return a;
  }

  // Equivalent for scalar expressions; not really needed
  /*
  template <typename EType, class E>
  typename enable_if<E::rank==0 && !E::is_active, EType>::type
  eval(const Expression<EType,E>& e) {
    return static_cast<EType>(e);
  }

  template <typename EType, class E>
  typename enable_if<E::rank==0 && E::is_active, Active<EType> >::type
  eval(const Expression<EType,E>& e) {
    return static_cast<Active<EType> >(e);
  }
  */

} // End namespace adept

#endif
