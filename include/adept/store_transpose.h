/* store_transpose.h -- Store the transpose of a vector of Packets

    Copyright (C) 2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

   Vectorization of active expressions involves storage of the
   gradients in an object of type ScratchVector<N,Packet<Real>>, which
   we need to transpose when placing on the stack.

*/

#ifndef StoreTranspose_H
#define StoreTranspose_H 1

#include <adept/Packet.h>
#include <adept/ScratchVector.h>


namespace adept {

  namespace internal {

    // Unvectorized version
    template <int Len, typename Type>
    store_transpose(ScratchVector<Len,Packet<Type> >& src, Type* dest) {
      for (int i = 0; i < Len; ++i) {
	union {
	  Packet<Type>::intrinsic_type packet;
	  Type[Packet<Type>::size]     array;
	}
	packet = src[i];
	for (int j = 0; j < Packet<Type>::size; ++j) {
	  dest[j*Len] = array[j];
	}
	++dest;
      }
    }

  }
}


#endif
