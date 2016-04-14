/* Packet.h -- Vectorization support

    Copyright (C) 2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

   A Packet contains a short vector of values, and when it is used in
   a limited set of arithmetic operations, the appropriate vector
   instructions will be used.  For example if your hardware and
   compiler support SSE2 then Packet<float> is a vector of 4x 4-byte
   floats while Packet<double> is a vector of 2x 8-byte floats.
*/

#ifndef AdeptPacket_H
#define AdeptPacket_H 1

#ifdef __SSE2__
#include  "mmintrin.h"
#include "emmintrin.h"
#endif

namespace adept {

  // The default packet contains only one value
  template <typename T>
  struct Packet {
    typedef T intrinsic_type;
    static const int n = 1;
    static const bool is_vectorized = false;
    Packet(const T* d) : data(*d) { }
    Packet(T d) : data(d) { }
    void get(T* d) const { *d = data; }
    void operator=(const T& __restrict d) { data=d; }
    T data;      
  };

  // Define a specialization, and the basic mathematical operators
  // supported in hardware
#define ADEPT_DEF_PACKET_TYPE(TYPE, INT_TYPE, LEN,	\
			      LOAD, SET1, STORE,	\
			      ADD, SUB, MUL, DIV, SQRT) \
  template <> struct Packet<TYPE> {			\
    typedef INT_TYPE intrinsic_type;			\
    static const int n = LEN;				\
    static const bool is_vectorized = true;		\
    Packet(const TYPE* d) : data(LOAD(d)) { }		\
    Packet(TYPE d)  : data(SET1(d)) { }			\
    void get(TYPE* d) const { STORE(d, data); }		\
    void operator=(const INT_TYPE& __restrict d) { data=d; }	\
    INT_TYPE data;					\
  };							\
  inline						\
  INT_TYPE operator+(const Packet<TYPE>& __restrict x,	\
		     const Packet<TYPE>& __restrict y)	\
  { return ADD(x.data,y.data); }			\
  inline						\
  INT_TYPE operator-(const Packet<TYPE>& __restrict x,	\
		     const Packet<TYPE>& __restrict y)	\
  { return SUB(x.data,y.data); }			\
  inline						\
  INT_TYPE operator*(const Packet<TYPE>& __restrict x,	\
		     const Packet<TYPE>& __restrict y)	\
  { return MUL(x.data,y.data); }			\
  inline						\
  INT_TYPE operator/(const Packet<TYPE>& __restrict x,	\
		     const Packet<TYPE>& __restrict y)	\
  { return DIV(x.data,y.data); }			\
  inline						\
  INT_TYPE sqrt(const Packet<TYPE>& x)			\
  { return SQRT(x.data); }

#ifdef __SSE2__
  // SSE2 deals with 16-byte vectors, which may contain 4x 4-byte
  // floats...
  ADEPT_DEF_PACKET_TYPE(float, __m128, 4, 
			_mm_load_ps, _mm_set1_ps, _mm_store_ps,
			_mm_add_ps, _mm_sub_ps,
			_mm_mul_ps, _mm_div_ps, _mm_sqrt_ps);
  // ...or 2x 8-byte doubles.
  ADEPT_DEF_PACKET_TYPE(double, __m128d, 2, 
			_mm_load_pd, _mm_set1_pd, _mm_store_pd,
			_mm_add_pd, _mm_sub_pd,
			_mm_mul_pd, _mm_div_pd, _mm_sqrt_pd);
#endif
#undef ADEPT_DEF_PACKET_TYPE

};

#endif
