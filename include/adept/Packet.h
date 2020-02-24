/* Packet.h -- Vectorization support

    Copyright (C) 2016-2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

   A Packet contains a short vector of values, and when it is used in
   a limited set of arithmetic operations, the appropriate vector
   instructions will be used.  For example if your hardware and
   compiler support SSE2 then Packet<float> is a vector of 4x 4-byte
   floats while Packet<double> is a vector of 2x 8-byte floats. This
   header file also provides for allocating aligned data
*/

#ifndef AdeptPacket_H
#define AdeptPacket_H 1

#include <iostream>
#include <cstdlib>
#include <cmath>

#include "quick_e.h"

// Headers needed for allocation of aligned memory
#include <new>

#ifdef __unix__
#include <unistd.h>  // Defines _POSIX_VERSION
#endif
#include <stdlib.h>
#ifdef _MSC_VER
#include <malloc.h> // Provides _aligned_malloc on Windows
#endif

#include "base.h"

// -------------------------------------------------------------------
// Determine how many floating point values will be held in a packet
// -------------------------------------------------------------------

#ifndef ADEPT_FLOAT_PACKET_SIZE
#define ADEPT_FLOAT_PACKET_SIZE QE_LONGEST_FLOAT_PACKET
//static const int ADEPT_FLOAT_PACKET_SIZE = quick_e::longest_packet<float>::size;
#endif
#ifndef ADEPT_DOUBLE_PACKET_SIZE
#define ADEPT_DOUBLE_PACKET_SIZE QE_LONGEST_DOUBLE_PACKET
//static const int ADEPT_DOUBLE_PACKET_SIZE = quick_e::longest_packet<double>::size
#endif

// -------------------------------------------------------------------
// Determine how many floating point values will be held in packet of Real
// -------------------------------------------------------------------

#if ADEPT_REAL_TYPE_SIZE == 4
#define ADEPT_REAL_PACKET_SIZE ADEPT_FLOAT_PACKET_SIZE
#elif ADEPT_REAL_TYPE_SIZE == 8
#define ADEPT_REAL_PACKET_SIZE ADEPT_DOUBLE_PACKET_SIZE
#else
#define ADEPT_REAL_PACKET_SIZE 1
#endif

namespace adept {

  namespace internal {

    using namespace quick_e;
    
    // -------------------------------------------------------------------
    // Define packet type
    // -------------------------------------------------------------------
    
    template <typename T>
    struct Packet {
      // Static definitions
      typedef typename quick_e::longest_packet<T>::type intrinsic_type;
      static const int size = quick_e::longest_packet<T>::size;
      //      static const int intrinsic_size = 1; // What is this for?
      static const std::size_t alignment_bytes = sizeof(intrinsic_type);
       // T=float/double -> all bits = 1
      static const std::size_t align_mask = (size == 1) ? -1 : alignment_bytes-1;
      static const bool        is_vectorized = (size > 1);
      // Data
      union {
	intrinsic_type data;
	T value_[size];
      };
      // Constructors
      Packet() : data(set0<intrinsic_type>()) { }
      Packet(const Packet& d) : data(d.data) { }
      template <typename TT, typename enable_if<is_same<TT,intrinsic_type>::value,int>::type = 0>
      Packet(TT d) : data(d) { }
      explicit Packet(const T* d) : data(load<intrinsic_type>(d)) { }
      //      explicit Packet(T d) : data(set1<intrinsic_type>(d)) { }
      template <typename TT, typename enable_if<is_same<TT,T>::value&&is_vectorized,int>::type = 0>
      explicit Packet(TT d) : data(set1<intrinsic_type>(d)) { }
      // Member functions
      void put(T* __restrict d) const { store(d, data); }
      void put_unaligned(T* __restrict d) const { storeu(d, data); }
      //      void operator=(T d)              { data = set1<intrinsic_type>(d); }
      template <typename TT, typename enable_if<is_same<T,TT>::value||is_same<T,intrinsic_type>::value,int>::type = 0>
      void operator=(TT d)              { data = set1<intrinsic_type>(d); }
      //      void operator=(intrinsic_type d) { data = d;       }
      void operator=(const Packet& d)  { data = d.data;  }
      void operator+=(const Packet& d) { data = add(data, d.data); }
      void operator-=(const Packet& d) { data = sub(data, d.data); }
      void operator*=(const Packet& d) { data = mul(data, d.data); }
      void operator/=(const Packet& d) { data = div(data, d.data); }
      Packet operator-()               { return neg(data); }
      T value() const { return value_[0]; }
      T& operator[](int i) { return value_[i]; }
      const T& operator[](int i) const { return value_[i]; }
    };

    //#define QE_PACKET_ARG Packet<T>
    #define QE_PACKET_ARG const Packet<T>& __restrict
        
    // Default functions
    template <typename T> Packet<T> operator+(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return add(x.data,y.data); }
    template <typename T> Packet<T> operator-(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return sub(x.data,y.data); }
    template <typename T> Packet<T> operator*(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return mul(x.data,y.data); }
    template <typename T> Packet<T> operator/(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return div(x.data,y.data); }
    template <typename T> Packet<T> fmin(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return fmin(x.data,y.data); }
    template <typename T> Packet<T> fmax(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return fmax(x.data,y.data); }
    template <typename T> Packet<T> sqrt(QE_PACKET_ARG x) {
      using std::sqrt;
      return sqrt(x.data);
    }
    template <typename T> Packet<T> fastexp(QE_PACKET_ARG x) {
      return quick_e::fastexp(x.data);
    }
    template <typename T> T hsum(QE_PACKET_ARG x)  { return hsum(x.data); }
    template <typename T> T hprod(QE_PACKET_ARG x) { return hmul(x.data); }
    template <typename T> T hmin(QE_PACKET_ARG x)  { return hmin(x.data); }
    template <typename T> T hmax(QE_PACKET_ARG x)  { return hmax(x.data); }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, QE_PACKET_ARG x) {
      os << "{";
      for (int i = 0; i < Packet<T>::size; ++i) {
	os << " " << x.value[i];
      }
      os << "}";
      return os;
    }

    // -------------------------------------------------------------------
    // Aligned allocation and freeing of memory
    // -------------------------------------------------------------------
    template <typename Type>
    inline
    Type* alloc_aligned(Index n) {
      std::size_t n_align = Packet<Type>::alignment_bytes;
      if (n_align < sizeof(void*)) {
	// Note that the requested byte alignment passed to
	// posix_memalign must be at least sizeof(void*)
	return new Type[n];
      }
      else {
	Type* result;
#ifdef _POSIX_VERSION
#if _POSIX_VERSION >= 200112L
	if (posix_memalign(reinterpret_cast<void**>(&result), 
			   n_align, n*sizeof(Type)) != 0) {
	  throw std::bad_alloc();
	}
#else
	result = new Type[n];
#endif
#elif defined(_MSC_VER)
	result = reinterpret_cast<Type*>(_aligned_malloc(n*sizeof(Type),
							 n_align));
	if (result == 0) {
	  throw std::bad_alloc();
	}
#else
	result = new Type[n];	
#endif
      return result;
      }
    }
    
    template <typename Type>
    inline
    void free_aligned(Type* data) {
      // Note that we need to use the same condition as used in
      // alloc_aligned() in order that new[] is followed by delete[]
      // and posix_memalign is followed by free
      if (Packet<Type>::alignment_bytes < sizeof(void*)) {
	delete[] data;
      }
      else { 
#ifdef _POSIX_VERSION
#if _POSIX_VERSION >= 200112L   
	free(data);
#else
	delete[] data;
#endif
#elif defined(_MSC_VER)
	_aligned_free(data);
#else
	delete[] data;
#endif
      }
    }


    // -------------------------------------------------------------------
    // Check if templated object is a packet: is_packet
    // -------------------------------------------------------------------
    template <typename T>
    struct is_packet {
      static const bool value = false;
    };
    template <typename T>
    struct is_packet<Packet<T> > {
      static const bool value = true;
    };


  } // End namespace internal

} // End namespace adept

#endif
