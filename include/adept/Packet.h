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

// Headers needed for allocation of aligned memory
#include <new>

#ifdef __unix__
#include <unistd.h>  // Defines _POSIX_VERSION
#endif
#include <stdlib.h>
#ifdef _MSC_VER
#include <malloc.h> // Provides _aligned_malloc on Windows
#endif

#include <adept/quick_e.h>
#include <adept/base.h>

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

    // Trait to define packet size
    template <typename T> struct packet_traits
    { static const int size = 1; };
    template <> struct packet_traits<float>
    { static const int size = ADEPT_FLOAT_PACKET_SIZE; };
    template <> struct packet_traits<double>
    { static const int size = ADEPT_DOUBLE_PACKET_SIZE; };
    

    // -------------------------------------------------------------------
    // Define packet type
    // -------------------------------------------------------------------

    // Unfortunately, with C++98, unions cannot contain std::complex
    // because ith as a constructor... therefore Packet inherits from
    // PacketData to contain the data in order that union is only used
    // for Packets of types that are actually vectorized (which are
    // floats and doubles).
    template <typename T, class Enable = void>
    struct PacketData {
      // Static definitions
      static const int size = packet_traits<T>::size;
      typedef typename quick_e::packet<T,size>::type intrinsic_type;
      PacketData(intrinsic_type d) : data(d) { }
      union {
	intrinsic_type data;
	T value_[size];
      };
      T value() const { return value_[0]; }
      T& operator[](int i) { return value_[i]; }
      const T& operator[](int i) const { return value_[i]; }
    };
    template <typename T>
    struct PacketData<T, typename enable_if<packet_traits<T>::size == 1>::type>
    {
      // Static definitions
      static const int size = 1;
      typedef T intrinsic_type;
      PacketData(intrinsic_type d) : data(d) { }
      T data;
      T value() const { return data; }
      T& operator[](int i) { return data; }
      const T& operator[](int i) const { return data; }
    };
    
    template <typename T>
    struct Packet : public PacketData<T> {
      using PacketData<T>::data;
      static const int size = packet_traits<T>::size;
      typedef typename quick_e::packet<T,size>::type intrinsic_type;
      //      static const int intrinsic_size = 1; // What is this for?
      static const std::size_t alignment_bytes = sizeof(intrinsic_type);
       // T=float/double -> all bits = 1
      static const std::size_t align_mask = (size == 1) ? -1 : alignment_bytes-1;
      static const bool        is_vectorized = (size > 1);
      // Constructors
      Packet() : PacketData<T>(quick_e::set0<intrinsic_type>()) { }
      Packet(const Packet& d) : PacketData<T>(d.data) { }
      template <typename TT>
      Packet(TT d, typename enable_if<is_same<TT,intrinsic_type>::value,int>::type = 0)
	: PacketData<T>(d) { }
      explicit Packet(const T* d) : PacketData<T>(quick_e::load<intrinsic_type>(d)) { }
      //      explicit Packet(T d) : PacketData<T>(quick_e::set1<intrinsic_type>(d)) { }
      template <typename TT>
      explicit Packet(TT d, typename enable_if<is_same<TT,T>::value&&is_vectorized,int>::type = 0)
	: PacketData<T>(quick_e::set1<intrinsic_type>(d)) { }
      // Member functions
      void put(T* __restrict d) const { quick_e::store(d, data); }
      void put_unaligned(T* __restrict d) const { quick_e::storeu(d, data); }
      //      void operator=(T d)              { data = quick_e::set1<intrinsic_type>(d); }
      template <typename TT> //, typename enable_if<is_same<T,TT>::value||is_same<T,intrinsic_type>::value,int>::type = 0>
      void operator=(TT d)              { data = quick_e::set1<intrinsic_type>(d); }
      //      void operator=(intrinsic_type d) { data = d;       }
      void operator=(const Packet& d)  { data = d.data;  }
      void operator+=(const Packet& d) { data = quick_e::add(data, d.data); }
      void operator-=(const Packet& d) { data = quick_e::sub(data, d.data); }
      void operator*=(const Packet& d) { data = quick_e::mul(data, d.data); }
      void operator/=(const Packet& d) { data = quick_e::div(data, d.data); }
      Packet operator-() const         { return quick_e::neg(data); }
      Packet operator+() const         { return *this; }
    };

    //#define QE_PACKET_ARG Packet<T>
    #define QE_PACKET_ARG const Packet<T>& __restrict
        
    // Default functions
    template <typename T> Packet<T> operator+(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return quick_e::add(x.data,y.data); }
    template <typename T> Packet<T> operator-(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return quick_e::sub(x.data,y.data); }
    template <typename T> Packet<T> operator*(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return quick_e::mul(x.data,y.data); }
    template <typename T> Packet<T> operator/(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return quick_e::div(x.data,y.data); }
    template <typename T> Packet<T> fmin(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return quick_e::fmin(x.data,y.data); }
    template <typename T> Packet<T> fmax(QE_PACKET_ARG x, QE_PACKET_ARG y)
    { return quick_e::fmax(x.data,y.data); }
    template <typename T> Packet<T> sqrt(QE_PACKET_ARG x) {
      using std::sqrt;
      using quick_e::sqrt;
      return sqrt(x.data);
    }
    template <typename T> Packet<T> fastexp(QE_PACKET_ARG x) {
      return quick_e::exp(x.data);
    }
#ifdef ADEPT_FAST_EXPONENTIAL
    template <typename T> Packet<T> exp(QE_PACKET_ARG x) {
      return quick_e::exp(x.data);
    }
#else
    template <typename T> Packet<T> exp(QE_PACKET_ARG x) {
      return std::exp(x.data);
    }
#endif

    template <typename T> T hsum(QE_PACKET_ARG x)  { return quick_e::hsum(x.data); }
    template <typename T> T hprod(QE_PACKET_ARG x) { return quick_e::hmul(x.data); }
    template <typename T> T hmin(QE_PACKET_ARG x)  { return quick_e::hmin(x.data); }
    template <typename T> T hmax(QE_PACKET_ARG x)  { return quick_e::hmax(x.data); }

    template <typename T>
    std::ostream& operator<<(std::ostream& os, QE_PACKET_ARG x) {
      os << "{";
      for (int i = 0; i < Packet<T>::size; ++i) {
	os << " " << x[i];
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


  // -------------------------------------------------------------------
  // Fast exponential function
  // -------------------------------------------------------------------

#ifdef ADEPT_FAST_SCALAR_EXPONENTIAL
  // Bring scalar exp from quick_e into this namespace
  inline float  exp(float x)  { return quick_e::exp(x); }
  inline double exp(double x) { return quick_e::exp(x); }
#endif
  inline float  fastexp(float x)  { return quick_e::exp(x); }
  inline double fastexp(double x) { return quick_e::exp(x); }

  // This namespace is only for use in array operations
  namespace functions {
#ifdef ADEPT_FAST_EXPONENTIAL
    // Bring scalar exp from quick_e into this namespace
    inline float  exp(float x)  { return quick_e::exp(x); }
    inline double exp(double x) { return quick_e::exp(x); }
#else
    inline float  exp(float x)  { return std::exp(x); }
    inline double exp(double x) { return std::exp(x); }
#endif
  }

} // End namespace adept

#endif
