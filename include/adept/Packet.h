/* Packet.h -- Vectorization support

    Copyright (C) 2016-2018 European Centre for Medium-Range Weather Forecasts

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

// Headers needed for x86 vector intrinsics
#ifdef __SSE2__
#include <xmmintrin.h> // SSE
#include <emmintrin.h> // SSE2
// Numerous platforms don't define _mm_undefined_ps in xmmintrin.h, so
// we assume none do, except GCC >= 4.9.1 and CLANG >= 3.8.0.  Those
// that don't use an equivalent function that sets the elements to
// zero.
#define ADEPT_MM_UNDEFINED_PS _mm_setzero_ps
#ifdef __clang__
  #if __has_builtin(__builtin_ia32_undef128)
    #undef ADEPT_MM_UNDEFINED_PS
    #define ADEPT_MM_UNDEFINED_PS _mm_undefined_ps
  #endif
#elif defined(__GNUC__)
  #define GCC_VERSION (__GNUC__ * 10000 \
		       + __GNUC_MINOR__ * 100	\
		       + __GNUC_PATCHLEVEL__)
  #if GCC_VERSION >= 40901
    #undef ADEPT_MM_UNDEFINED_PS
    #define ADEPT_MM_UNDEFINED_PS _mm_undefined_ps
  #endif
  #undef GCC_VERSION
#endif // __clang__/__GNUC__
#endif // __SSE2__

#ifdef __AVX__
#include <tmmintrin.h> // SSE3
#include <immintrin.h> // AVX
#endif

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
#  ifdef __AVX__
#    define ADEPT_FLOAT_PACKET_SIZE 8
#  elif defined(__SSE2__)
#    define ADEPT_FLOAT_PACKET_SIZE 4
#  else
#    define ADEPT_FLOAT_PACKET_SIZE 1
#  endif
#endif
#ifndef ADEPT_DOUBLE_PACKET_SIZE
#  ifdef __AVX__
#    define ADEPT_DOUBLE_PACKET_SIZE 4
#  elif defined(__SSE2__)
#    define ADEPT_DOUBLE_PACKET_SIZE 2
#  else
#    define ADEPT_DOUBLE_PACKET_SIZE 1
#  endif
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

    // -------------------------------------------------------------------
    // Default packet types containing only one value
    // -------------------------------------------------------------------

    template <typename T>
    struct Packet {
      typedef T intrinsic_type;
      static const int size = 1;
      static const std::size_t alignment_bytes = sizeof(T);
      static const std::size_t align_mask = -1; // all bits = 1
      static const bool is_vectorized = false;
      Packet() : data(0.0) { }
      explicit Packet(const T* d) : data(*d) { }
      explicit Packet(T d) : data(d) { }
      void put(T* __restrict d) const { *d = data; }
      void operator=(T d)  { data=d; }
      void operator+=(T d) { data+=d; }
      void operator-=(T d) { data-=d; }
      void operator*=(T d) { data*=d; }
      void operator/=(T d) { data/=d; }
      T value() const { return data; }
      T data;
    };

    // Default functions
#ifdef ADEPT_CXX11_FEATURES
    template <typename T>
    Packet<T> fmin(const Packet<T>& __restrict x,
		   const Packet<T>& __restrict y)
    { return std::fmin(x.data,y.data); }
    template <typename T>
    Packet<T> fmax(const Packet<T>& __restrict x,
		   const Packet<T>& __restrict y)
    { return std::fmax(x.data,y.data); }
#else
    template <typename T>
    Packet<T> fmin(const Packet<T>& __restrict x,
		   const Packet<T>& __restrict y)
    { return std::min(x.data,y.data); }
    template <typename T>
    Packet<T> fmax(const Packet<T>& __restrict x,
		   const Packet<T>& __restrict y)
    { return std::max(x.data,y.data); }
#endif

    // -------------------------------------------------------------------
    // Define a specialization, and the basic mathematical operators
    // supported in hardware, for Packet in the case that each
    // contains a single SSE2/AVX intrinsic data object.
    // -------------------------------------------------------------------

#define ADEPT_DEF_PACKET_TYPE(TYPE, INTRINSIC_TYPE, SET0,	\
			      LOAD, LOADU, SET1, STORE, STOREU,	\
			      ADD, SUB, MUL, DIV, SQRT,		\
			      MIN, MAX, HSUM, HPROD,		\
			      HMIN, HMAX)			\
    template <> struct Packet<TYPE> {				\
      typedef INTRINSIC_TYPE intrinsic_type;			\
      static const int size					\
        = sizeof(INTRINSIC_TYPE) / sizeof(TYPE);		\
      static const int intrinsic_size				\
        = sizeof(INTRINSIC_TYPE) / sizeof(TYPE);		\
      static const std::size_t					\
	alignment_bytes = sizeof(INTRINSIC_TYPE);		\
      static const std::size_t					\
        align_mask = sizeof(INTRINSIC_TYPE)-1;			\
      static const bool is_vectorized = true;			\
      Packet()              : data(SET0())  { }			\
      Packet(const TYPE* d) : data(LOAD(d)) { }			\
      Packet(const TYPE* d, int) : data(LOADU(d)) { }		\
      Packet(TYPE d)        : data(SET1(d)) { }			\
      Packet(INTRINSIC_TYPE d)    : data(d) { }			\
      void put(TYPE* __restrict d) const { STORE(d, data); }	\
      void put_unaligned(TYPE* __restrict d) const		\
      { STOREU(d, data); }					\
      void operator=(INTRINSIC_TYPE d) { data=d; }		\
      void operator=(const Packet<TYPE>& __restrict d)		\
      { data=d.data; }						\
      void operator+=(const Packet<TYPE>& __restrict d)		\
      { data = ADD(data, d.data); }				\
      void operator-=(const Packet<TYPE>& __restrict d)		\
      { data = SUB(data, d.data); }				\
      void operator*=(const Packet<TYPE>& __restrict d)		\
      { data = MUL(data, d.data); }				\
      void operator/=(const Packet<TYPE>& __restrict d)		\
      { data = DIV(data, d.data); }				\
      TYPE value() const { return value_; }			\
      union {							\
	INTRINSIC_TYPE data;					\
	TYPE value_;						\
      };							\
    };								\
    inline							\
    std::ostream& operator<<(std::ostream& os,			\
			     const Packet<TYPE>& x) {		\
      TYPE d[Packet<TYPE>::size];				\
      STOREU(d, x.data);					\
      os << "(";						\
      for (int i = 0; i < Packet<TYPE>::size; ++i) {		\
	os << " " << d[i];					\
      }								\
      os << ")";						\
      return os;						\
    }								\
    inline							\
    Packet<TYPE> operator+(const Packet<TYPE>& __restrict x,	\
			   const Packet<TYPE>& __restrict y)	\
    { return ADD(x.data,y.data); }				\
    inline							\
    Packet<TYPE> operator-(const Packet<TYPE>& __restrict x,	\
			   const Packet<TYPE>& __restrict y)	\
    { return SUB(x.data,y.data); }				\
    inline							\
    Packet<TYPE> operator*(const Packet<TYPE>& __restrict x,	\
			   const Packet<TYPE>& __restrict y)	\
    { return MUL(x.data,y.data); }				\
    inline							\
    Packet<TYPE> operator/(const Packet<TYPE>& __restrict x,	\
			   const Packet<TYPE>& __restrict y)	\
    { return DIV(x.data,y.data); }				\
    inline							\
    Packet<TYPE> fmin(const Packet<TYPE>& __restrict x,		\
		     const Packet<TYPE>& __restrict y)		\
    { return MIN(x.data,y.data); }				\
    inline							\
    Packet<TYPE> fmax(const Packet<TYPE>& __restrict x,		\
		     const Packet<TYPE>& __restrict y)		\
    { return MAX(x.data,y.data); }				\
    inline							\
    Packet<TYPE> sqrt(const Packet<TYPE>& __restrict x)		\
    { return SQRT(x.data); }			\
    inline							\
    TYPE hsum(const Packet<TYPE>& __restrict x)			\
    { return HSUM(x.data); }					\
    inline							\
    TYPE hprod(const Packet<TYPE>& __restrict x)		\
    { return HPROD(x.data); }					\
    inline							\
    TYPE hmin(const Packet<TYPE>& __restrict x)			\
    { return HMIN(x.data); }					\
    inline							\
    TYPE hmax(const Packet<TYPE>& __restrict x)			\
    { return HMAX(x.data); }					\
    inline							\
    Packet<TYPE> operator-(const Packet<TYPE>& __restrict x)	\
    { return Packet<TYPE>() - x; }				\
    inline							\
    Packet<TYPE> operator+(const Packet<TYPE>& __restrict x)	\
    { return x; }


    // -------------------------------------------------------------------
    // Define horizontal sum, product, min and max functions (found on
    // Stack Overflow and Intel Forum...)
    // -------------------------------------------------------------------
#ifdef __SSE2__
    // Functions for an SSE packed vector of 4 floats
    inline float mm_hsum_ps(__m128 v) {
      __m128 shuf   = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
      __m128 sums   = _mm_add_ps(v, shuf);
      shuf          = _mm_movehl_ps(shuf, sums);
      return    _mm_cvtss_f32(_mm_add_ss(sums, shuf));    
    }
    inline float mm_hprod_ps(__m128 v) {
      __m128 shuf   = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
      __m128 sums   = _mm_mul_ps(v, shuf);
      shuf          = _mm_movehl_ps(shuf, sums);
      return    _mm_cvtss_f32(_mm_mul_ss(sums, shuf));    
    }
    inline float mm_hmin_ps(__m128 v) {
      __m128 shuf   = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
      __m128 sums   = _mm_min_ps(v, shuf);
      shuf          = _mm_movehl_ps(shuf, sums);
      return    _mm_cvtss_f32(_mm_min_ss(sums, shuf));    
    }
    inline float mm_hmax_ps(__m128 v) {
      __m128 shuf   = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
      __m128 sums   = _mm_max_ps(v, shuf);
      shuf          = _mm_movehl_ps(shuf, sums);
      return    _mm_cvtss_f32(_mm_max_ss(sums, shuf));    
    }

    // Functions for an SSE2 packed vector of 2 doubles
    inline double mm_hsum_pd(__m128d vd) {
      __m128 shuftmp= _mm_movehl_ps(ADEPT_MM_UNDEFINED_PS(),
				    _mm_castpd_ps(vd));
      __m128d shuf  = _mm_castps_pd(shuftmp);
      return  _mm_cvtsd_f64(_mm_add_sd(vd, shuf));
    }
    inline double mm_hprod_pd(__m128d vd) {
      __m128 shuftmp= _mm_movehl_ps(ADEPT_MM_UNDEFINED_PS(),
				    _mm_castpd_ps(vd));
      __m128d shuf  = _mm_castps_pd(shuftmp);
      return  _mm_cvtsd_f64(_mm_mul_sd(vd, shuf));
    }
    inline double mm_hmin_pd(__m128d vd) {
      __m128 shuftmp= _mm_movehl_ps(ADEPT_MM_UNDEFINED_PS(),
				    _mm_castpd_ps(vd));
      __m128d shuf  = _mm_castps_pd(shuftmp);
      return  _mm_cvtsd_f64(_mm_min_sd(vd, shuf));
    }
    inline double mm_hmax_pd(__m128d vd) {
      __m128 shuftmp= _mm_movehl_ps(ADEPT_MM_UNDEFINED_PS(),
				    _mm_castpd_ps(vd));
      __m128d shuf  = _mm_castps_pd(shuftmp);
      return  _mm_cvtsd_f64(_mm_max_sd(vd, shuf));
    }
#undef ADEPT_MM_UNDEFINED_PS
#endif // __SSE2__

#ifdef __AVX__
    // Functions for an AVX packed vector of 8 floats
    inline float mm256_hsum_ps(__m256 v) {
      __m128 vlow  = _mm256_castps256_ps128(v);
      __m128 vhigh = _mm256_extractf128_ps(v, 1);
      vlow  = _mm_add_ps(vlow, vhigh);
      __m128 shuf = _mm_movehdup_ps(vlow);
      __m128 sums = _mm_add_ps(vlow, shuf);
      shuf        = _mm_movehl_ps(shuf, sums);
      return        _mm_cvtss_f32(_mm_add_ss(sums, shuf));
    }
    inline float mm256_hprod_ps(__m256 v) {
      __m128 vlow  = _mm256_castps256_ps128(v);
      __m128 vhigh = _mm256_extractf128_ps(v, 1);
      vlow  = _mm_mul_ps(vlow, vhigh);
      __m128 shuf = _mm_movehdup_ps(vlow);
      __m128 sums = _mm_mul_ps(vlow, shuf);
      shuf        = _mm_movehl_ps(shuf, sums);
      return        _mm_cvtss_f32(_mm_mul_ss(sums, shuf));
    }
    inline float mm256_hmin_ps(__m256 v) {
      __m128 vlow  = _mm256_castps256_ps128(v);
      __m128 vhigh = _mm256_extractf128_ps(v, 1);
      vlow  = _mm_min_ps(vlow, vhigh);
      __m128 shuf = _mm_movehdup_ps(vlow);
      __m128 sums = _mm_min_ps(vlow, shuf);
      shuf        = _mm_movehl_ps(shuf, sums);
      return        _mm_cvtss_f32(_mm_min_ss(sums, shuf));
    }
    inline float mm256_hmax_ps(__m256 v) {
      __m128 vlow  = _mm256_castps256_ps128(v);
      __m128 vhigh = _mm256_extractf128_ps(v, 1);
      vlow  = _mm_max_ps(vlow, vhigh);
      __m128 shuf = _mm_movehdup_ps(vlow);
      __m128 sums = _mm_max_ps(vlow, shuf);
      shuf        = _mm_movehl_ps(shuf, sums);
      return        _mm_cvtss_f32(_mm_max_ss(sums, shuf));
    }

    // Functions for an AVX packed vector of 4 doubles
    inline double mm256_hsum_pd(__m256d vd) {
      __m256d h = _mm256_add_pd(vd, _mm256_permute2f128_pd(vd, vd, 0x1));
      return _mm_cvtsd_f64(_mm_hadd_pd(_mm256_castpd256_pd128(h),
				       _mm256_castpd256_pd128(h) ) );
    }
    inline double mm256_hprod_pd(__m256d vd) {
      __m256d h = _mm256_mul_pd(vd, _mm256_permute2f128_pd(vd, vd, 0x1));
      return mm_hprod_pd(_mm256_castpd256_pd128(h));
    }
    inline double mm256_hmin_pd(__m256d vd) {
      __m256d h = _mm256_min_pd(vd, _mm256_permute2f128_pd(vd, vd, 0x1));
      return mm_hmin_pd(_mm256_castpd256_pd128(h));
    }
    inline double mm256_hmax_pd(__m256d vd) {
      __m256d h = _mm256_max_pd(vd, _mm256_permute2f128_pd(vd, vd, 0x1));
      return mm_hmax_pd(_mm256_castpd256_pd128(h));
    }

#endif

    // -------------------------------------------------------------------
    // Define single-precision Packet
    // -------------------------------------------------------------------
#ifdef __AVX__

#if ADEPT_FLOAT_PACKET_SIZE == 4
    // Need to use SSE2
    ADEPT_DEF_PACKET_TYPE(float, __m128, _mm_setzero_ps, 
			  _mm_load_ps, _mm_loadu_ps, _mm_set1_ps,
			  _mm_store_ps, _mm_storeu_ps,
			  _mm_add_ps, _mm_sub_ps,
			  _mm_mul_ps, _mm_div_ps, _mm_sqrt_ps,
			  _mm_min_ps, _mm_max_ps,
			  mm_hsum_ps, mm_hprod_ps,
			  mm_hmin_ps, mm_hmax_ps); 
#elif ADEPT_FLOAT_PACKET_SIZE == 8
    // Use AVX
    ADEPT_DEF_PACKET_TYPE(float, __m256, _mm256_setzero_ps,
			  _mm256_load_ps, _mm256_loadu_ps, _mm256_set1_ps,
			  _mm256_store_ps, _mm256_storeu_ps,
			  _mm256_add_ps, _mm256_sub_ps,
			  _mm256_mul_ps, _mm256_div_ps, _mm256_sqrt_ps,
			  _mm256_min_ps, _mm256_max_ps,
			  mm256_hsum_ps, mm256_hprod_ps,
			  mm256_hmin_ps, mm256_hmax_ps);
#elif ADEPT_FLOAT_PACKET_SIZE != 1
#error With AVX, ADEPT_FLOAT_PACKET_SIZE must be 1, 4 or 8
#endif

#elif __SSE2__

#if ADEPT_FLOAT_PACKET_SIZE == 4
    ADEPT_DEF_PACKET_TYPE(float, __m128, _mm_setzero_ps, 
			  _mm_load_ps, _mm_loadu_ps, _mm_set1_ps,
			  _mm_store_ps, _mm_storeu_ps,
			  _mm_add_ps, _mm_sub_ps,
			  _mm_mul_ps, _mm_div_ps, _mm_sqrt_ps,
			  _mm_min_ps, _mm_max_ps,
			  mm_hsum_ps, mm_hprod_ps,
			  mm_hmin_ps, mm_hmax_ps); 
#elif ADEPT_FLOAT_PACKET_SIZE != 1
#error With SSE2, ADEPT_FLOAT_PACKET_SIZE must be 1 or 4
#endif

#elif ADEPT_FLOAT_PACKET_SIZE > 1

#error ADEPT_FLOAT_PACKET_SIZE > 1 requires SSE2 or AVX

#endif

    // -------------------------------------------------------------------
    // Define double-precision Packet
    // -------------------------------------------------------------------
#ifdef __AVX__

#if ADEPT_DOUBLE_PACKET_SIZE == 2
    // Need to use SSE2
    ADEPT_DEF_PACKET_TYPE(double, __m128d, _mm_setzero_pd, 
			  _mm_load_pd, _mm_loadu_pd, _mm_set1_pd,
			  _mm_store_pd, _mm_storeu_pd,
			  _mm_add_pd, _mm_sub_pd,
			  _mm_mul_pd, _mm_div_pd, _mm_sqrt_pd,
			  _mm_min_pd, _mm_max_pd,
			  mm_hsum_pd, mm_hprod_pd,
			  mm_hmin_pd, mm_hmax_pd);
#elif ADEPT_DOUBLE_PACKET_SIZE == 4
    ADEPT_DEF_PACKET_TYPE(double, __m256d, _mm256_setzero_pd, 
			  _mm256_load_pd, _mm256_loadu_pd, _mm256_set1_pd,
			  _mm256_store_pd, _mm256_storeu_pd,
			  _mm256_add_pd, _mm256_sub_pd,
			  _mm256_mul_pd, _mm256_div_pd, _mm256_sqrt_pd,
			  _mm256_min_pd, _mm256_max_pd,
			  mm256_hsum_pd, mm256_hprod_pd,
			  mm256_hmin_pd, mm256_hmax_pd);
#elif ADEPT_DOUBLE_PACKET_SIZE != 1
#error With AVX, ADEPT_DOUBLE_PACKET_SIZE must be 1, 2 or 4
#endif

#elif defined(__SSE2__)

#if ADEPT_DOUBLE_PACKET_SIZE == 2
    ADEPT_DEF_PACKET_TYPE(double, __m128d, _mm_setzero_pd, 
			  _mm_load_pd, _mm_loadu_pd, _mm_set1_pd,
			  _mm_store_pd, _mm_storeu_pd,
			  _mm_add_pd, _mm_sub_pd,
			  _mm_mul_pd, _mm_div_pd, _mm_sqrt_pd,
			  _mm_min_pd, _mm_max_pd,
			  mm_hsum_pd, mm_hprod_pd,
			  mm_hmin_pd, mm_hmax_pd);
#elif ADEPT_DOUBLE_PACKET_SIZE != 1
#error With SSE2, ADEPT_DOUBLE_PACKET_SIZE must be 1 or 2
#endif

#elif ADEPT_DOUBLE_PACKET_SIZE > 1

#error ADEPT_DOUBLE_PACKET_SIZE > 1 requires SSE2 or AVX

#endif
    
#undef ADEPT_DEF_PACKET_TYPE


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
