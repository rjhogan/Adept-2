/* quick_e.h -- Fast exponential function for Intel and ARM intrinsics

   Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

   Author: Robin Hogan <r.j.hogan@ecmwf.int>

   This file is part of the Adept library, although can be used
   stand-alone.

   The exponential function for real arguments is used in many areas
   of physics, yet is not vectorized by many compilers.  This C++
   header file provides a fast exponential function (quick_e::exp) for
   single and double precision floating point numbers, Intel
   intrinsics representing packets of 2, 4, 8 and 16 such numbers, and
   ARM NEON intrinsics representing 2 doubles or 4 floats.  The
   algorithm has been taken from Agner Fog's Vector Class Library. It
   is designed to be used in other libraries that make use of Intel or
   ARM intrinsics.  Since such libraries often define their own
   classes for representing vectors of numbers, this file does not
   define any such classes itself.

   Also in the namespace quick_e, this file defines the following
   inline functions that work on intrinsics of type "Vec" and the
   corresponding scalar type "Sca":

     Vec add(Vec x, Vec y)   Add the elements of x and y
     Vec sub(Vec x, Vec y)   Subtract the elements of x and y
     Vec mul(Vec x, Vec y)   Multiply the elements of x and y
     Vec div(Vec x, Vec y)   Divide the elements of x and y
     Vec set0<Vec>()         Returns zero in all elements
     Vec set1<Vec>(Sca a)    Returns all elements set to a
     Vec sqrt(Vec x)         Square root of all elements
     Vec fmin(Vec x, Vec y)  Minimum of elements of x and y
     Vec fmax(Vec x, Vec y)  Maximum of elements of x and y
     Vec load(const Sca* d)  Aligned load from memory location d
     Vec loadu(const Sca* d) Unaligned load from memory location d
     void store(Sca* d, Vec x)  Aligned store of x to d
     void storeu(Sca* d, Vec x) Unaligned store of x to d
     Sca hsum(Vec x)         Horizontal sum of elements of x
     Sca hmul(Vec x)         Horizontal product of elements of x
     Sca hmin(Vec x)         Horizontal minimum of elements of x
     Sca hmax(Vec x)         Horizontal maximum of elements of x
     Vec fma(Vec x, Vec y, Vec z)  Fused multiply-add: (x*y)+z
     Vec fnma(Vec x, Vec y, Vec z) Returns z-(x*y)
     Vec pow2n(Vec x)        Returns 2 to the power of x
     Vec exp(Vec x)          Returns exponential of x
   
 */

#ifndef QuickE_H
#define QuickE_H 1

#include <cmath>

// Microsoft compiler doesn't define __SSE2__ even if __AVX__ is
// defined
#ifdef __AVX__
#ifndef __SSE2__
#define __SSE2__ 1
#endif
#endif

// Headers needed for x86 vector intrinsics
#ifdef __SSE2__
  #include <xmmintrin.h> // SSE
  #include <emmintrin.h> // SSE2
  // Numerous platforms don't define _mm_undefined_ps in xmmintrin.h,
  // so we assume none do, except GCC >= 4.9.1 and CLANG >= 3.8.0.
  // Those that don't use an equivalent function that sets the
  // elements to zero.
  #define QE_MM_UNDEFINED_PS _mm_setzero_ps
  #ifdef __clang__
    #if __has_builtin(__builtin_ia32_undef128)
      #undef QE_MM_UNDEFINED_PS
      #define QE_MM_UNDEFINED_PS _mm_undefined_ps
    #endif
  #elif defined(__GNUC__)
    #define GCC_VERSION (__GNUC__ * 10000 \
			 + __GNUC_MINOR__ * 100	\
			 + __GNUC_PATCHLEVEL__)
    #if GCC_VERSION >= 40901
      #undef QE_MM_UNDEFINED_PS
      #define QE_MM_UNDEFINED_PS _mm_undefined_ps
    #endif
    #undef GCC_VERSION
  #endif // __clang__/__GNUC__
#endif // __SSE2__

#ifdef __SSE4_1__
#include <smmintrin.h>
#endif

#ifdef __AVX__
  #include <tmmintrin.h> // SSE3
  #include <immintrin.h> // AVX
#endif

#ifdef __AVX512F__
  #include <immintrin.h>
#endif

#ifdef __ARM_NEON
  // We only have sufficient floating-point intrinsics to vectorize on
  // 64-bit ARM targets
  #if defined(__aarch64__) || defined(_M_ARM64)
    #define QE_HAVE_ARM64_NEON 1
    #include "arm_neon.h"
  #endif
#endif

namespace quick_e {

  // -------------------------------------------------------------------
  // Traits
  // -------------------------------------------------------------------

  template <typename Type, int Size> struct packet {
    static const bool is_available = false;
    static const int  size         = 1;
    typedef Type type;
  };
  template <typename Type> struct longest_packet {
    typedef Type type;
    static const int size = 1;
  };

  // g++ issues ugly warnings if VEC is an Intel intrinsic, disabled
  // with -Wno-ignored-attributes
#define QE_DEFINE_TRAITS(TYPE, SIZE, VEC, HALF_TYPE)   \
  template <> struct packet<TYPE,SIZE> {	       \
    static const bool is_available = true;	       \
    static const int  size = SIZE;		       \
    typedef VEC type;				       \
    typedef HALF_TYPE half_type;		       \
  };

#define QE_DEFINE_LONGEST(VECS, VECD)			\
  template <> struct longest_packet<float> {		\
    typedef VECS type;					\
    static const int size = sizeof(VECS)/sizeof(float);	\
  };							\
  template <> struct longest_packet<double> {		\
    typedef VECD type;					\
    static const int size = sizeof(VECD)/sizeof(double);\
  };
  
#ifdef __SSE2__
  #define QE_HAVE_FAST_EXP 1
  QE_DEFINE_TRAITS(float, 4, __m128, __m128)
  QE_DEFINE_TRAITS(double, 2, __m128d, double)
  #ifdef __AVX__
    QE_DEFINE_TRAITS(float, 8, __m256, __m128)
    QE_DEFINE_TRAITS(double, 4, __m256d, __m128d)
    #ifdef __AVX512F__
      QE_DEFINE_TRAITS(float, 16, __m512, __m256)
      QE_DEFINE_TRAITS(double, 8, __m512d, __m256d)
      QE_DEFINE_LONGEST(__m512, __m512d)
      #define QE_LONGEST_FLOAT_PACKET 16
      #define QE_LONGEST_DOUBLE_PACKET 8
    #else
      QE_DEFINE_LONGEST(__m256, __m256d)
      #define QE_LONGEST_FLOAT_PACKET 8
      #define QE_LONGEST_DOUBLE_PACKET 4
    #endif
  #else
    QE_DEFINE_LONGEST(__m128, __m128d)
    #define QE_LONGEST_FLOAT_PACKET 4
    #define QE_LONGEST_DOUBLE_PACKET 2
  #endif
  // If QE_AVAILABLE is defined then we can use the fast exponential
  #define QE_AVAILABLE
#elif defined(QE_HAVE_ARM64_NEON)
  #define QE_HAVE_FAST_EXP 1
  QE_DEFINE_TRAITS(float, 4, float32x4_t, float32x4_t)
  QE_DEFINE_TRAITS(double, 2, float64x2_t, double)
  QE_DEFINE_LONGEST(float32x4_t, float64x2_t)
  #define QE_LONGEST_FLOAT_PACKET 4
  #define QE_LONGEST_DOUBLE_PACKET 2
#else
  // No vectorization available: longest packet is of size 1
  QE_DEFINE_LONGEST(float, double);
#define QE_LONGEST_FLOAT_PACKET 1
#define QE_LONGEST_DOUBLE_PACKET 1
#endif
  
  
  // -------------------------------------------------------------------
  // Scalars
  // -------------------------------------------------------------------
  
  // Define a few functions for scalars in order that the same
  // implementation of "exp" can be used for both scalars and SIMD
  // vectors
  template <typename T> T add(T x, T y) { return x+y; }
  template <typename T> T sub(T x, T y) { return x-y; }
  template <typename T> T mul(T x, T y) { return x*y; }
  template <typename T> T div(T x, T y) { return x/y; }
  template <typename T> T neg(T x)      { return -x;  }
  template <typename T, typename V> void store(T* d, V x) { *d = x;  }
  template <typename T, typename V> void storeu(T* d, V x){ *d = x;  }
  template <typename V, typename T> V load(const T* d) { return *d;  }
  template <typename V, typename T> V loadu(const T* d){ return *d;  }
  template <typename V, typename T> V set1(T x) { return x;   }
  template <typename V> inline V set0() { return 0.0; };
  template <typename T> T sqrt(T x) { return std::sqrt(x); }
  
  template <typename T> T hsum(T x) { return x; }
  template <typename T> T hmul(T x) { return x; }
  template <typename T> T hmin(T x) { return x; }
  template <typename T> T hmax(T x) { return x; }
  
  template <typename T> T fma(T x, T y, T z)  { return (x*y)+z; }
  template <typename T> T fnma(T x, T y, T z) { return z-(x*y); }
  template <typename T> T fmin(T x, T y)  { return std::min(x,y); }
  template <typename T> T fmax(T x, T y)  { return std::max(x,y); }
 
#if __cplusplus > 199711L
  template <> inline float  fmin(float x, float y)   { return std::fmin(x,y); }
  template <> inline double fmin(double x, double y) { return std::fmin(x,y); }
  template <> inline float  fmax(float x, float y)   { return std::fmax(x,y); }
  template <> inline double fmax(double x, double y) { return std::fmax(x,y); }
#endif

  inline float select_gt(float x1, float x2, float y1, float y2) {
    if (x1 > x2) { return y1; } else { return y2; }
  }
  inline double select_gt(double x1, double x2, double y1, double y2) {
    if (x1 > x2) { return y1; } else { return y2; }
  }
  
  inline bool all_in_range(float x, float low_bound, float high_bound) {
    return x >= low_bound && x <= high_bound;
  }
  inline bool all_in_range(double x, double low_bound, double high_bound) {
    return x >= low_bound && x <= high_bound;
  }
  
  // -------------------------------------------------------------------
  // Macros to define mathematical operations
  // -------------------------------------------------------------------

  // Basic load store, arithmetic, sqrt, min and max
#define QE_DEFINE_BASIC(TYPE, VEC, LOAD, LOADU, SET0, SET1,	\
			STORE, STOREU, ADD, SUB, MUL, DIV,	\
			SQRT, FMIN, FMAX)			\
  inline VEC add(VEC x, VEC y)       { return ADD(x, y); }	\
  inline VEC sub(VEC x, VEC y)       { return SUB(x, y); }	\
  inline VEC mul(VEC x, VEC y)       { return MUL(x, y); }	\
  inline VEC div(VEC x, VEC y)       { return DIV(x, y); }	\
  inline VEC neg(VEC x)              { return SUB(SET0(), x); }	\
  template <> inline VEC set0<VEC>()        { return SET0();  }	\
  template <> inline VEC set1<VEC>(TYPE x)  { return SET1(x); }	\
  inline VEC sqrt(VEC x)             { return SQRT(x);   }	\
  inline VEC fmin(VEC x, VEC y)      { return FMIN(x,y); }	\
  inline VEC fmax(VEC x, VEC y)      { return FMAX(x,y); }	\
  template <> inline VEC load<VEC,TYPE>(const TYPE* d)		\
  { return LOAD(d);  }						\
  template <> inline VEC loadu<VEC,TYPE>(const TYPE* d)         \
  { return LOADU(d); }						\
  inline void store(TYPE* d, VEC x)  { STORE(d, x);      }	\
  inline void storeu(TYPE* d, VEC x) { STOREU(d, x);     }	\
  inline std::ostream& operator<<(std::ostream& os, VEC x) {	\
    static const int size = sizeof(VEC)/sizeof(TYPE);		\
    union { VEC v; TYPE d[size]; };				\
    v = x; os << "{";						\
    for (int i = 0; i < size; ++i)				\
      { os << " " << d[i]; }					\
    os << "}"; return os;					\
  }
  
#define QE_DEFINE_CHOP(VEC, HALF_TYPE, LOW, HIGH, PACK)		\
  inline HALF_TYPE low(VEC x)   { return LOW;       }		\
  inline HALF_TYPE high(VEC x)  { return HIGH;      }		\
  inline VEC pack(HALF_TYPE x, HALF_TYPE y) { return PACK; }
  
  // Reduction operations: horizontal sum, product, min and max
#define QE_DEFINE_HORIZ(TYPE, VEC, HSUM, HMUL, HMIN, HMAX)	\
  inline TYPE hsum(VEC x)            { return HSUM(x);   }	\
  inline TYPE hmul(VEC x)            { return HMUL(x);   }	\
  inline TYPE hmin(VEC x)            { return HMIN(x);   }	\
  inline TYPE hmax(VEC x)            { return HMAX(x);   }

  // Define fused multiply-add functions
#define QE_DEFINE_FMA(TYPE, VEC, FMA, FNMA)			\
  inline VEC fma(VEC x,VEC y,VEC z)  { return FMA(x,y,z); }	\
  inline VEC fma(VEC x,TYPE y,VEC z)				\
  { return FMA(x,set1<VEC>(y),z); }				\
  inline VEC fma(TYPE x, VEC y, TYPE z)				\
  { return FMA(set1<VEC>(x),y,set1<VEC>(z)); }			\
  inline VEC fma(VEC x, VEC y, TYPE z)				\
  { return FMA(x,y,set1<VEC>(z)); }				\
  inline VEC fnma(VEC x,VEC y,VEC z) { return FNMA(x,y,z);}

  // Alternative order of arguments for ARM NEON
#define QE_DEFINE_FMA_ALT(TYPE, VEC, FMA, FNMA)			\
  inline VEC fma(VEC x,VEC y,VEC z)  { return FMA(z,x,y); }	\
  inline VEC fma(VEC x,TYPE y,VEC z)				\
  { return FMA(z,x,set1<VEC>(y)); }				\
  inline VEC fma(TYPE x, VEC y, TYPE z)				\
  { return FMA(set1<VEC>(z),set1<VEC>(x),y); }			\
  inline VEC fma(VEC x, VEC y, TYPE z)				\
  { return FMA(set1<VEC>(z),x,y); }				\
  inline VEC fnma(VEC x,VEC y,VEC z) { return FNMA(z,x,y);}
  
  // Emulate fused multiply-add if instruction not available
#define QE_EMULATE_FMA(TYPE, VEC)				\
  inline VEC fma(VEC x,VEC y,VEC z)  { return add(mul(x,y),z);}	\
  inline VEC fma(VEC x,TYPE y,VEC z)				\
  { return add(mul(x,set1<VEC>(y)),z); }			\
  inline VEC fma(TYPE x, VEC y, TYPE z)				\
  { return add(mul(set1<VEC>(x),y),set1<VEC>(z)); }		\
  inline VEC fma(VEC x, VEC y, TYPE z)				\
  { return add(mul(x,y),set1<VEC>(z)); }			\
  inline VEC fnma(VEC x,VEC y,VEC z) { return sub(z,mul(x,y));}

#define QE_DEFINE_POW2N_S(VEC, VECI, CASTTO, CASTBACK, SHIFTL,  \
			  SETELEM)				\
  inline VEC pow2n(VEC n) {					\
    const float pow2_23 = 8388608.0;				\
    const float bias = 127.0;					\
    VEC  a = add(n, set1<VEC>(bias+pow2_23));			\
    VECI b = CASTTO(a);						\
    VECI c = SHIFTL(b, SETELEM(23));				\
    VEC  d = CASTBACK(c);					\
    return d;							\
  }
#define QE_DEFINE_POW2N_D(VEC, VECI, CASTTO, CASTBACK, SHIFTL,  \
			  SETELEM)				\
  inline VEC pow2n(VEC n) {					\
    const double pow2_52 = 4503599627370496.0;			\
    const double bias = 1023.0;					\
    VEC  a = add(n, set1<VEC>(bias+pow2_52));			\
    VECI b = CASTTO(a);						\
    VECI c = SHIFTL(b, SETELEM(52));				\
    VEC  d = CASTBACK(c);					\
    return d;							\
  }

  // -------------------------------------------------------------------
  // Define operations for SSE2: vector of 4 floats or 2 doubles
  // -------------------------------------------------------------------
  

#ifdef __SSE2__
  QE_DEFINE_BASIC(float, __m128, _mm_load_ps, _mm_loadu_ps,
		  _mm_setzero_ps, _mm_set1_ps, _mm_store_ps, _mm_storeu_ps,
		  _mm_add_ps, _mm_sub_ps, _mm_mul_ps, _mm_div_ps,
		  _mm_sqrt_ps, _mm_min_ps, _mm_max_ps)
  QE_DEFINE_BASIC(double, __m128d, _mm_load_pd, _mm_loadu_pd,
		  _mm_setzero_pd, _mm_set1_pd, _mm_store_pd, _mm_storeu_pd,
		  _mm_add_pd, _mm_sub_pd, _mm_mul_pd, _mm_div_pd,
		  _mm_sqrt_pd, _mm_min_pd, _mm_max_pd)
  // Don't define chop operations for __m128 because we don't have a
  // container for two floats
  QE_DEFINE_CHOP(__m128d, double, _mm_cvtsd_f64(x),
		 _mm_cvtsd_f64(_mm_unpackhi_pd(x,x)),
		 _mm_set_pd(y,x))
		 
  // No built-in horizontal operations for SSE2, so need to implement
  // by hand
#define QE_DEFINE_HORIZ_SSE2(FUNC, OP_PS, OP_SS, OP_PD)			\
  inline float FUNC(__m128 x) {						\
    __m128 shuf = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));	\
    __m128 sums = OP_PS(x, shuf);					\
    shuf        = _mm_movehl_ps(shuf, sums);				\
    return _mm_cvtss_f32(OP_SS(sums, shuf));				\
  }									\
  inline double FUNC(__m128d x) {					\
    __m128 shuftmp= _mm_movehl_ps(QE_MM_UNDEFINED_PS(),			\
				  _mm_castpd_ps(x));			\
    __m128d shuf  = _mm_castps_pd(shuftmp);				\
    return  _mm_cvtsd_f64(OP_PD(x, shuf));				\
  }
  QE_DEFINE_HORIZ_SSE2(hsum, _mm_add_ps, _mm_add_ss, _mm_add_pd)
  QE_DEFINE_HORIZ_SSE2(hmul, _mm_mul_ps, _mm_mul_ss, _mm_mul_pd)
  QE_DEFINE_HORIZ_SSE2(hmin, _mm_min_ps, _mm_min_ss, _mm_min_pd)
  QE_DEFINE_HORIZ_SSE2(hmax, _mm_max_ps, _mm_max_ss, _mm_max_pd)

#undef QE_MM_UNDEFINED_PS
#undef QE_DEFINE_HORIZ_SSE2
  
#ifdef __FMA__
  QE_DEFINE_FMA(float, __m128, _mm_fmadd_ps, _mm_fnmadd_ps)
  QE_DEFINE_FMA(double, __m128d, _mm_fmadd_pd, _mm_fnmadd_pd)
#else
  QE_EMULATE_FMA(float, __m128)
  QE_EMULATE_FMA(double, __m128d)
#endif
#ifdef __SSE4_1__
  inline __m128 unchecked_round(__m128 x)
  { return _mm_round_ps(x, (_MM_FROUND_TO_NEAREST_INT
			      |_MM_FROUND_NO_EXC)); }
  inline __m128d unchecked_round(__m128d x)
  { return _mm_round_pd(x, (_MM_FROUND_TO_NEAREST_INT
			      |_MM_FROUND_NO_EXC)); }
#else
  // No native function available, but since the arguments are limited
  // to +/- 700, we don't need to check for going out of bounds
  inline __m128 unchecked_round(__m128 x)
  { return _mm_cvtepi32_ps(_mm_cvtps_epi32(x)); }
  inline __m128d unchecked_round(__m128d x)
  { return _mm_cvtepi32_pd(_mm_cvtpd_epi32(x)); }

#endif
  inline float unchecked_round(float x)
  { return _mm_cvtss_f32(unchecked_round(_mm_set_ss(x))); }
  inline double unchecked_round(double x)
  { return low(unchecked_round(_mm_set_sd(x))); }

  QE_DEFINE_POW2N_S(__m128, __m128i, _mm_castps_si128,
		    _mm_castsi128_ps, _mm_sll_epi32, _mm_cvtsi32_si128)
  QE_DEFINE_POW2N_D(__m128d, __m128i, _mm_castpd_si128,
		    _mm_castsi128_pd, _mm_sll_epi64, _mm_cvtsi32_si128)
  inline float pow2n(float x)
  { return _mm_cvtss_f32(pow2n(quick_e::set1<__m128>(x))); }
  inline double pow2n(double x)
  { return low(pow2n(quick_e::set1<__m128d>(x))); }

  
  inline bool horiz_and(__m128i a) {
#ifdef __SSE4_1__
    return _mm_testc_si128(a, _mm_set1_epi32(-1)) != 0;
#else
    __m128i t1 = _mm_unpackhi_epi64(a, a); // get 64 bits down
    __m128i t2 = _mm_and_si128(a, t1);     // and 64 bits
#ifdef __x86_64__
    int64_t t5 = _mm_cvtsi128_si64(t2);    // transfer 64 bits to integer
    return  t5 == int64_t(-1);
#else
    __m128i t3 = _mm_srli_epi64(t2, 32);   // get 32 bits down
    __m128i t4 = _mm_and_si128(t2, t3);    // and 32 bits
    int     t5 = _mm_cvtsi128_si32(t4);    // transfer 32 bits to integer
    return  t5 == -1;
#endif  // __x86_64__
#endif  // SSE 4.1
  }
  inline bool all_in_range(__m128 x, float low_bound, float high_bound) {
    return horiz_and(_mm_castps_si128(_mm_and_ps(
			 _mm_cmpge_ps(x,set1<__m128>(low_bound)),
			 _mm_cmple_ps(x,set1<__m128>(high_bound)))));
  }
  inline bool all_in_range(__m128d x, double low_bound, double high_bound) {
    return horiz_and(_mm_castpd_si128(_mm_and_pd(
			 _mm_cmpge_pd(x,set1<__m128d>(low_bound)),
			 _mm_cmple_pd(x,set1<__m128d>(high_bound)))));
  }

  // If x1 > x2, select y1, or select y2 otherwise
  inline __m128 select_gt(__m128 x1, __m128 x2,
			  __m128 y1, __m128 y2) {
    __m128 mask = _mm_cmpgt_ps(x1,x2);
#ifdef __SSE4_1__
    return _mm_blendv_ps(y2, y1, mask);
#else
    return _mm_or_ps(_mm_and_ps(mask, y1),
		     _mm_andnot_ps(mask, y2));
#endif
  }
  inline __m128d select_gt(__m128d x1, __m128d x2,
			   __m128d y1, __m128d y2) {
    __m128d mask = _mm_cmpgt_pd(x1,x2);
#ifdef __SSE4_1__
    return _mm_blendv_pd(y2, y1, mask);
#else
    return _mm_or_pd(_mm_and_pd(mask, y1),
		     _mm_andnot_pd(mask, y2));
#endif
  }
#endif

  // -------------------------------------------------------------------
  // Define operations for AVX: vector of 8 floats or 4 doubles
  // -------------------------------------------------------------------
#ifdef __AVX__
  QE_DEFINE_BASIC(float, __m256, _mm256_load_ps, _mm256_loadu_ps,
		  _mm256_setzero_ps, _mm256_set1_ps,
		  _mm256_store_ps, _mm256_storeu_ps,
		  _mm256_add_ps, _mm256_sub_ps,
		  _mm256_mul_ps, _mm256_div_ps, _mm256_sqrt_ps,
		  _mm256_min_ps, _mm256_max_ps)
  QE_DEFINE_BASIC(double, __m256d, _mm256_load_pd, _mm256_loadu_pd,
		  _mm256_setzero_pd, _mm256_set1_pd,
		  _mm256_store_pd, _mm256_storeu_pd,
		  _mm256_add_pd, _mm256_sub_pd,
		  _mm256_mul_pd, _mm256_div_pd, _mm256_sqrt_pd,
		  _mm256_min_pd, _mm256_max_pd)
  QE_DEFINE_CHOP(__m256, __m128,
		 _mm256_castps256_ps128(x), _mm256_extractf128_ps(x,1),
		 _mm256_permute2f128_ps(_mm256_castps128_ps256(x),
					_mm256_castps128_ps256(y), 0x20))
  QE_DEFINE_CHOP(__m256d, __m128d, _mm256_castpd256_pd128(x),
		 _mm256_extractf128_pd(x,1),
		 _mm256_permute2f128_pd(_mm256_castpd128_pd256(x),
					_mm256_castpd128_pd256(y), 0x20));

  // Implement by calling SSE2 h* functions
  inline float  hsum(__m256 x)  { return hsum(add(low(x), high(x))); }
  inline float  hmul(__m256 x)  { return hmul(mul(low(x), high(x))); }
  inline float  hmin(__m256 x)  { return hmin(fmin(low(x), high(x))); }
  inline float  hmax(__m256 x)  { return hmax(fmax(low(x), high(x))); }
  inline double hsum(__m256d x) { return hsum(add(low(x),  high(x))); } // Alternative would be to use _mm_hadd_pd
  inline double hmul(__m256d x) { return hmul(mul(low(x),  high(x))); }
  inline double hmin(__m256d x) { return hmin(fmin(low(x), high(x))); }
  inline double hmax(__m256d x) { return hmax(fmax(low(x), high(x))); }
  
  // Define extras
#ifdef __FMA__
  QE_DEFINE_FMA(float, __m256,  _mm256_fmadd_ps, _mm256_fnmadd_ps)
  QE_DEFINE_FMA(double, __m256d, _mm256_fmadd_pd, _mm256_fnmadd_pd)
#else
  QE_EMULATE_FMA(float, __m256)
  QE_EMULATE_FMA(double, __m256d)
#endif
  
  inline __m256 unchecked_round(__m256 x)
  { return _mm256_round_ps(x, (_MM_FROUND_TO_NEAREST_INT
			       |_MM_FROUND_NO_EXC)); }
  inline __m256d unchecked_round(__m256d x)
  { return _mm256_round_pd(x, (_MM_FROUND_TO_NEAREST_INT
			       |_MM_FROUND_NO_EXC)); }
  #ifdef __AVX2__
    QE_DEFINE_POW2N_S(__m256, __m256i, _mm256_castps_si256,
		      _mm256_castsi256_ps, _mm256_sll_epi32, _mm_cvtsi32_si128)
    QE_DEFINE_POW2N_D(__m256d, __m256i, _mm256_castpd_si256,
		      _mm256_castsi256_pd, _mm256_sll_epi64, _mm_cvtsi32_si128)
  #else
    // Suboptimized versions call the SSE2 functions on the upper and
    // lower parts
    inline __m256 pow2n(__m256 n) {
      return pack(pow2n(low(n)), pow2n(high(n)));
    }
    inline __m256d pow2n(__m256d n) {
      return pack(pow2n(low(n)), pow2n(high(n)));
    }
  #endif
 
  // Return true if all elements of x are in the range (inclusive) of
  // low_bound to high_bound.  If so the exp call can exit before the
  // more costly case of working out what to do with inputs out of
  // bounds.  Note that _CMP_GE_OS means compare
  // greater-than-or-equal-to, ordered, signaling, where "ordered"
  // means that if either operand is NaN, the result is false.
  inline bool all_in_range(__m256 x, float low_bound, float high_bound) {
    return _mm256_testc_si256(_mm256_castps_si256(_mm256_and_ps(
		 _mm256_cmp_ps(x,set1<__m256>(low_bound), _CMP_GE_OS),
		 _mm256_cmp_ps(x,set1<__m256>(high_bound), _CMP_LE_OS))),
			      _mm256_set1_epi32(-1)) != 0;
  }
  inline bool all_in_range(__m256d x, double low_bound, double high_bound) {
    return _mm256_testc_si256(_mm256_castpd_si256(_mm256_and_pd(
		 _mm256_cmp_pd(x,set1<__m256d>(low_bound), _CMP_GE_OS),
		 _mm256_cmp_pd(x,set1<__m256d>(high_bound), _CMP_LE_OS))),
			      _mm256_set1_epi32(-1)) != 0;
  }
  inline __m256 select_gt(__m256 x1, __m256 x2,
			  __m256 y1, __m256 y2) {
    return _mm256_blendv_ps(y2, y1, _mm256_cmp_ps(x1,x2,_CMP_GT_OS));
  }
  inline __m256d select_gt(__m256d x1, __m256d x2,
			   __m256d y1, __m256d y2) {
    return _mm256_blendv_pd(y2, y1, _mm256_cmp_pd(x1,x2,_CMP_GT_OS));
  }

#endif
  

  // -------------------------------------------------------------------
  // Define operations for AVX512: vector of 16 floats or 8 doubles
  // -------------------------------------------------------------------
#ifdef __AVX512F__
  QE_DEFINE_BASIC(float, __m512, _mm512_load_ps, _mm512_loadu_ps,
		  _mm512_setzero_ps, _mm512_set1_ps,
		  _mm512_store_ps, _mm512_storeu_ps,
		  _mm512_add_ps, _mm512_sub_ps,
		  _mm512_mul_ps, _mm512_div_ps, _mm512_sqrt_ps,
		  _mm512_min_ps, _mm512_max_ps)
  QE_DEFINE_HORIZ(float, __m512,
		  _mm512_reduce_add_ps, _mm512_reduce_mul_ps,
		  _mm512_reduce_min_ps, _mm512_reduce_max_ps)
  QE_DEFINE_BASIC(double, __m512d, _mm512_load_pd, _mm512_loadu_pd,
		  _mm512_setzero_pd, _mm512_set1_pd,
		  _mm512_store_pd, _mm512_storeu_pd,
		  _mm512_add_pd, _mm512_sub_pd,
		  _mm512_mul_pd, _mm512_div_pd, _mm512_sqrt_pd,
		  _mm512_min_pd, _mm512_max_pd)
  QE_DEFINE_HORIZ(double, __m512d,
		  _mm512_reduce_add_pd, _mm512_reduce_mul_pd,
		  _mm512_reduce_min_pd, _mm512_reduce_max_pd)
  
  inline __m512 unchecked_round(__m512 x)   { return _mm512_roundscale_ps(x, 0); }
  inline __m512d unchecked_round(__m512d x) { return _mm512_roundscale_pd(x, 0); }

  QE_DEFINE_FMA(float, __m512,  _mm512_fmadd_ps, _mm512_fnmadd_ps)
  QE_DEFINE_FMA(double, __m512d, _mm512_fmadd_pd, _mm512_fnmadd_pd)
  
  QE_DEFINE_POW2N_S(__m512, __m512i, _mm512_castps_si512,
		    _mm512_castsi512_ps, _mm512_sll_epi32, _mm_cvtsi32_si128)
  QE_DEFINE_POW2N_D(__m512d, __m512i, _mm512_castpd_si512,
		    _mm512_castsi512_pd, _mm512_sll_epi64, _mm_cvtsi32_si128)

  inline bool all_in_range(__m512 x, float low_bound, float high_bound) {
    return static_cast<unsigned short int>(_mm512_kand(
	      _mm512_cmp_ps_mask(x,set1<__m512>(low_bound),_CMP_GE_OS),
	      _mm512_cmp_ps_mask(x,set1<__m512>(high_bound),_CMP_LE_OS)))
      == static_cast<unsigned short int>(65535);
  }
  inline bool all_in_range(__m512d x, double low_bound, double high_bound) {
    return static_cast<unsigned short int>(_mm512_kand(
	      _mm512_cmp_pd_mask(x,set1<__m512d>(low_bound),_CMP_GE_OS),
	      _mm512_cmp_pd_mask(x,set1<__m512d>(high_bound),_CMP_LE_OS)))
      == static_cast<unsigned short int>(255);
  }
  inline __m512 select_gt(__m512 x1, __m512 x2,
			  __m512 y1, __m512 y2) {
    return _mm512_mask_mov_ps(y2, _mm512_cmp_ps_mask(x1,x2,_CMP_GT_OS), y1);
  }
  inline __m512d select_gt(__m512d x1, __m512d x2,
			   __m512d y1, __m512d y2) {
    return _mm512_mask_mov_pd(y2, _mm512_cmp_pd_mask(x1,x2,_CMP_GT_OS), y1);
  }

#endif

  
#ifdef QE_HAVE_ARM64_NEON

  // Implement ARM version of x86 setzero
  inline float32x4_t vzeroq_f32() { return vdupq_n_f32(0.0); }
  inline float64x2_t vzeroq_f64() { return vdupq_n_f64(0.0); }
  // Horizontal multiply across vector
  inline float vmulvq_f32(float32x4_t x) {
    union {
      float32x2_t v;
      float data[2];
    };
    v = vmul_f32(vget_low_f32(x), vget_high_f32(x));
    return data[0] * data[1];
  }
  inline double vmulvq_f64(float64x2_t x) {
    union {
      float64x2_t v;
      double data[2];
    };
    v = x;
    return data[0] * data[1];
  }
  
  QE_DEFINE_BASIC(float, float32x4_t, vld1q_f32, vld1q_f32,
		  vzeroq_f32, vdupq_n_f32, vst1q_f32, vst1q_f32,
		  vaddq_f32, vsubq_f32, vmulq_f32, vdivq_f32,
		  vsqrtq_f32, vminq_f32, vmaxq_f32)
  QE_DEFINE_HORIZ(float, float32x4_t,
		  vaddvq_f32, vmulvq_f32,
		  vminvq_f32, vmaxvq_f32)
  QE_DEFINE_BASIC(double, float64x2_t, vld1q_f64, vld1q_f64,
		  vzeroq_f64, vdupq_n_f64, vst1q_f64, vst1q_f64,
		  vaddq_f64, vsubq_f64, vmulq_f64, vdivq_f64,
		  vsqrtq_f64, vminq_f64, vmaxq_f64)
  QE_DEFINE_HORIZ(double, float64x2_t,
		  vaddvq_f64, vmulvq_f64,
		  vminvq_f64, vmaxvq_f64)
  QE_DEFINE_POW2N_S(float32x4_t, int32x4_t, vreinterpretq_s32_f32,
		    vreinterpretq_f32_s32, vshlq_s32, vdupq_n_s32)
  QE_DEFINE_POW2N_D(float64x2_t, int64x2_t, vreinterpretq_s64_f64,
		    vreinterpretq_f64_s64, vshlq_s64, vdupq_n_s64)
  QE_DEFINE_FMA_ALT(float, float32x4_t, vfmaq_f32, vfmsq_f32)
  QE_DEFINE_FMA_ALT(double, float64x2_t, vfmaq_f64, vfmsq_f64)
  inline bool all_in_range(float32x4_t x, double low_bound, double high_bound) {
    union {
      uint32x2_t v;
      uint32_t data[2];
    };
    uint32x4_t tmp = vandq_u32(vcgeq_f32(x,vdupq_n_f32(low_bound)),
			       vcleq_f32(x,vdupq_n_f32(high_bound)));
    v = vand_u32(vget_low_u32(tmp), vget_high_u32(tmp));
    return data[0] && data[1];
  }
  inline bool all_in_range(float64x2_t x, double low_bound, double high_bound) {
    union {
      uint64x2_t v;
      uint64_t data[2];
    };
    v = vandq_u64(vcgeq_f64(x,vdupq_n_f64(low_bound)),
		  vcleq_f64(x,vdupq_n_f64(high_bound)));
    return data[0] && data[1];
  }

  inline float32x4_t unchecked_round(float32x4_t x) {
    return vcvtq_f32_s32(vcvtaq_s32_f32(x));
  }
  inline float64x2_t unchecked_round(float64x2_t x) {
    return vcvtq_f64_s64(vcvtaq_s64_f64(x));
  }
  inline float32x4_t select_gt(float32x4_t x1, float32x4_t x2,
			       float32x4_t y1, float32x4_t y2) {
    return vbslq_f32(vcgtq_f32(x1,x2), y1, y2);
  }
  inline float64x2_t select_gt(float64x2_t x1, float64x2_t x2,
			       float64x2_t y1, float64x2_t y2) {
    return vbslq_f64(vcgtq_f64(x1,x2), y1, y2);
  }

  inline float unchecked_round(float x)
  { return vgetq_lane_f32(unchecked_round(vdupq_n_f32(x)), 0); }
  inline double unchecked_round(double x)
  { return vgetq_lane_f64(unchecked_round(vdupq_n_f64(x)), 0); }

  inline float pow2n(float x) {
    return vgetq_lane_f32(pow2n(vdupq_n_f32(x)),0);
  }
  inline double pow2n(double x) {
    return vgetq_lane_f64(pow2n(vdupq_n_f64(x)),0);
  }

#endif
 
  
#ifdef QE_HAVE_FAST_EXP
  
  // -------------------------------------------------------------------
  // Implementation of fast exponential
  // -------------------------------------------------------------------

  template<typename Type, typename Vec>
  static inline
  Vec polynomial_5(Vec const x, Type c0, Type c1, Type c2, Type c3, Type c4, Type c5) {
    // calculates polynomial c5*x^5 + c4*x^4 + c3*x^3 + c2*x^2 + c1*x + c0
    using quick_e::fma;
    Vec x2 = mul(x, x);
    Vec x4 = mul(x2, x2);
    return fma(fma(c3, x, c2), x2, fma(fma(c5, x, c4), x4, fma(c1, x, c0)));
  }

  template<typename Vec>
  inline
  Vec fastexp_float(Vec const initial_x) {
    using namespace quick_e;
    using quick_e::unchecked_round;
    using quick_e::fma;
    
    // Taylor coefficients
    const float P0expf   =  1.f/2.f;
    const float P1expf   =  1.f/6.f;
    const float P2expf   =  1.f/24.f;
    const float P3expf   =  1.f/120.f; 
    const float P4expf   =  1.f/720.f; 
    const float P5expf   =  1.f/5040.f; 
    const float VM_LOG2E = 1.44269504088896340736;  // 1/log(2)
    const float ln2f_hi  =  0.693359375f;
    const float ln2f_lo  = -2.12194440e-4f;
#ifndef __FAST_MATH__
    const float min_x    = -87.3f;
    const float max_x    = +89.0f;
#endif

    Vec r = unchecked_round(mul(initial_x,set1<Vec>(VM_LOG2E)));
    Vec x = fnma(r, set1<Vec>(ln2f_hi), initial_x); //  x -= r * ln2f_hi;
    x = fnma(r, set1<Vec>(ln2f_lo), x);             //  x -= r * ln2f_lo;
 
    Vec z = polynomial_5(x,P0expf,P1expf,P2expf,P3expf,P4expf,P5expf);

    Vec x2 = mul(x, x);
    z = fma(z, x2, x);                       // z *= x2;  z += x;

    // multiply by power of 2 
    Vec n2 = pow2n(r);

    z = fma(z,n2,n2);
    
#ifdef __FAST_MATH__
    return z;
#else
    if (all_in_range(initial_x, min_x, max_x)) {
      return z;
    }
    else {
      // When initial_x<-87.3, set exp(x) to -Inf
      z = select_gt(set1<Vec>(min_x), initial_x, set0<Vec>(), z);
      // When initial_x>+89.0, set exp(x) to +Inf
      z = select_gt(initial_x, set1<Vec>(max_x),
		    set1<Vec>(std::numeric_limits<float>::infinity()),
		    z);
      return z;
    }
#endif
  }


  template <typename Type, typename Vec>
  Vec polynomial_13m(Vec const x,
		     Type c2, Type c3, Type c4, Type c5, Type c6, Type c7,
		     Type c8, Type c9, Type c10, Type c11, Type c12, Type c13) {
    // calculates polynomial c13*x^13 + c12*x^12 + ... + x + 0
    using quick_e::fma;
    
    Vec x2 = mul(x, x);
    Vec x4 = mul(x2, x2);
    //    Vec x8 = mul(x4, x4);
    return fma(fma(fma(c13, x, c12), x4,
		   fma(fma(c11, x, c10), x2, fma(c9, x, c8))), mul(x4, x4),
	       fma(fma(fma(c7, x, c6), x2, fma(c5, x, c4)), x4,
		   fma(fma(c3, x, c2), x2, x)));
    //return fma(fma(fma(fma(fma(fma(fma(fma(fma(fma(fma(fma(c13, x, c12), x, c11), x, c10), x, c9), x, c8), x, c7), x, c6), x, c5), x, c4), x, c3), x, c2), mul(x,x), x);
    
  }

  
  // Template function implementing the fast exponential, where Vec
  // can be double, __m128d, __m256d or __m512d
  template <typename Vec>
  inline
  Vec fastexp_double(Vec const initial_x) {
    using namespace quick_e;
    using quick_e::unchecked_round;
    using quick_e::fma;
    
    const double p2  = 1./2.;
    const double p3  = 1./6.;
    const double p4  = 1./24.;
    const double p5  = 1./120.; 
    const double p6  = 1./720.; 
    const double p7  = 1./5040.; 
    const double p8  = 1./40320.; 
    const double p9  = 1./362880.; 
    const double p10 = 1./3628800.; 
    const double p11 = 1./39916800.; 
    const double p12 = 1./479001600.; 
    const double p13 = 1./6227020800.; 
    const double VM_LOG2E = 1.44269504088896340736;  // 1/log(2)
    const double ln2d_hi = 0.693145751953125;
    const double ln2d_lo = 1.42860682030941723212E-6;
#ifndef __FAST_MATH__
    const double min_x = -708.39;
    const double max_x = +709.70;
#endif

    Vec r = unchecked_round(mul(initial_x,set1<Vec>(VM_LOG2E)));
    // subtraction in two steps for higher precision
    Vec x = fnma(r, set1<Vec>(ln2d_hi), initial_x);   //  x -= r * ln2d_hi;
    x = fnma(r, set1<Vec>(ln2d_lo), x);               //  x -= r * ln2d_lo;

    // multiply by power of 2 
    Vec n2 = pow2n(r);
    
    Vec z = polynomial_13m(x, p2, p3, p4, p5, p6, p7,
			   p8, p9, p10, p11, p12, p13);
    z = fma(z,n2,n2);
#ifdef __FAST_MATH__
    return z;
#else
    if (all_in_range(initial_x, min_x, max_x)) {
      // Fast normal path
      return z;
    }
    else {
      // When initial_x<-708.39, set exp(x) to 0.0
      z = select_gt(set1<Vec>(min_x), initial_x, set0<Vec>(), z);
      // When initial_x>+709.70.0, set exp(x) to +Inf
      z = select_gt(initial_x, set1<Vec>(max_x),
		    set1<Vec>(std::numeric_limits<double>::infinity()),
		    z);
      return z;
    }
#endif
  }
#endif
  

  // Define the various overloads for the quick_e::exp function taking
  // Intel intrinsics as an argument

#ifdef __SSE2__
  inline __m128  exp(__m128 x)  { return fastexp_float(x);  }
  inline __m128d exp(__m128d x) { return fastexp_double(x); }
#endif

#ifdef __AVX__
  inline __m256  exp(__m256 x)  { return fastexp_float(x);  }
  inline __m256d exp(__m256d x) { return fastexp_double(x); }
#endif

#ifdef __AVX512F__
  inline __m512  exp(__m512 x)  { return fastexp_float(x);  }
  inline __m512d exp(__m512d x) { return fastexp_double(x); }
#endif

#ifdef QE_HAVE_ARM64_NEON
  inline float32x4_t exp(float32x4_t x) { return fastexp_float(x);  }
  inline float64x2_t exp(float64x2_t x) { return fastexp_double(x); }
#endif

  // Define the quick_e::exp function for scalar arguments
#ifdef QE_HAVE_FAST_EXP
  inline float  exp(float x)  { return quick_e::fastexp_float(x); }
  inline double exp(double x) { return quick_e::fastexp_double(x); }
#else
  // If no vectorization available then we fall back to the standard
  // library scalar version
  inline float  exp(float x)  { return std::exp(x); }
  inline double exp(double x) { return std::exp(x); }
#endif

#undef QE_DEFINE_TRAITS
#undef QE_DEFINE_LONGEST
#undef QE_DEFINE_BASIC
#undef QE_DEFINE_CHOP
#undef QE_DEFINE_HORIZ
#undef QE_DEFINE_FMA
#undef QE_DEFINE_FMA_ALT
#undef QE_EMULATE_FMA
#undef QE_DEFINE_POW2N_S
#undef QE_DEFINE_POW2N_D
#undef QE_HAVE_FAST_EXP
#undef QE_HAVE_ARM64_NEON
}

#endif
