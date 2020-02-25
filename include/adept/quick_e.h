/* quick_e.h -- Fast exponential function for Intel intrinsics

   Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

   Author: Robin Hogan <r.j.hogan@ecmwf.int>

   This file is part of the Adept library, although can be used
   stand-alone.

   The exponential function for real arguments is used in many areas
   of physics, yet is not vectorized by many compilers.  This C++
   header file provides a fast exponential function (quick_e::exp) for
   single and double precision floating point numbers, as well as for
   Intel intrinsics representing packets of 2, 4, 8 and 16 such
   numbers.  The algorithm has been taken from Agner Fog's Vector
   Class Library. It is designed to be used in other libraries that
   make use of Intel intrinsics.  Since such libraries often define
   their own classes for representing vectors of numbers, this file
   does not define any such classes itself.

   Also in the namespace quick_e, this file defines the following
   inline functions that work on Intel intrinsics of type "Vec" and
   the corresponding scalar type "Sca":

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

// Headers needed for x86 vector intrinsics
#ifdef __SSE2__
  #include <xmmintrin.h> // SSE
  #include <emmintrin.h> // SSE2
  // Numerous platforms don't define _mm_undefined_ps in xmmintrin.h,
  // so we assume none do, except GCC >= 4.9.1 and CLANG >= 3.8.0.
  // Those that don't use an equivalent function that sets the
  // elements to zero.
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
#else
  // No vectorization available: longest packet is of size 1
  QE_DEFINE_LONGEST(float, double);
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
  
  template <typename T> T fma(T x, T y, T z)  { return (x*y)+z; }
  template <typename T> T fnma(T x, T y, T z)  { return z-(x*y); }
  template <typename T> T fmin(T x, T y)  { return std::min(x,y); }
  template <typename T> T fmax(T x, T y)  { return std::min(x,y); }
 
#if __cplusplus > 199711L
  template <> inline float fmin(float x, float y)  { return std::fmin(x,y); }
  template <> inline double fmin(double x, double y)  { return std::fmin(x,y); }
  template <> inline float fmax(float x, float y)  { return std::fmax(x,y); }
  template <> inline double fmax(double x, double y)  { return std::fmax(x,y); }
#endif
  
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
  template <> inline VEC set0<VEC>()        { return SET0();    }	\
  template <> inline VEC set1<VEC>(TYPE x)  { return SET1(x);   }		\
  inline VEC sqrt(VEC x)             { return SQRT(x);   }		\
  inline VEC fmin(VEC x, VEC y)      { return FMIN(x,y); }		\
  inline VEC fmax(VEC x, VEC y)      { return FMAX(x,y); }		\
  template <> inline VEC load<VEC,TYPE>(const TYPE* d){ return LOAD(d);   } \
  template <> inline VEC loadu<VEC,TYPE>(const TYPE* d){ return LOADU(d); }	\
  inline void store(TYPE* d, VEC x)  { STORE(d, x);      }	\
  inline void storeu(TYPE* d, VEC x) { STORE(d, x);      }	\
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

#define QE_DEFINE_POW2N_S(VEC, VECI, CASTTO, CASTBACK, SHIFTL)	\
  inline VEC pow2n(VEC n) {					\
    const float pow2_23 = 8388608.0;				\
    const float bias = 127.0;					\
    VEC  a = add(n, set1<VEC>(bias+pow2_23));			\
    VECI b = CASTTO(a);						\
    VECI c = SHIFTL(b, _mm_cvtsi32_si128(23));			\
    VEC  d = CASTBACK(c);					\
    return d;							\
  }
#define QE_DEFINE_POW2N_D(VEC, VECI, CASTTO, CASTBACK, SHIFTL)	\
  inline VEC pow2n(VEC n) {					\
    const double pow2_52 = 4503599627370496.0;			\
    const double bias = 1023.0;					\
    VEC  a = add(n, set1<VEC>(bias+pow2_52));			\
    VECI b = CASTTO(a);						\
    VECI c = SHIFTL(b, _mm_cvtsi32_si128(52));			\
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
  // Don't define chop operations for __m128
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
    __m128 shuftmp= _mm_movehl_ps(ADEPT_MM_UNDEFINED_PS(),		\
				  _mm_castpd_ps(x));			\
    __m128d shuf  = _mm_castps_pd(shuftmp);				\
    return  _mm_cvtsd_f64(OP_PD(x, shuf));				\
  }
  QE_DEFINE_HORIZ_SSE2(hsum, _mm_add_ps, _mm_add_ss, _mm_add_pd)
  QE_DEFINE_HORIZ_SSE2(hmul, _mm_mul_ps, _mm_mul_ss, _mm_mul_pd)
  QE_DEFINE_HORIZ_SSE2(hmin, _mm_min_ps, _mm_min_ss, _mm_min_pd)
  QE_DEFINE_HORIZ_SSE2(hmax, _mm_max_ps, _mm_max_ss, _mm_max_pd)

#undef ADEPT_MM_UNDEFINED_PS
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
  inline __m128 unchecked_round(__m128 x) {
    return _mm_cvtepi32_ps(_mm_cvtps_epi32(x));
  }
  inline __m128d unchecked_round(__m128d x) {
    return _mm_cvtepi32_pd(_mm_cvtpd_epi32(x));
  }

#endif
  inline float unchecked_round(float x) { return _mm_cvtss_f32(unchecked_round(_mm_set_ss(x))); }
  inline double unchecked_round(double x) { return low(unchecked_round(_mm_set_sd(x))); }

  QE_DEFINE_POW2N_S(__m128, __m128i, _mm_castps_si128,
		    _mm_castsi128_ps, _mm_sll_epi32)
  QE_DEFINE_POW2N_D(__m128d, __m128i, _mm_castpd_si128,
		    _mm_castsi128_pd, _mm_sll_epi64)
  inline float pow2n(float x) { return _mm_cvtss_f32(pow2n(quick_e::set1<__m128>(x))); }
  inline double pow2n(double x) { return low(pow2n(quick_e::set1<__m128d>(x))); }

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
  inline float  hmul(__m256 x)  { return hmul(add(low(x), high(x))); }
  inline float  hmin(__m256 x)  { return hmin(add(low(x), high(x))); }
  inline float  hmax(__m256 x)  { return hmax(add(low(x), high(x))); }
  inline double hsum(__m256d x) { return hsum(add(low(x), high(x))); } // Alternative would be to use _mm_hadd_pd
  inline double hmul(__m256d x) { return hmul(add(low(x), high(x))); }
  inline double hmin(__m256d x) { return hmin(add(low(x), high(x))); }
  inline double hmax(__m256d x) { return hmax(add(low(x), high(x))); }
  
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
		      _mm256_castsi256_ps, _mm256_sll_epi32)
    QE_DEFINE_POW2N_D(__m256d, __m256i, _mm256_castpd_si256,
		      _mm256_castsi256_pd, _mm256_sll_epi64)
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
		    _mm512_castsi512_ps, _mm512_sll_epi32)
  QE_DEFINE_POW2N_D(__m512d, __m512i, _mm512_castpd_si512,
		    _mm512_castsi512_pd, _mm512_sll_epi64)
#endif

#ifdef __SSE2__
  
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
    //return (c2+c3*x)*x2 + ((c4+c5*x)*x4 + (c0+c1*x));
    return fma(fma(c3, x, c2), x2, fma(fma(c5, x, c4), x4, fma(c1, x, c0)));
  }

  template<typename Vec>
  inline
  Vec fastexp_float(Vec x) {
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

    // maximum abs(x), value depends on BA, defined below
    // The lower limit of x is slightly more restrictive than the upper limit.
    // We are specifying the lower limit, except for BA = 1 because it is not used for negative x
    //    float max_x;
    //        max_x = (BA == 0) ? 87.3f : 89.0f;

    const float ln2f_hi  =  0.693359375f;
    const float ln2f_lo  = -2.12194440e-4f;

    Vec r = unchecked_round(mul(x,set1<Vec>(VM_LOG2E)));
    x = fnma(r, set1<Vec>(ln2f_hi), x);      //  x -= r * ln2f_hi;
    x = fnma(r, set1<Vec>(ln2f_lo), x);      //  x -= r * ln2f_lo;
 
    Vec z = polynomial_5(x,P0expf,P1expf,P2expf,P3expf,P4expf,P5expf);    
    Vec x2 = mul(x, x);
    z = fma(z, x2, x);                       // z *= x2;  z += x;

    // multiply by power of 2 
    Vec n2 = pow2n(r);

    //z = (z + 1.0f) * n2;
    z = fma(z,n2,n2);
    return z;
    /*
    // check for overflow
    auto inrange  = abs(initial_x) < max_x;      // boolean vector
    // check for INF and NAN
    inrange &= is_finite(initial_x);

    if (horizontal_and(inrange)) {
        // fast normal path
        return z;
    }
    else {
        // overflow, underflow and NAN
        r = select(sign_bit(initial_x), 0.f-(M1&1), infinite_vec<Vec>()); // value in case of +/- overflow or INF
        z = select(inrange, z, r);                         // +/- underflow
        z = select(is_nan(initial_x), initial_x, z);       // NAN goes through
        return z;
    }
    */
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
    // Slower...
    //return fma(fma(fma(fma(fma(fma(fma(fma(fma(fma(fma(fma(c13, x, c12), x, c11), x, c10), x, c9), x, c8), x, c7), x, c6), x, c5), x, c4), x, c3), x, c2), mul(x,x), x);
    
  }

  
  // Template function implementing the fast exponential, where PType
  // can be double, __m128d, __m256d or __m512d
  template <typename Vec>
  inline
  Vec fastexp_double(Vec x) {
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
    
    // maximum abs(x)
    //      double max_x = 708.39;
    
    const double ln2d_hi = 0.693145751953125;
    const double ln2d_lo = 1.42860682030941723212E-6;
    Vec r = unchecked_round(mul(x,set1<Vec>(VM_LOG2E)));
    // subtraction in two steps for higher precision
    x = fnma(r, set1<Vec>(ln2d_hi), x);   //  x -= r * ln2d_hi;
    x = fnma(r, set1<Vec>(ln2d_lo), x);   //  x -= r * ln2d_lo;

    // multiply by power of 2 
    //n2.data = vm_pow2n(r.data);
    Vec n2 = pow2n(r);
    
    Vec z = polynomial_13m(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);
    
    //z = (z + set1<Vec>(1.0)) * n2;
    z = fma(z,n2,n2);
    
    /*
    // check for overflow
    auto inrange  = abs(initial_x) < max_x;      // boolean vector
    // check for INF and NAN
    inrange &= is_finite(initial_x);
    
    if (horizontal_and(inrange)) {
    // fast normal path
    return z;
    }
    else {
    // overflow, underflow and NAN
    r = select(sign_bit(initial_x), 0.-(M1&1), infinite_vec<VTYPE>()); // value in case of +/- overflow or INF
    z = select(inrange, z, r);                         // +/- underflow
    z = select(is_nan(initial_x), initial_x, z);       // NAN goes through
    return z;
    }
    */
    return z;
  }
#endif
  
#ifdef __SSE2__
  inline __m128 exp(__m128 x) { return fastexp_float(x); }
  inline __m128 fastexp(__m128 x) { return fastexp_float(x); }
  inline __m128d exp(__m128d x) { return fastexp_double(x); }
  inline __m128d fastexp(__m128d x) { return fastexp_double(x); }
#endif

#ifdef __AVX__
  inline __m256 exp(__m256 x) { return fastexp_float(x); }
  inline __m256 fastexp(__m256 x) { return fastexp_float(x); }
  inline __m256d exp(__m256d x) { return fastexp_double(x); }
  inline __m256d fastexp(__m256d x) { return fastexp_double(x); }
#endif

#ifdef __AVX512F__
  inline __m512 exp(__m512 x) { return fastexp_float(x); }
  inline __m512 fastexp(__m512 x) { return fastexp_float(x); }
  inline __m512d exp(__m512d x) { return fastexp_double(x); }
  inline __m512d fastexp(__m512d x) { return fastexp_double(x); }
#endif

#ifdef __SSE2__
  //  inline double exp(double x) { return quick_e::fastexp_double(x); }
  inline float fastexp(float x) { 
    return quick_e::fastexp_float(x); 
  }
  inline double fastexp(double x) { 
    //asm("### FASTEXP START");
    return quick_e::fastexp_double(x); 
    //asm("### FASTEXP END");
    //    return ans;
  }
#else
  // If no vectorization available then we fall back to the standard
  // library scalar version

  //  inline double exp(double x) { return quick_e::fastexp_double(x); }
  inline float fastexp(float x) { 
    return std::exp(x); 
  }
  inline double fastexp(double x) { 
    return std::exp(x);
  }
#endif
  
}

#endif
