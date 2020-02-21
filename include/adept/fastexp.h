/* fastexp.h -- Fast exponential, vectorized

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptFastexp_H
#define AdeptFastexp_H 1


namespace adept {

  namespace internal {

    // Functions to return 2^n
#ifdef __AVX512F__
    // Was Vec8d
    inline __m512d vm_pow2n(__m512d n) {
      const double pow2_52 = 4503599627370496.0;   // 2^52
      const double bias = 1023.0;                  // bias in exponent
      __m512d a = n + _mm512_set1_pd(bias+pow2_52);// put n + bias in least significant bits
      __m512i b = _mm512_castpd_si512(a);          // bit-cast to integer
      // shift left 52 places to get value into exponent field
      __m512i c = _mm512_sll_epi64(b, _mm_cvtsi32_si128(52));
      __m512d d = _mm512_castsi512_pd(c);          // bit-cast back to double
      return d;
    }

#endif
    
#ifdef __AVX__
    // Was Vec4d
    inline __m256d vm_pow2n(__m256d n) {
      const double pow2_52 = 4503599627370496.0;   // 2^52
      const double bias = 1023.0;                  // bias in exponent
      __m256d a = n + _mm256_set1_pd(bias+pow2_52);// put n + bias in least significant bits
#ifdef __AVX2__
      __m256i b = _mm256_castpd_si256(a);          // bit-cast to integer
      // shift left 52 places to get value into exponent field
      __m256i c = _mm256_sll_epi64(b, _mm_cvtsi32_si128(52));
#else
      // This is completely unoptimized but it works
      union {
	__m256i b;
	int64_t b_data[4];
      };
      b = _mm256_castpd_si256(a);          // bit-cast to integer
      union {
	__m256i c;
	int64_t c_data[4];
      };
      for (int i = 0; i < 4; ++i) {
	c_data[i] = b_data[i] << 52;
      }
#endif      
      __m256d d = _mm256_castsi256_pd(c);          // bit-cast back to double
      return d;
    }
#endif

#ifdef __SSE2__
    // Was Vec2d
    inline __m128d vm_pow2n(__m128d n) {
      const double pow2_52 = 4503599627370496.0;   // 2^52
      const double bias = 1023.0;                  // bias in exponent
      __m128d a = n + _mm_set1_pd(bias+pow2_52);// put n + bias in least significant bits
      __m128i b = _mm_castpd_si128(a);          // bit-cast to integer
      // shift left 52 places to get value into exponent field
      __m128i c = _mm_sll_epi64(b, _mm_cvtsi32_si128(52));
      __m128d d = _mm_castsi128_pd(c);          // bit-cast back to double
      return d;
    }
#endif

    /*
    static inline Vec8f vm_pow2n (Vec8f const n) {
      const float pow2_23 =  8388608.0;            // 2^23
      const float bias = 127.0;                    // bias in exponent
      Vec8f a = n + (bias + pow2_23);              // put n + bias in least significant bits
      Vec8i b = reinterpret_i(a);                  // bit-cast to integer
      Vec8i c = b << 23;                           // shift left 23 places to get into exponent field
      Vec8f d = reinterpret_f(c);                  // bit-cast back to float
      return d;
    }
    */

    inline float fma(float a, float b, float c) { return (a*b)+c; };
    inline double fma(double a, double b, double c) { return (a*b)+c; };
    inline float fnma(float a, float b, float c) { return -(a*b)+c; };
    inline double fnma(double a, double b, double c) { return -(a*b)+c; };

    template <typename PType>
    inline PType pow2n(PType n) { return n.data; }
    template <>
    inline double pow2n<double>(double n) {
      union {
	double ans;
	__m128d d;
      };
      d = vm_pow2n(_mm_loadu_pd(&n));
      return ans;
    }      

    template <typename Type, typename PType>
    PType polynomial_13m(PType const x,
	        Type c2, Type c3, Type c4, Type c5, Type c6, Type c7,
		Type c8, Type c9, Type c10, Type c11, Type c12, Type c13) {
      // calculates polynomial c13*x^13 + c12*x^12 + ... + x + 0
      PType x2 = x  * x;
      PType x4 = x2 * x2;
      PType x8 = x4 * x4;
      return fma(fma(fma(c13, x, c12), x4,
		     fma(fma(c11, x, c10), x2, fma(c9, x, c8))), x8,
		 fma(fma(fma(c7, x, c6), x2, fma(c5, x, c4)), x4,
		     fma(fma(c3, x, c2), x2, x)));
    }

    // PType can either be Packet<double> or double
    template <typename PType>
    inline
    PType fastexp_double(PType initial_x) {
      using std::round;

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

      // data vectors
      PType  x, r, z, n2;

      const double ln2d_hi = 0.693145751953125;
      const double ln2d_lo = 1.42860682030941723212E-6;
      x = initial_x;
      r = round(initial_x*PType(VM_LOG2E));
      // subtraction in two steps for higher precision
      x = fnma(r, PType(ln2d_hi), x);             //  x -= r * ln2d_hi;
      x = fnma(r, PType(ln2d_lo), x);             //  x -= r * ln2d_lo;

      z = polynomial_13m(x, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13);

      // multiply by power of 2 
      //n2.data = vm_pow2n(r.data);
      n2 = pow2n(r);

      z = (z + PType(1.0)) * n2;

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
    inline
    Packet<double> fastexp(Packet<double> x) { return fastexp_double(x); }
  }
  inline float fastexp(float x) { return std::exp(x); }
  inline double fastexp(double x) { return internal::fastexp_double(x); }
}



#endif
