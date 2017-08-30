/* traits.h -- Traits used to support array/automatic differentiation expressions

    Copyright (C) 2012-2014 University of Reading
    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptTraits_H
#define AdeptTraits_H 1

#include <complex>
#include <limits>
#include <iostream>

#include <adept/base.h>

namespace adept {

  // Forward declaration of "Active"
  template <typename T> class Active;


  // All traits are in the adept::internal namespace
  namespace internal {

    // ----- CONTENTS -----
    // 1. ADEPT_STATIC_ASSERT
    // 2. enable_if
    // 3. if_then_else
    // 4. is_not_expression
    // 5. is_complex
    // 6. is_active
    // 7. is_array
    // 8. is_scalar_int
    // 9. all_scalar_ints
    // 10. underlying_real
    // 11. underlying_passive
    // 12. promote
    // 13. rank_compatible
    // 14. is_same
    // 15. remove_reference
    // --------------------

    // ---------------------------------------------------------------------
    // 1. ADEPT_STATIC_ASSERT
    // ---------------------------------------------------------------------

    // Heavily templated C++ code as in the Adept library can produce
    // very long and cryptic compiler error messages. This macro is
    // useful to check for conditions that should not happen. It check
    // a bool known at compile time is true, otherwise fail to compile
    // with a message that is hopefully understandable.
    // E.g. ADEPT_STATIC_ASSERT(0 > 1, ZERO_IS_NOT_GREATER_THAN_ONE)
    // would fail at compile time with a message containing
    // ERROR_ZERO_IS_NOT_GREATER_THAN_ONE, which should hopefully
    // stand out even in a long error message.

    // Helper class
    template<bool> struct compile_time_check 
    { typedef int STATIC_ASSERTION_HAS_FAILED; };
    template<> struct compile_time_check<false> { };

    // Define the macro in which a struct is defined that inherits
    // from compile_time_check
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic warning "-Wpragmas"
#endif
#define ADEPT_STATIC_ASSERT(condition, msg)				\
    do { struct ERROR_##msg : public ::adept::internal::compile_time_check<(condition)> { }; \
	typedef typename ERROR_##msg ::STATIC_ASSERTION_HAS_FAILED type; \
    } while (0)


    // ---------------------------------------------------------------------
    // 2. enable_if
    // ---------------------------------------------------------------------

    // To enable a function "Type function()" only if CONDITION is
    // true, replace "Type" in the function declaration with "typename
    // enable_if<CONDITIONAL,Type>::type"
    template <bool, typename T = void> struct enable_if { };
    // Partial specialization for true.
    template <typename T> struct enable_if<true, T> { typedef T type; };


    // ---------------------------------------------------------------------
    // 3. if_then_else
    // ---------------------------------------------------------------------

    // "if_then_else<CONDITION, YES, NO>::type" resolves to YES if
    // CONDITION is "true", NO otherwise. A limitation is that both Y
    // and N must be valid types
    template <bool, typename Y, typename N>
    struct if_then_else { typedef Y type; };

    template <typename Y, typename N>
    struct if_then_else<false, Y, N> { typedef N type; };


    // ---------------------------------------------------------------------
    // 4. is_not_expression
    // ---------------------------------------------------------------------

    // The following enables us to provide functions that work only on
    // types *not* derived from the Expression struct:
    // "is_not_expression<E>::value" is "false" if E is not an
    // expression and "true" otherwise
    template <typename T>
    struct is_not_expression
    {
    private:
      typedef char yes;
      typedef struct { char array[2]; } no;
      template <typename C> static yes test(typename C::_adept_expression_flag*);
      template <typename C> static no  test(...);
    public:
      static const bool value = sizeof(test<T>(0)) != sizeof(yes);
    };


    // ---------------------------------------------------------------------
    // 5. is_complex
    // ---------------------------------------------------------------------

    // Test for complex numbers: "is_complex<S>::value" is "true" if S
    // is complex, "false" otherwise
    template <typename> struct is_complex
    { static const bool value = false; };
    template <> struct is_complex<std::complex<float> > 
    { static const bool value = true; };
    template <> struct is_complex<std::complex<double> > 
    { static const bool value = true; };
    template <> struct is_complex<std::complex<long double> > 
    { static const bool value = true; };


    // ---------------------------------------------------------------------
    // 6. is_active
    // ---------------------------------------------------------------------

    // Test for active numbers: "is_active<S>::value" is "true" if S
    // is active, "false" otherwise.
    // Then the default case for non-expressions returns false
    
    template <typename T, class Enable = void>
    struct is_active { };

    template <typename T>
    struct is_active<T, typename enable_if<is_not_expression<T>::value>::type>
    { static const bool value = false; };
    
    // Expressions define a static const bool called "is_active"
    template <typename T>
    struct is_active<T, typename enable_if<!is_not_expression<T>::value>::type>
    { static const bool value = T::is_active; };
    

    // ---------------------------------------------------------------------
    // 7. is_array
    // ---------------------------------------------------------------------
    
    // "is_array<E>::value" is "true" if E is an array expression and
    // "false" otherwise.  The default case for non-expressions
    // returns false
    template <typename T, class Enable = void>
    struct is_array { };
    template <typename T>
    struct is_array<T, typename enable_if<is_not_expression<T>::value>::type>
    { static const bool value = false; };
    // Expressions define a static const bool called "is_array"
    template <typename T>
    struct is_array<T, typename enable_if<!is_not_expression<T>::value>::type>
    { static const bool value = T::is_array; };
    

    // ---------------------------------------------------------------------
    // 8. is_scalar_int
    // ---------------------------------------------------------------------

    // Return whether template argument is of integer type, or is a
    // 0-dimensional expression of integer type
    template <typename T, class Enable = void>
    struct is_scalar_int { };
    
    template <typename T>
    struct is_scalar_int<T, 
	      typename enable_if<is_not_expression<T>::value>::type> {
      static const bool value = std::numeric_limits<T>::is_integer;
      static const int  count = value;
    };
    
    template <typename T>
    struct is_scalar_int<T, 
	      typename enable_if<!is_not_expression<T>::value>::type>
    {
      static const bool value
      = std::numeric_limits<typename T::type>::is_integer
	&& T::rank == 0; 
      static const int  count = value;
    };


    // ---------------------------------------------------------------------
    // 9. all_scalar_ints
    // ---------------------------------------------------------------------

    // all_scalar_ints<Rank,I0,I1...>::value returns true if I[0] to
    // I[Rank-1] are all scalar integers

    // First define a "null" type
    struct null_type { };
    template <typename T> struct is_null_type { 
      static const bool value = false; 
      static const int  count = 0; 
    };
    template <> struct is_null_type<null_type>{
      static const bool value = true; 
      static const int  count = 1;
    };

    template <int Rank, typename I0, typename I1 = null_type, 
	      typename I2 = null_type, typename I3 = null_type,
	      typename I4 = null_type, typename I5 = null_type,
	      typename I6 = null_type>
    struct all_scalar_ints {
      static const bool value = (Rank == (is_scalar_int<I0>::count
					  +is_scalar_int<I1>::count
					  +is_scalar_int<I2>::count
					  +is_scalar_int<I3>::count
					  +is_scalar_int<I4>::count
					  +is_scalar_int<I5>::count
					  +is_scalar_int<I6>::count));
    };



    // ---------------------------------------------------------------------
    // 10. underlying_real
    // ---------------------------------------------------------------------
  
    // Return the underlying real type for a complex argument:
    // "underlying_real<S>::type returns T if S is of type
    // std::complex<T>, or returns S if it is not complex
    template <typename T>
    struct underlying_real
    {
    private:
      template <bool, typename S>
      struct _underlying_real
      { typedef S type; };
      template <typename S>
      struct _underlying_real<true, S>
      { typedef typename S::type type; };
    public:
      typedef typename _underlying_real<is_complex<T>::value,
					T>::type type;
    };
    // ---------------------------------------------------------------------
    // 11. underlying_passive
    // ---------------------------------------------------------------------
  
    // Return the underlying passive type for an active argument:
    // "underlying_passive<S>::type returns T if S is of type
    // adept::Active<T>, or returns S if it is not active.
    template <typename T>
    struct underlying_passive
    {
    private:
      template <bool, typename S>
      struct _underlying_passive
      { typedef S type; };
      template <typename S>
      struct _underlying_passive<true, S>
      { typedef typename S::type type; };
    public:
      typedef typename _underlying_passive<is_active<T>::value,
					T>::type type;
    };
    

    // ---------------------------------------------------------------------
    // 12. promote
    // ---------------------------------------------------------------------
  
    // "promote<L,R>::type" returns the type that a binary operation
    // (e.g. multiplication) between types L and R should result in.
    // Note that "complexity" and "precision" are promoted separately,
    // so double + std::complex<float> will result in an object of
    // type std::complex<double> >.
    template <typename L, typename R>
    struct promote {
    private:
      template <typename A, typename B>
      struct promote_primitive {
	static const bool A_bigger_than_B = (sizeof(A) > sizeof(B));
	static const bool A_float_B_int = (!std::numeric_limits<A>::is_integer) 
	  && std::numeric_limits<B>::is_integer;
	static const bool A_int_B_float = std::numeric_limits<A>::is_integer
	  && (!std::numeric_limits<B>::is_integer);
	static const bool prefer_float = A_float_B_int || A_int_B_float;
	typedef typename if_then_else<A_float_B_int, A, B>::type float_type;
	typedef typename if_then_else<A_bigger_than_B, A, B>::type biggest_type;
	typedef typename if_then_else<prefer_float, float_type, biggest_type>::type type;
      };
      
      typedef typename promote_primitive<
        typename underlying_real<typename underlying_passive<L>::type>::type,
	typename underlying_real<typename underlying_passive<R>::type>::type>::type real;
      typedef typename if_then_else<is_complex<L>::value
				    || is_complex<R>::value,
				    std::complex<real>,
				    real>::type complex_type;
    public: 
      typedef typename if_then_else<is_active<L>::value || is_active<R>::value,
				    adept::Active<complex_type>, 
				    complex_type>::type type;
    };

  
    // ---------------------------------------------------------------------
    // 13. rank_compatible
    // ---------------------------------------------------------------------

    // Check that an array of rank LRank could enter an operation
    // (e.g. addition) with an array of rank RRank: the two ranks must
    // either be the same, or either can be zero
    template <int LRank, int RRank>
    struct rank_compatible {
      static const bool value = (LRank == RRank || LRank == 0 || RRank == 0);
    };


    // ---------------------------------------------------------------------
    // 14. is_same
    // ---------------------------------------------------------------------

    // Compare two types to see if they're the same
    template<typename T, typename U>
    struct is_same { static const bool value = false;  };
    
    template<typename T>
    struct is_same<T,T>  { static const bool value = true; };
    

    // ---------------------------------------------------------------------
    // 15. remove_reference
    // ---------------------------------------------------------------------

    // Remove reference from a type if present
    template<typename T>  struct remove_reference { typedef T type; };
    template<typename T>  struct remove_reference<T&> { typedef T type; };

  } // End namespace internal

} // End namespace adept



#endif