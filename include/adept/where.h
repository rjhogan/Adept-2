/* where.h -- Support for Fortran-90-like "where" construct

    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

  
   Consider the following:

     A.where(B) = C;
     A.where(B) = either_or(C, D);

   where A is an Array, B is a boolean expression, and C and D are
   expressions, and the arrays and expressions have the same rank and
   size, except that C and or D may have rank zero. The first line has
   the effect of setting every element of A for which B is true to the
   corresponding value in C. The second line does this but for
   elements where B is false it sets A instead to D.

   

*/


#ifndef AdeptWhere_H
#define AdeptWhere_H 1

#include <vector>

#include <adept/Expression.h>

namespace adept {

  namespace internal {


    // ---------------------------------------------------------------------
    // Section 1. EitherOr object returned by either_or function
    // ---------------------------------------------------------------------
    template <class C, class D>
    class EitherOr {
    public:
      typedef bool _adept_either_or_flag;
      EitherOr(const C& c, const D& d) : either_(c), or_(d) { }
      const C& value_if_true() const { return either_; }
      const D& value_if_false() const { return or_; }
    protected:
      const C& either_;
      const D& or_;
    };


    template <typename T>
    struct is_not_either_or
    {
    private:
      typedef char yes;
      typedef struct { char array[2]; } no;
      template <typename C> static yes test(typename C::_adept_either_or_flag*);
      template <typename C> static no  test(...);
    public:
      static const bool value = sizeof(test<T>(0)) != sizeof(yes);
    };


    // ---------------------------------------------------------------------
    // Section 2. Where class returned by A.where(B)
    // ---------------------------------------------------------------------
    template <class A, class B>
    class Where {
    public:
      Where(A& a, const B& b) : array_(a), bool_expr_(b) { }

      template <class C>
      typename enable_if<is_not_either_or<C>::value, Where&>::type
      operator=(const C& c) {
	array_.assign_conditional(bool_expr_, c);
	return *this;
      }

      // With either_or on the right-hand-side: this implementation
      // could be faster if bool_expr was not evaluated twice
      template <class C>
      typename enable_if<!is_not_either_or<C>::value, Where&>::type
      operator=(const C& c) {
	array_.assign_conditional(!const_cast<B&>(bool_expr_), c.value_if_false());
	array_.assign_conditional(bool_expr_,  c.value_if_true());
	return *this;
      }

#define ADEPT_WHERE_OPERATOR(EQ_OP, OP)					\
      template <class C>						\
      typename enable_if<is_not_either_or<C>::value, Where&>::type	\
      EQ_OP(const C& c) {						\
	array_.assign_conditional(bool_expr_, noalias(*this) OP c);	\
        return *this;							\
      }									\
      template <class C>						\
      typename enable_if<!is_not_either_or<C>::value, Where&>::type	\
      EQ_OP(const C& c) {						\
	array_.assign_conditional(!const_cast<B&>(bool_expr_),		\
				  noalias(*this) OP c.value_if_false()); \
	array_.assign_conditional(bool_expr_,				\
				  noalias(*this) OP c.value_if_true()); \
	return *this;							\
      }									
      ADEPT_WHERE_OPERATOR(operator+=, +)
      ADEPT_WHERE_OPERATOR(operator-=, -)
      ADEPT_WHERE_OPERATOR(operator*=, *)
      ADEPT_WHERE_OPERATOR(operator/=, /)
#undef ADEPT_WHERE_OPERATOR

    protected:
      A& array_;
      const B& bool_expr_;

    };

  } // end namespace internal



  template <class C, class D>
  EitherOr<C,D> either_or(const C& c, const D& d) {
    return EitherOr<C,D>(c, d);
  }

} // end namespace adept

#endif 
