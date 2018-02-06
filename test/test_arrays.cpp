/* test_arrays.cpp - Test Adept's array functionality

    Copyright (C) 2016-2018 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

  This program can be compiled to run in three ways: (1) normal
  compilation tests inactive arrays, (2) with -DALL_ACTIVE tests
  active arrays, and (3) "-DALL_ACTIVE -DADEPT_RECORDING_PAUSABLE"
  tests whether a "paused" recording correctly records nothing to the
  automatic-differentiation stack.

*/

#include <iostream>
#include <complex>

#define ADEPT_BOUNDS_CHECKING 1

#include <adept_arrays.h>


// The following controls whether to use active variables or not
//#define ALL_ACTIVE 1
//#define MARVEL_STYLE 1
//#define ALL_COMPLEX 1

using namespace adept;


int
main(int argc, const char** argv) {
  using namespace adept;
#ifdef ALL_ACTIVE
#define IsActive true
  Stack stack;
#else
#define IsActive false
#endif
  
#define HEADING(MESSAGE)						\
  std::cout << "====================================================================\n" \
	    << "   TESTING " << MESSAGE << "\n"

#define COMMA ,


#define SIMPLE_EVAL(MESSAGE, TYPE, X, INIT, EXPR)			\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n";	\
  try {								\
    TYPE X;								\
    if (INIT) {								\
      X = test. X;							\
    }									\
    std::cout << "Evaluating " << #EXPR << "\n";			\
    std::cout.flush();						\
    EXPR;								\
    if (should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	\
      anomalous_results++;						\
    }									\
  } catch (adept::exception e) {					\
    std::cout << "*** Failed with: " << e.what() << "\n";		\
    if (!should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	\
      anomalous_results++;						\
    }									\
    else {								\
      std::cout << "*** Correct behaviour\n";				\
    }									\
  }

#define EVAL(MESSAGE, TYPE, X, INIT, EXPR)				\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n";	\
  try {								\
    TYPE X;								\
    if (INIT) {								\
      X = test. X;							\
      std::cout << #TYPE << " " << #X << " = " << X << "\n";		\
    }									\
    else {								\
      std::cout << #TYPE << " " << #X << " = " << X << "\n";		\
    }									\
    std::cout << "Evaluating " << #EXPR << "\n";			\
    std::cout.flush();						\
    EXPR;								\
    std::cout << "Result: " << #X << " = " << X << "\n";		\
    if (should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	\
      anomalous_results++;						\
    }									\
  } catch (adept::exception e) {					\
    std::cout << "*** Failed with: " << e.what() << "\n";		\
    if (!should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	\
      anomalous_results++;						\
    }									\
    else {								\
      std::cout << "*** Correct behaviour\n";				\
    }									\
  }

#define EVAL2(MESSAGE, TYPEX, X, INITX, TYPEY, Y, EXPR)			\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n";	\
  try {									\
    TYPEX X;								\
    if (INITX) {							\
      X = test. X;							\
      std::cout << #TYPEX << " " << #X << " = " << X << "\n";		\
    }									\
    else {								\
      std::cout << #TYPEX << " " << #X << " = " << X << "\n";		\
    }									\
    TYPEY Y; Y = test. Y;						\
    std::cout << #TYPEY << " " << #Y << " = " << Y << "\n";		\
    std::cout << "Evaluating " << #EXPR << "\n";			\
    std::cout.flush();							\
    EXPR;								\
    std::cout << "Result: " << #X << " = " << X << "\n";		\
    if (should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	        \
      anomalous_results++;						\
    }									\
  } catch (adept::exception e) {					\
    std::cout << "*** Failed with: " << e.what() << "\n";		\
    if (!should_fail) { std::cout << "*** INCORRECT OUTCOME\n";		\
      anomalous_results++;						\
    }									\
    else {								\
      std::cout << "*** Correct behaviour\n";				\
    }									\
  }


#define EVAL3(MESSAGE, TYPEX, X, INITX, TYPEY, Y, TYPEZ, Z, EXPR)	\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n"; \
  try {									\
    TYPEX X;								\
    if (INITX) {							\
      X = test. X;							\
      std::cout << #TYPEX << " " << #X << " = " << X << "\n";		\
    }									\
    else {								\
      std::cout << #TYPEX << " " << #X << " = " << X << "\n";		\
    }									\
    TYPEY Y; Y.link( test. Y );						\
    TYPEZ Z; Z.link( test. Z );						\
    std::cout << #TYPEY << " " << #Y << " = " << Y << "\n";		\
    std::cout << #TYPEZ << " " << #Z << " = " << Z << "\n";		\
    std::cout << "Evaluating " << #EXPR << "\n";			\
    std::cout.flush();							\
    EXPR;								\
    std::cout << "Result: " << #X << " = " << X << "\n";		\
    if (should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	        \
      anomalous_results++;						\
    }									\
  } catch (adept::exception e) {					\
    std::cout << "*** Failed with: " << e.what() << "\n";		\
    if (!should_fail) { std::cout << "*** INCORRECT OUTCOME\n";		\
      anomalous_results++;						\
    }									\
    else {								\
      std::cout << "*** Correct behaviour\n";				\
    }									\
  }

#define EVAL_NO_TRAP(MESSAGE, TYPE, X, INIT, EXPR)				\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n";	\
  {									\
    TYPE X;								\
    if (INIT) {								\
      X = test. X;							\
      std::cout << #TYPE << " " << #X << " = " << X << "\n";		\
    }									\
    else {								\
      std::cout << #TYPE << " " << #X << " = " << X << "\n";		\
    }									\
    std::cout << "Evaluating " << #EXPR << "\n";			\
    std::cout.flush();						\
    EXPR;								\
    std::cout << "Result: " << #X << " = " << X << "\n";		\
    if (should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	\
      anomalous_results++;						\
    }									\
  }  

#define EVAL2_NO_TRAP(MESSAGE, TYPEX, X, INITX, TYPEY, Y, EXPR)			\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n###  " << #EXPR << "\n";	\
  {									\
    TYPEX X;								\
    if (INITX) {								\
      X = test. X;							\
      std::cout << #TYPEX << " " << #X << " = " << X << "\n";		\
    }									\
    else {								\
      std::cout << #TYPEX << " " << #X << " = " << X << "\n";		\
    }									\
    TYPEY Y; Y = test. Y;						\
    std::cout << #TYPEY << " " << #Y << " = " << Y << "\n";		\
    std::cout << "Evaluating " << #EXPR << "\n";			\
    std::cout.flush();							\
    EXPR;								\
    std::cout << "Result: " << #X << " = " << X << "\n";		\
    if (should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	\
      anomalous_results++;						\
    }									\
  }

#ifndef ALL_COMPLEX

#ifdef ALL_ACTIVE
#ifndef MARVEL_STYLE
  typedef aReal myReal;
  typedef aMatrix myMatrix;
  typedef aVector myVector;
  typedef aSymmMatrix mySymmMatrix;
  //typedef aSquareMatrix mySymmMatrix;
  typedef aDiagMatrix myDiagMatrix;
  typedef aTridiagMatrix myTridiagMatrix;
  typedef aLowerMatrix myLowerMatrix;
  typedef aUpperMatrix myUpperMatrix;
  typedef SpecialMatrix<Real,BandEngine<ROW_MAJOR,2,1>,true> myOddBandMatrix;
  typedef aArray3D myArray3D;
#else
  typedef aReal myReal;
  typedef Array<2,aReal,false> myMatrix;
  typedef Array<1,aReal,false> myVector;
  typedef SpecialMatrix<aReal,SquareEngine<ROW_MAJOR>,false> mySymmMatrix;
  typedef SpecialMatrix<aReal,BandEngine<ROW_MAJOR,0,0>,false> myDiagMatrix;
  typedef SpecialMatrix<aReal,BandEngine<ROW_MAJOR,1,1>,false> myTridiagMatrix;
  typedef SpecialMatrix<aReal,internal::LowerEngine<ROW_MAJOR>, false> myLowerMatrix;
  typedef SpecialMatrix<aReal,internal::UpperEngine<ROW_MAJOR>, false> myUpperMatrix;
  typedef SpecialMatrix<aReal,BandEngine<ROW_MAJOR,2,1>,false> myOddBandMatrix;

#endif
#else

  typedef Real   myReal;
  typedef Matrix myMatrix;
  typedef Vector myVector;
  typedef Array3D myArray3D;

  typedef SymmMatrix mySymmMatrix;
  //typedef SquareMatrix mySymmMatrix;
  typedef DiagMatrix myDiagMatrix;
  typedef TridiagMatrix myTridiagMatrix;
  typedef LowerMatrix myLowerMatrix;
  typedef UpperMatrix myUpperMatrix;
  typedef SpecialMatrix<Real,BandEngine<ROW_MAJOR,2,1>,false> myOddBandMatrix;

  /*    
  typedef SpecialMatrix<Real,SymmEngine<ROW_UPPER_COL_LOWER>,false> mySymmMatrix;
  typedef SpecialMatrix<Real,BandEngine<COL_MAJOR,0,0>,false> myDiagMatrix;
  typedef SpecialMatrix<Real,BandEngine<COL_MAJOR,1,1>,false> myTridiagMatrix;
  typedef SpecialMatrix<Real,BandEngine<COL_MAJOR,2,1>,false> myOddBandMatrix;
  */

#endif


#else
  typedef std::complex<Real> myReal;
  typedef Array<1,std::complex<Real>,IsActive> myVector;
  typedef Array<2,std::complex<Real>,IsActive> myMatrix;
  typedef Array<3,std::complex<Real>,IsActive> myArray3D;
  typedef SpecialMatrix<std::complex<Real>,SquareEngine<ROW_MAJOR>,IsActive> mySymmMatrix;
  typedef SpecialMatrix<std::complex<Real>,BandEngine<ROW_MAJOR,0,0>,IsActive> myDiagMatrix;
  typedef SpecialMatrix<std::complex<Real>,BandEngine<ROW_MAJOR,1,1>,IsActive> myTridiagMatrix;
  typedef SpecialMatrix<std::complex<Real>,internal::LowerEngine<ROW_MAJOR>, IsActive> myLowerMatrix;
  typedef SpecialMatrix<std::complex<Real>,internal::UpperEngine<ROW_MAJOR>, IsActive> myUpperMatrix;
  typedef SpecialMatrix<std::complex<Real>,BandEngine<ROW_MAJOR,2,1>,IsActive> myOddBandMatrix;

#endif

  struct Test {

    bool b;
    boolVector B;
    int c;
    myReal x;
    myVector v, w, vlong;
    myMatrix M, N;
    myMatrix Mstrided;
    myMatrix S;
    mySymmMatrix O, P;
    myDiagMatrix D, E;
    myTridiagMatrix T, TT;
    myLowerMatrix L, LL;
    myUpperMatrix U, UU;
    myOddBandMatrix Q, R;
    intVector index;
    myArray3D A;

#define MINI_TEST
#ifdef MINI_TEST
#define DIM1 3
#define DIM2 2
#define DIM3 5
#define DIMLONG 12
#else
#define DIM1 12
#define DIM2 10
#define DIM3 15
#define DIMLONG 20
#endif
    Test() {
#ifdef ALL_COMPLEX
#define I std::complex<Real>(0.0,1.0)
#else
#define I 0.0
#endif
      b = false;
      B.resize(DIM1); B = false;
      c = 0;
      x = -2;
      v.resize(DIM1);
      vlong.resize(DIMLONG); vlong = linspace(1,DIMLONG,DIMLONG);
      w.resize(DIM1);
      M.resize(DIM2,DIM1);
      myMatrix Mtmp(DIM2*3,DIM1*2);
      Mstrided.link(Mtmp(stride(0,end,3),stride(0,end,2)));
      N.resize(DIM2,DIM1);
      S.resize(DIM1,DIM1);
      O.resize(DIM1);
      Q.resize(DIM3);
      index.resize(DIM2);
      v(0) = 2.0 + 3.0*I; v(1) = 3; v(2) = 5;
      w(0) = 7.0 + 4.0*I; w(1) = 11; w(2) = 13;
      M(0,0) = 2.0 + 3.0*I; M(0,1) = 3; M(0,2) = 5;
      M(1,0) = 7; M(1,1) = 11; M(1,2) = 13;
      Mstrided = M;
      N(0,0) = 17.0+5.0*I; N(0,1) = 19; N(0,2) = 23;
      N(1,0) = 29; N(1,1) = 31; N(1,2) = 37;
      S(0,0) = 2.0+3.0*I; S(0,1) = 3; S(0,2) = 5;
      S(1,0) = 7.0+4.0*I; S(1,1) = 11; S(1,2) = 13;
      S(2,0) = 17; S(2,1) = 19; S(2,2) = 23;

      O(0,0) = 7.0+3.0*I;
      O(1,0) = 2; O(1,1) = 11;
      O(2,0) = 3; O(2,1) = 5; O(2,2) = 13;

      P = 14.0 - O;

      Q.diag_vector(-2) = 1;
      Q.diag_vector(-1) = 2;
      Q.diag_vector(0)  = 3;
      Q.diag_vector(1)  = 4;

      D = S;
      T = S;
      L = S;
      U = S;

      A.resize(DIM2,DIM1,DIM2);
      A << 2.0+3.0*I, 3, 5, 7, 11, 13,
	17, 19, 23, 29, 31,37;

      index << 1, 0;
    }
  };

#ifdef ALL_ACTIVE
#ifndef ADEPT_RECORDING_PAUSABLE
  stack.new_recording();
#else
  stack.pause_recording();
#endif
#endif

  Test test;

  bool should_fail=false;
  int anomalous_results=0;

  std::cout << adept::configuration();

#ifdef ALL_ACTIVE
  std::cout << "Testing ACTIVE arrays\n";
#else
  std::cout << "Testing INACTIVE arrays\n";
#endif
#ifdef ALL_COMPLEX
  std::cout << "Testing COMPLEX arrays\n";
#endif


  HEADING("ARRAY FUNCTIONALITY");
  EVAL("Array \"resize\" member function", myMatrix, M, true, M.resize(1,5));
  
  should_fail=true;
  EVAL("Array \"resize\" with invalid dimensions", myMatrix, M, true, M.resize(1));
  should_fail=false;
  EVAL("Array \"resize\" with \"dimensions\" function", myMatrix, M, true, M.resize(dimensions(4,2)));

  EVAL("Array \"clear\" member function", myMatrix, M, true, M.clear());

#ifdef ADEPT_CXX11_FEATURES
  HEADING("INITIALIZER LISTS (C++11 ONLY)");
  EVAL("Vector assignment to initializer list from empty", myVector, v,
       false, v = {1 COMMA 2});
  EVAL("Vector assignment to initializer list with underfill", myVector, v,
       true, v = {1.0 COMMA 2.0});
  should_fail = true;
  EVAL("Vector assignment to initializer list with overfill (SHOULD FAIL)", myVector, v,
    true, v = {1.0 COMMA 2.0 COMMA 3.0 COMMA 4.0});
  should_fail = false;
  EVAL("Matrix assignment to initializer list from empty", myMatrix, M,
    false, M = { {1 COMMA 2} COMMA {3 COMMA 4} });
  EVAL("Matrix assignment to initializer list with underfill", myMatrix, M,
    true, M = { {1.0 COMMA 2.0} COMMA {3.0 COMMA 4.0} });
  should_fail = true;
  EVAL("Matrix assignment to initializer list with overfill (SHOULD FAIL)", myMatrix, M,
    true, M = { {1.0 COMMA 2.0 COMMA 3.0 COMMA 4.0} });
  should_fail = false;
  EVAL("Initializer list in expression", myVector, v,
    true, v = v + Vector({1.0 COMMA 2.0 COMMA 3.0}));
  EVAL2("Indexed matrix assigned to initializer list", myMatrix, M, true, intVector, index, 
	M(index,index) = {{1 COMMA 2} COMMA {3 COMMA 4}});

#endif


  HEADING("BASIC EXPRESSIONS");
  EVAL2("Vector assignment to vector from empty", myVector, v, false, myVector, w, v = w);
  EVAL2("Vector assignment to expression from empty", myVector, v, false, myVector, w, v = log(w) + 1.0);

  /*
  should_fail=true;
  EVAL("Vector = operator from empty (SHOULD FAIL)", myVector, v, false, v = 1.0);
  EVAL("Vector += operator from empty (SHOULD FAIL)", myVector, v, false, v += 1.0);
  should_fail=false;
  */

  EVAL("Matrix *= operator", myMatrix, M, true, M *= 0.5);
  EVAL2("Matrix = scalar", myMatrix, M, true, myReal, x, M = x);
  EVAL2("Matrix = scalar expression", myMatrix, M, true, myReal, x, M = (10.0*x));
#ifndef ALL_COMPLEX
  HEADING("BASIC FUNCTIONS");
  EVAL2("max", myVector, v, true, myVector, w, v = max(v,w/3.0));
  EVAL2("min", myVector, v, true, myVector, w, v = min(v,w/3.0));
#endif

  HEADING("ARRAY SLICING");
  EVAL2("Array indexing rvalue", myReal, x, true, myMatrix, M, x = M(1,end-1));

  should_fail=true;
  EVAL2("Array indexing rvalue out of range (SHOULD FAIL)", myReal, x, true, myMatrix, M, x = M(1,3));
  should_fail=false;

  EVAL("Array indexing lvalue", myMatrix, M, true, M(1,end-1) *= -1.0);

  EVAL2("contiguous subarray rvalue", myVector, v, false, myMatrix, M, v = M(__,end));
  EVAL("contiguous subarray lvalue", myMatrix, M, true, M(end-1,__) /= 2.0);
  EVAL2("contiguous subarray rvalue and lvalue", myMatrix, M, true, myMatrix, N, M(__,1) = N(__,2));
  EVAL2("contiguous subarray rvalue using range", myVector, v, false, myMatrix, M, v = 2.0 * M(1,range(1,2)));
  EVAL2("contiguous subarray lvalue using range", myMatrix, M, true, myVector, v, M(end-1,range(0,1)) = log(v(range(1,2))));
  EVAL2("contiguous subarray rvalue using subset", myMatrix, M, false, myMatrix, N, M = 2.0 * N.subset(1,1,1,2));
  EVAL("contiguous subarray lvalue using subset", myVector, v, true, v.subset(end-1,end) *= 10.0);
  EVAL2("regular subarray rvalue", myVector, v, false, myVector, w, v = w(stride(end,0,-1)));
  EVAL2("regular subarray lvalue", myMatrix, M, true, myVector, w, M(0,stride(0,end,2)) *= w(stride(end,0,-2)));
#ifndef ALL_COMPLEX
  EVAL2("irregular subarray rvalue", myMatrix, M, false, myMatrix, N, M = N(stride(1,0,-1),find(N(0,__)>18)));
  EVAL("irregular subarray lvalue", myMatrix, M, true, M(stride(1,0,-1),find(M(0,__)>4)) = 0);
#endif
  EVAL("slice leading dimension", myMatrix, M, true, M[end] = 0);
  EVAL("slice two dimensions", myMatrix, M, true, M[end][0] = 0);
  EVAL2("diag_vector member function as rvalue", myVector, v, false, myMatrix, S, v = diag_vector(S,1));
  EVAL2("diag_vector member function as lvalue", myMatrix, S, true, myVector, v, S.diag_vector() += v);
  EVAL2("diag_matrix member function", myMatrix, S, false, myVector, v, S = v.diag_matrix());
  EVAL2("diag_matrix external function", myMatrix, S, false, myVector, v, S = diag_matrix(v));
  EVAL2("transpose as rvalue via T member function", myMatrix, N, false, myMatrix, M, N = 2.0 * M.T());
  EVAL2("transpose as rvalue via permute member function", myMatrix, N, false, myMatrix, M, N = 2.0 * M.permute(1,0));
  EVAL3("matrix indexing (scalar,non-contiguous)", myVector, v, false, myMatrix, N, intVector, index, v = N(1,index)); 
  EVAL3("matrix indexing (non-contiguous,scalar)", myVector, v, false, myMatrix, N, intVector, index, v = N(index,1)); 
  EVAL3("2D arbitrary index as rvalue", myMatrix, M, false, myMatrix, N, intVector, index, M = const_cast<const myMatrix&>(N)(index,index));
  EVAL3("2D arbitrary index as lvalue assigned to scalar expression", myMatrix, M, true, myMatrix, N, intVector, index, M(index,index) = 2.0*(myReal)(4.0));
  EVAL3("2D arbitrary index as lvalue", myMatrix, M, true, myMatrix, N, intVector, index, M(index,index) = N(__,range(1,2)));
  EVAL2("2D arbitrary index as lvalue with assign-multiply operator", myMatrix, M, true, intVector, index, M(index,index) *= 10.0);
  EVAL2("2D arbitrary index as lvalue with aliased right-hand-side", myMatrix, M, true, intVector, index, M(index,index) = M(__,range(0,1)));
  EVAL2("2D arbitrary index as lvalue with aliased right-hand-side and eval function", myMatrix, M, true, intVector, index, M(index,index) = eval(M(__,range(0,1))));
  EVAL2("reshape member function", myMatrix, M, false, myVector, vlong, M >>= vlong.reshape(3,4));
  should_fail=true;
  EVAL2("reshape member function with invalid dimensions", myMatrix, M, false, myVector, vlong, M >>= vlong.reshape(5,5));
  should_fail=false;
  EVAL("end/2 indexing", myVector, vlong, true, vlong(range(end/2,end)) = 0.0);
  EVAL("end/2 indexing", myVector, vlong, true, vlong(range(0,end/2)) = 0.0);
  EVAL("end/2 indexing", myVector, vlong, true, vlong.subset(end/2,end) = 0.0);

  HEADING("REDUCTION OPERATIONS"); 
  EVAL2("full sum", myReal, x, true, myMatrix, M, x = sum(M));
  EVAL2("full product", myReal, x, true, myMatrix, M, x = product(M));
#ifndef ALL_COMPLEX
  EVAL2("full maxval", myReal, x, true, myMatrix, M, x = maxval(M));
  EVAL2("full minval", myReal, x, true, myMatrix, M, x = minval(M));
#endif
  EVAL2("full norm2", myReal, x, true, myMatrix, M, x = norm2(M));
  EVAL2("1-dimension mean", myVector, v, false, myMatrix, M, v = 0.5 * mean(M,0));
  EVAL2("1-dimension norm2", myVector, v, false, myMatrix, M, v = norm2(M,1));
  EVAL2("dot product", myReal, x, true, myVector, w, x = dot_product(w,w(stride(end,0,-1))));
  EVAL2("dot product on expressions", myReal, x, true, myVector, w, x = dot_product(2.0*w,w(stride(end,0,-1))+1.0));
#ifndef ALL_COMPLEX
  EVAL2("1-dimension maxval", myVector, v, false, myMatrix, M, v = maxval(M,1));
  EVAL2("1-dimension minval", myVector, v, false, myMatrix, M, v = minval(M,1));
  EVAL2("1D interpolation", myVector, v, true, myVector, w, v = interp(value(v), w, Vector(value(w)/3.0)));
  EVAL2("1D interpolation", myVector, v, true, myVector, w, v = interp(value(v), w, value(w)/3.0));
  EVAL2("all reduction", bool, b, true, myMatrix, M, b = all(M > 8.0));
  EVAL2("any reduction", bool, b, true, myMatrix, M, b = any(M > 8.0));
  EVAL2("count reduction", int, c, true, myMatrix, M, c = count(M > 8.0));
  EVAL2("1-dimension all reduction", boolVector, B, false, myMatrix, M, B = all(M > 8.0, 1));
  EVAL2("1-dimension any reduction", boolVector, B, false, myMatrix, M, B = any(M > 8.0, 1));
  EVAL2("1-dimension count reduction", intVector, index, false, myMatrix, M, index = count(M > 8.0, 1));
  HEADING("CONDITIONAL OPERATIONS");
  EVAL2("where construct, scalar right-hand-side", myMatrix, M, true, myMatrix, N, M.where(N > 20) = 0);
  EVAL2("where construct, expression right-hand-side", myMatrix, M, true, myMatrix, N, M.where(N > 20) = -N);
  EVAL2("where construct, scalar either-or right-hand-side", myMatrix, M, true, myMatrix, N, M.where(N > 20) = either_or(0,1));
  EVAL2("where construct, expression either-or right-hand-side", myMatrix, M, true, myMatrix, N, M.where(N > 20) = either_or(-N,N));
  EVAL_NO_TRAP("find construct, scalar right-hand-side", myVector, v, true, v(find(v > 3.5)) = 0);
  EVAL("find construct, expression right-hand-side", myVector, v, true, v(find(v > 3.5)) = -v(range(end,end)));
  EVAL("find construct, multiply-assign right-hand-side", myVector, v, true, v(find(v != 5.0)) *= 10.0);
#endif
  HEADING("SPECIAL SQUARE MATRICES");
  EVAL("SymmMatrix \"resize\" member function", mySymmMatrix, O, true, O.resize(5));

  should_fail = true;
  EVAL("SymmMatrix \"resize\" with invalid dimensions", mySymmMatrix, O, true, O.resize(4,5));
  should_fail = false;

  EVAL("SymmMatrix \"clear\" member function", mySymmMatrix, O, true, O.clear());
  EVAL2("SymmMatrix assign from dense matrix", mySymmMatrix, O, false, myMatrix, S, O = S);
  EVAL2("DiagMatrix assign from dense matrix", myDiagMatrix, D, false, myMatrix, S, D = S);
  EVAL2("TridiagMatrix assign from dense matrix", myTridiagMatrix, T, false, myMatrix, S, T = S);
  EVAL2("LowerMatrix assign from dense matrix", myLowerMatrix, L, false, myMatrix, S, L = S);
  EVAL2("UpperMatrix assign from dense matrix", myUpperMatrix, U, false, myMatrix, S, U = S);
  EVAL("SymmMatrix += operator", mySymmMatrix, O, true, O += 3.0);
  EVAL("DiagMatrix += operator", myDiagMatrix, D, true, D += 3.0);
  EVAL("TridiagMatrix += operator", myTridiagMatrix, T, true, T += 3.0);
  EVAL("LowerMatrix += operator", myLowerMatrix, L, true, L += 3.0);
  EVAL("UpperMatrix += operator", myUpperMatrix, U, true, U += 3.0);
  EVAL2("SymmMatrix as rvalue", myMatrix, M, false, mySymmMatrix, O, M = O);
  EVAL2("DiagMatrix as rvalue", myMatrix, M, false, myDiagMatrix, D, M = D);
  EVAL2("TridiagMatrix as rvalue", myMatrix, M, false, myTridiagMatrix, T, M = T);
  EVAL2("LowerMatrix as rvalue", myMatrix, M, false, myLowerMatrix, L, M = L);
  EVAL2("UpperMatrix as rvalue", myMatrix, M, false, myUpperMatrix, U, M = U);
  EVAL("SymmMatrix assign from scalar expression", mySymmMatrix, O, true, O = 2.0*(myReal)(4.0));
  EVAL("UpperMatrix assign from scalar expression", myUpperMatrix, U, true, U = 2.0*(myReal)(4.0));


  EVAL("SymmMatrix diag_vector member function as lvalue (upper)", mySymmMatrix, O, true, O.diag_vector(1) = 0);
  EVAL("SymmMatrix diag_vector member function as lvalue (lower)", mySymmMatrix, O, true, O.diag_vector(-2) += 10.0);
  EVAL("DiagMatrix diag_vector member function as lvalue", myDiagMatrix, D, true, D.diag_vector() = 0.0);

  should_fail = true;
  EVAL("DiagMatrix diag_vector member function incorrectly using offdiagonal", myDiagMatrix, D, true, D.diag_vector(1) = 0.0);
  should_fail = false;

  EVAL("TridiagMatrix diag_vector member function as lvalue (upper)", myTridiagMatrix, T, true, T.diag_vector(1) += 10.0);
  EVAL("TridiagMatrix diag_vector member function as lvalue (lower)", myTridiagMatrix, T, true, T.diag_vector(-1) = 0.0);
  EVAL("LowerMatrix diag_vector member function as lvalue (lower)", myLowerMatrix, L, true, L.diag_vector(-1) = 0.0);

  should_fail = true;
  EVAL("LowerMatrix diag_vector member function as lvalue (upper)", myLowerMatrix, L, true, L.diag_vector(1) = 0.0);
  EVAL("UpperMatrix diag_vector member function as lvalue (lower)", myUpperMatrix, U, true, U.diag_vector(-1) = 0.0);
  should_fail = false;

  EVAL("UpperMatrix diag_vector member function as lvalue (upper)", myUpperMatrix, U, true, U.diag_vector(1) = 0.0);
  EVAL("Odd band matrix \"diag_vector\" member function", myOddBandMatrix, Q, true, Q.diag_vector(1) = -1.0);
  EVAL("Odd band matrix \"diag_vector\" member function", myOddBandMatrix, Q, true, Q.diag_vector(0) = -1.0);
  EVAL("Odd band matrix \"diag_vector\" member function", myOddBandMatrix, Q, true, Q.diag_vector(-1) = -1.0);
  EVAL("Odd band matrix \"diag_vector\" member function", myOddBandMatrix, Q, true, Q.diag_vector(-2) = -1.0);

  EVAL2("Array submatrix_on_diagonal member function", myMatrix, M, false, myMatrix, S, M = S.submatrix_on_diagonal(1,2));
  EVAL("Array submatrix_on_diagonal member function as lvalue", myMatrix, S, true, S.submatrix_on_diagonal(0,1) = 0.0);

  should_fail = true;
  EVAL2("Array submatrix_on_diagonal member function to non-square matrix", myMatrix, M, false, myMatrix, N, M = N.submatrix_on_diagonal(1,2));
  should_fail = false;

  EVAL2("SymmMatrix submatrix_on_diagonal member function", mySymmMatrix, P, false, mySymmMatrix, O, P = O.submatrix_on_diagonal(1,2));
  EVAL2("DiagMatrix submatrix_on_diagonal member function", myDiagMatrix, E, false, myDiagMatrix, D, E = D.submatrix_on_diagonal(1,2));
  EVAL2("TridiagMatrix submatrix_on_diagonal member function", myTridiagMatrix, TT, false, myTridiagMatrix, T, TT = T.submatrix_on_diagonal(1,2));
  EVAL2("LowerMatrix submatrix_on_diagonal member function", myLowerMatrix, LL, false, myLowerMatrix, L, LL = L.submatrix_on_diagonal(1,2));
  EVAL2("UpperMatrix submatrix_on_diagonal member function", myUpperMatrix, UU, false, myUpperMatrix, U, UU = U.submatrix_on_diagonal(1,2));
  EVAL2("Odd band matrix submatrix_on_diagonal member function", myOddBandMatrix, R, false, myOddBandMatrix, Q, R = Q.submatrix_on_diagonal(1,3));
  EVAL("Odd band matrix submatrix_on_diagonal as lvalue", myOddBandMatrix, Q, true, Q.submatrix_on_diagonal(1,3) = -1);
  EVAL2("SymmMatrix transpose as rvalue via T member function", mySymmMatrix, P, false, mySymmMatrix, O, P = O.T());
  EVAL2("DiagMatrix transpose as rvalue via T member function", myDiagMatrix, E, false, myDiagMatrix, D, E = D.T());
  EVAL2("TridiagMatrix transpose as rvalue via T member function", myTridiagMatrix, TT, false, myTridiagMatrix, T, TT = T.T());
  EVAL2("LowerMatrix transpose as rvalue via T member function", myUpperMatrix, U, false, myLowerMatrix, L, U = L.T());
  EVAL2("UpperMatrix transpose as rvalue via T member function", myLowerMatrix, L, false, myUpperMatrix, U, L = U.T());

  HEADING("EXPANSION OPERATIONS");
  EVAL2("Outer product", myMatrix, M, false, myVector, v, M = outer_product(v,v));
  EVAL2("Outer product on indexed array", myMatrix, M, false, myVector, v, M = outer_product(v,v(stride(end,0,-1))));
  EVAL2("Outer product on expressions", myMatrix, M, false, myVector, v, M = outer_product(2.0*v,v-1.0));
  EVAL2("Vector spread of dimension 0", myMatrix, M, false, myVector, v, M = spread<0>(v,2));
  EVAL2("Vector spread of dimension 1", myMatrix, M, false, myVector, v, M = spread<1>(v,2));
  EVAL2("Vector spread with expression argument", myMatrix, M, false, myVector, v, M = spread<1>(v*2.0,2));
  EVAL2("Matrix spread of dimension 0", myArray3D, A, false, myMatrix, M, A = spread<0>(M,2));
  EVAL2("Matrix spread of dimension 1", myArray3D, A, false, myMatrix, M, A = spread<1>(M,2));
  EVAL2("Matrix spread of dimension 2", myArray3D, A, false, myMatrix, M, A = spread<2>(M,2));

#ifndef ALL_COMPLEX

#ifndef MARVEL_STYLE
  if (adept::have_matrix_multiplication()) {
    HEADING("MATRIX MULTIPLICATION");
    EVAL3("Matrix-Vector multiplication", myVector, w, false, myMatrix, M, myVector, v, w = M ** v);
    EVAL3("Matrix-Vector multiplication with strided matrix", myVector, w, false, myMatrix, Mstrided, myVector, v, w = Mstrided ** v);
    EVAL2("Matrix-Matrix multiplication", myMatrix, M, false, myMatrix, N, M = N.T() ** N);
    EVAL2("Matrix-Matrix multiplication with matmul", myMatrix, M, false, myMatrix, N, M = matmul(N.T(), N));
    
    should_fail = true;
    EVAL2("Matrix-Matrix multiplication with inner dimension mismatch", myMatrix, M, false, myMatrix, N, M = N ** N);
    should_fail = false;
    
    // TESTING!
    EVAL2("Matrix-Matrix-Vector multiplication", myVector, v, true, myMatrix, S, v = S ** S ** v);
    
    EVAL2("Matrix-Matrix-Vector multiplication", myVector, v, false, myMatrix, S, v = S ** log(S) ** S(0,__));
    EVAL2("Vector-Matrix multiplication", myVector, v, true, myMatrix, S, v = v ** S);
    EVAL2("Vector-Matrix multiplication with matmul", myVector, v, true, myMatrix, S, v = matmul(v, S));
    EVAL2("SymmMatrix-Vector multiplication", myVector, v, true, mySymmMatrix, O, v = O ** v);
    EVAL2("SymmMatrix-Matrix multiplication", myMatrix, S, true, mySymmMatrix, O, S = O ** S);
    EVAL2("Vector-SymmMatrix multiplication", myVector, v, true, mySymmMatrix, O, v = v ** O);
    EVAL2("Matrix-SymmMatrix multiplication", myMatrix, M, true, mySymmMatrix, O, M = M ** O);
    EVAL2("DiagMatrix-Vector multiplication", myVector, v, true, myDiagMatrix, D, v = D ** v);
    EVAL2("TridiagMatrix-Vector multiplication", myVector, v, true, myTridiagMatrix, T, v = T ** v);
    EVAL2("TridiagMatrix-Matrix multiplication", myMatrix, S, true, myTridiagMatrix, T, S = T ** S);
    
    EVAL2("LowerMatrix-Matrix multiplication", myMatrix, S, true, myLowerMatrix, L, S = L ** S);
    
    EVAL2("Vector-TridiagMatrix multiplication", myVector, v, true, myTridiagMatrix, T, v = v ** T);
    EVAL2("Matrix-TridiagMatrix multiplication", myMatrix, M, true, myTridiagMatrix, T, M = M ** T);
  }
  else {
    std::cout << "NO MATRIX MULTIPLICATION TESTS PERFORMED BECAUSE ADEPT COMPILED WITHOUT LAPACK\n";
  }
    
#ifndef ALL_ACTIVE
  if (adept::have_linear_algebra()) {
    HEADING("LINEAR ALGEBRA");
    EVAL2("Solving general linear equations Ax=b", myVector, v, true, myMatrix, S, v = solve(S,v));
    EVAL2("Solving general linear equations Ax=b with expression arguments", myVector, v, true, myMatrix, S, v = solve(S,2*v));
    
    EVAL2("Solving general linear equations AX=B", myMatrix, M, true, myMatrix, S, M.T() = solve(S,M.T()));
    EVAL2("Solving general linear equations AX=B with expression arguments", myMatrix, M, true, myMatrix, S, M.T() = solve(2.0 * S,2.0 * M.T()));
    EVAL2("Solving linear equations Ax=b with symmetric A", myVector, v, true, mySymmMatrix, O, v = solve(O,v));
    EVAL2("Solving linear equations AX=B with symmetric A", myMatrix, M, true, mySymmMatrix, O, M.T() = solve(O,M.T()));
    EVAL3("Solving linear equations AX=B with symmetric A and B", myMatrix, S, false, mySymmMatrix, O, mySymmMatrix, P, S = solve(O,P));
    EVAL2("Solving linear equations Ax=b with upper-triangular A", myVector, v, true, myUpperMatrix, U, v = solve(U,v));
    EVAL2("Invert general matrix", myMatrix, M, false, myMatrix, S, M = inv(S));
    EVAL2("Invert symmetric matrix", mySymmMatrix, P, false, mySymmMatrix, O, P = inv(O));
  }
  else {
    std::cout << "NO LINEAR ALGEBRA TESTS PERFORMED BECAUSE ADEPT COMPILED WITHOUT LAPACK\n";
  }
#else
  std::cout << "NO LINEAR ALGEBRA TESTS PERFORMED BECAUSE ACTIVE ARRAYS NOT YET SUPPORTED\n";
#endif
#else
  std::cout << "NO MATRIX TESTS PERFORMED BECAUSE USING MARVEL-STYLE ACTIVE ARRAYS\n";
#endif

#endif

  HEADING("FILLING ARRAYS");
  EVAL("Fill vector with \"<<\"", myVector, v, true, (v << 0.1, 0.2));

  should_fail = true;
  EVAL("Overfill vector with \"<<\"", myVector, v, true, (v << 0.1, 0.2, 0.3, 0.4));
  should_fail = false;

  EVAL("Underfill matrix with \"<<\"", myMatrix, M, true, (M << 0.1, 0.2, 0.3, 0.4, 0.5));
  EVAL("Fill matrix with \"<<\"", myMatrix, M, true, (M << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6));

  should_fail = true;
  EVAL("Overfill matrix with \"<<\"", myMatrix, M, true, (M << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0));
  should_fail = false;

  EVAL("Fill vector with vectors using \"<<\"", myVector, v, true, v << v(range(1,2)) << 0.1);
  EVAL2("Fill matrix with vector using \"<<\"", myMatrix, M, true, myVector, v, M << 0.1 << 0.2 << 0.3 << v);
  EVAL2("Fill matrix with vector using \"<<\"", myMatrix, S, true, myVector, v, S << v << v << v);
  EVAL("Assign array using range", myVector, v, false, v = range(3,6));

  HEADING("PRINTING WITH PLAIN STYLE");
  adept::set_array_print_style(PRINT_STYLE_PLAIN);
  SIMPLE_EVAL("Printing empty vector", myVector, v, false, std::cout << v << '\n');
  SIMPLE_EVAL("Printing vector", myVector, v, true, std::cout << v << '\n');
  SIMPLE_EVAL("Printing matrix", myMatrix, M, true, std::cout << M << '\n');
  SIMPLE_EVAL("Printing 3D array", myArray3D, A, true, std::cout << A << '\n');

  HEADING("PRINTING WITH CSV STYLE");
  adept::set_array_print_style(PRINT_STYLE_CSV);
  SIMPLE_EVAL("Printing empty vector", myVector, v, false, std::cout << v << '\n');
  SIMPLE_EVAL("Printing vector", myVector, v, true, std::cout << v << '\n');
  SIMPLE_EVAL("Printing matrix", myMatrix, M, true, std::cout << M << '\n');
  SIMPLE_EVAL("Printing 3D array", myArray3D, A, true, std::cout << A << '\n');

  HEADING("PRINTING WITH CURLY STYLE");
  adept::set_array_print_style(PRINT_STYLE_CURLY);
  SIMPLE_EVAL("Printing empty vector", myVector, v, false, std::cout << v << '\n');
  SIMPLE_EVAL("Printing vector", myVector, v, true, std::cout << v << '\n');
  SIMPLE_EVAL("Printing matrix", myMatrix, M, true, std::cout << M << '\n');
  SIMPLE_EVAL("Printing 3D array", myArray3D, A, true, std::cout << A << '\n');

  HEADING("PRINTING WITH MATLAB STYLE");
  adept::set_array_print_style(PRINT_STYLE_MATLAB);
  SIMPLE_EVAL("Printing empty vector", myVector, v, false, std::cout << v << '\n');
  SIMPLE_EVAL("Printing vector", myVector, v, true, std::cout << v << '\n');
  SIMPLE_EVAL("Printing matrix", myMatrix, M, true, std::cout << M << '\n');
  SIMPLE_EVAL("Printing 3D array", myArray3D, A, true, std::cout << A << '\n');
  adept::set_array_print_style(PRINT_STYLE_CURLY);

  HEADING("EXPRESSION PRINTING");
  EVAL("Send expression to standard output", myMatrix, M, true,
       std::cout << M(0,__) + M(1,__) << '\n');
  EVAL("Send scalar expression to standard output", myVector, v, true,
       std::cout << v(0) + v(1) << '\n');

#ifdef ADEPT_BOUNDS_CHECKING
  HEADING("BOUNDS CHECKING");
  should_fail = true;
  EVAL("Access vector out of bounds", myVector, v, true, v(0) = v(4));
  EVAL("Access vector out of bounds", myVector, v, true, v(0) = v(end-4));
  EVAL("Access matrix out of bounds", myMatrix, M, true, M(0,0) = M(0,-1));
  EVAL("Access matrix out of bounds", myMatrix, M, true, M(0,0) = M(end+1,1));
  should_fail = false;
#endif

  std::cout << "====================================================================\n";
#ifdef ALL_ACTIVE
  std::cout << stack;
  std::cout << "====================================================================\n";
#endif

  if (anomalous_results > 0) {
    std::cout << "*** In terms of run-time errors, there were " << anomalous_results << " incorrect results\n";
  }
  else {
    std::cout << "In terms of run-time errors, all tests were passed\n";
  }

#ifdef ALL_ACTIVE
#ifdef ADEPT_RECORDING_PAUSABLE
  if (stack.n_statements() > 1) {
    std::cout << "*** Stack contains " << stack.n_statements()-1
	      << " statements and " << stack.n_operations()
	      << " operations but both should be 0 because recording has been paused\n";
    return 1;
  }
#endif
#endif
  if (anomalous_results > 0) {
    return 1;
  }
  else {
    return 0;
  }
}
