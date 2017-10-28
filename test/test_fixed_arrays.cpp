/* test_arrays.cpp - Test Adept's array functionality

    Copyright (C) 2016-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include <iostream>

#define ADEPT_BOUNDS_CHECKING 1

#include <adept_arrays.h>
#include <adept/FixedArray.h>

// The following controls whether to use active variables or not
//#define ALL_ACTIVE 1
//#define MARVEL_STYLE 1

using namespace adept;

int
main(int argc, const char** argv) {
  using namespace adept;
  Stack stack;
  
#define HEADING(MESSAGE) \
  std::cout << "====================================================================\n"	\
	    << "   TESTING " << MESSAGE << "\n"

#define EVAL(MESSAGE, TYPE, X, EXPR)					\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n";	\
  try {								\
    TYPE X;								\
    X = test. X;							\
    std::cout << #TYPE << " " << #X << " = " << X << "\n";		\
    std::cout << "Evaluating " << #EXPR << "\n";			\
    std::cout.flush();						\
    EXPR;								\
    std::cout << "Result: " << #X << " = " << X << "\n";		\
    if (should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	\
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

#define EVAL2(MESSAGE, TYPEX, X, TYPEY, Y, EXPR)			\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n";	\
  try {									\
    TYPEX X;								\
    X = test. X;							\
    std::cout << #TYPEX << " " << #X << " = " << X << "\n";		\
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


#define EVAL3(MESSAGE, TYPEX, X, TYPEY, Y, TYPEZ, Z, EXPR)		\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n";	\
  try {									\
    TYPEX X;								\
    X = test. X;							\
    std::cout << #TYPEX << " " << #X << " = " << X << "\n";		\
    TYPEY Y; Y = test. Y;						\
    TYPEZ Z; Z = test. Z;						\
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

#define EVAL_NO_TRAP(MESSAGE, TYPE, X, EXPR)				\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n";	\
  {									\
    TYPE X;								\
    X = test. X;							\
    std::cout << #TYPE << " " << #X << " = " << X << "\n";		\
    std::cout << "Evaluating " << #EXPR << "\n";			\
    std::cout.flush();						\
    EXPR;								\
    std::cout << "Result: " << #X << " = " << X << "\n";		\
    if (should_fail) { std::cout << "*** INCORRECT OUTCOME\n";	\
      anomalous_results++;						\
    }									\
  }  

#define EVAL2_NO_TRAP(MESSAGE, TYPEX, X, TYPEY, Y, EXPR)			\
  std::cout << "--------------------------------------------------------------------\n" \
	    << "### " << MESSAGE << "\n### " << #EXPR << "\n";	\
  {									\
    TYPEX X;								\
    X = test. X;							\
    std::cout << #TYPEX << " " << #X << " = " << X << "\n";		\
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


#ifdef ALL_ACTIVE
#define IS_ACTIVE true
#else
#define IS_ACTIVE false
#endif

  typedef FixedArray<double,IS_ACTIVE,2> myVector2;
  typedef FixedArray<double,IS_ACTIVE,3> myVector3;
  typedef FixedArray<double,IS_ACTIVE,1,2> myMatrix12;
  typedef FixedArray<double,IS_ACTIVE,3,3> myMatrix33;
  typedef FixedArray<double,IS_ACTIVE,2,3> myMatrix23;
  typedef FixedArray<double,IS_ACTIVE,3,2> myMatrix32;
  typedef FixedArray<double,IS_ACTIVE,2,2> myMatrix22;

#ifndef ALL_ACTIVE
  typedef Real myReal;
  typedef SymmMatrix mySymmMatrix;
  typedef DiagMatrix myDiagMatrix;
  typedef TridiagMatrix myTridiagMatrix;
  typedef LowerMatrix myLowerMatrix;
  typedef UpperMatrix myUpperMatrix;
#else
  typedef aReal myReal;
  typedef aSymmMatrix mySymmMatrix;
  typedef aDiagMatrix myDiagMatrix;
  typedef aTridiagMatrix myTridiagMatrix;
  typedef aLowerMatrix myLowerMatrix;
  typedef aUpperMatrix myUpperMatrix;
#endif

  //  typedef SpecialMatrix<Real,SymmEngine<ROW_UPPER_COL_LOWER>,false> mySymmMatrix;
  //  typedef SpecialMatrix<Real,BandEngine<COL_MAJOR,0,0>,false> myDiagMatrix;
  //  typedef SpecialMatrix<Real,BandEngine<COL_MAJOR,1,1>,false> myTridiagMatrix;



  struct Test {

    myReal x;
    myVector2 z;
    myVector3 v, w;
    myMatrix12 K;
    myMatrix23 M, N;
    myMatrix33 S, C;
    myMatrix32 A;
    myMatrix22 B;

    mySymmMatrix O, P;
    myDiagMatrix D, E;
    myTridiagMatrix T, TT;
    myLowerMatrix L, LL;
    myUpperMatrix U, UU;

    intVector index;


    Test() {
      x = -2;

      O.resize(3);
      //      Q.resize(5);
      index.resize(2);
      v(0) = 2; v(1) = 3; v(2) = 5;
      w(0) = 7; w(1) = 11; w(2) = 13;
      M(0,0) = 2; M(0,1) = 3; M(0,2) = 5;
      M(1,0) = 7; M(1,1) = 11; M(1,2) = 13;
      N(0,0) = 17; N(0,1) = 19; N(0,2) = 23;
      N(1,0) = 29; N(1,1) = 31; N(1,2) = 37;
      S(0,0) = 2; S(0,1) = 3; S(0,2) = 5;
      S(1,0) = 7; S(1,1) = 11; S(1,2) = 13;
      S(2,0) = 17; S(2,1) = 19; S(2,2) = 23;

      K << 57, 59;
      z << 37, 47;

      A << 21,22,23,24,25,26;
      B << 31,32,33,34;

      //      O = -M.T();


      O(0,0) = 7;
      O(1,0) = 2; O(1,1) = 11;
      O(2,0) = 3; O(2,1) = 5; O(2,2) = 13;
      /*

      P = 14-O;

      Q.diag_vector(-2) = 1;
      Q.diag_vector(-1) = 2;
      Q.diag_vector(0)  = 3;
      Q.diag_vector(1)  = 4;
      */

      C = 0;
      D = S;
      T = S;
      L = S;
      U = S;
      index << 1, 0;
    }
  };

  stack.new_recording();

  Test test;

  bool should_fail=false;
  int anomalous_results=0;

#ifdef ALL_ACTIVE
  std::cout << "Testing ACTIVE arrays\n";
#else
  std::cout << "Testing INACTIVE arrays\n";
#endif


  HEADING("BASIC EXPRESSIONS");
  EVAL2("Vector assignment to vector", myVector3, v, myVector3, w, v = w);
  EVAL2("Vector assignment to expression", myVector3, v, myVector3, w, v = log(w) + 1.0);

  EVAL("Matrix *= operator", myMatrix23, M, M *= 0.5);
  EVAL2("Matrix = scalar", myMatrix23, M, myReal, x, M = x);

  EVAL2("Matrix = scalar expression", myMatrix23, M, myReal, x, M = (10.0*x));
  HEADING("BASIC FUNCTIONS");
  EVAL2("max", myVector3, v, myVector3, w, v = max(v,w/3.0));
  EVAL2("min", myVector3, v, myVector3, w, v = min(v,w/3.0));

  HEADING("ARRAY SLICING");
  EVAL2("Array indexing rvalue", myReal, x, myMatrix23, M, x = M(1,end-1));

  should_fail=true;
  EVAL2("Array indexing rvalue out of range (SHOULD FAIL)", myReal, x, myMatrix23, M, x = M(1,3));
  should_fail=false;

  EVAL("Array indexing lvalue", myMatrix23, M, M(1,end-1) *= -1.0);

  EVAL2("contiguous subarray rvalue", myVector3, v, myMatrix23, M, v = M(end,__));
  EVAL("contiguous subarray lvalue", myMatrix23, M, M(end-1,__) /= 2.0);
  EVAL2("contiguous subarray rvalue using range", myVector2, z, myMatrix23, M, z = 2.0 * M(1,range(1,2)));
  EVAL2("contiguous subarray lvalue using range", myMatrix23, M, myVector3, v, M(end-1,range(0,1)) = log(v(range(1,2))));
  EVAL2("contiguous subarray rvalue using subset", myMatrix12, K, myMatrix23, N, K = 2.0 * N.subset(1,1,1,2));
  EVAL("contiguous subarray lvalue using subset", myVector3, v, v.subset(end-1,end) *= 10.0);

  EVAL2("regular subarray rvalue", myVector3, v, myVector3, w, v = w(stride(end,0,-1)));
  EVAL2("regular subarray lvalue", myMatrix23, M, myVector3, w, M(0,stride(0,end,2)) *= w(stride(end,0,-2)));
  EVAL("irregular subarray rvalue", myMatrix23, M, M(stride(1,0,-1),find(M(0,__)>4)) = 0);
  EVAL("slice leading dimension", myMatrix23, M, M[end] = 0);
  EVAL("slice two dimensions", myMatrix23, M, M[end][0] = 0);
  EVAL2("diag_vector member function as rvalue", myVector2, z, myMatrix33, S, z = diag_vector(S,1));
  EVAL2("diag_vector member function as lvalue", myMatrix33, S, myVector3, v, S.diag_vector() += v);
  EVAL2("diag_matrix member function", myMatrix33, S, myVector3, v, S = v.diag_matrix());
  EVAL2("diag_matrix external function", myMatrix33, S, myVector3, v, S = diag_matrix(v));
  EVAL2("transpose as rvalue via T member function", myMatrix32, A, myMatrix23, M, A = 2 * M.T());
  EVAL2("transpose as rvalue via permute member function", myMatrix32, A, myMatrix23, M, A = 2 * M.permute(1,0));
  //  EVAL3("2D arbitrary index as rvalue", myMatrix22, B, myMatrix23, N, intVector, index, B = const_cast<const myMatrix23&>(N)(index,index));
  EVAL3("2D arbitrary index as rvalue", myMatrix22, B, myMatrix23, N, intVector, index, B = N(index,index));
  EVAL3("2D arbitrary index as lvalue", myMatrix23, M, myMatrix23, N, intVector, index, M(index,index) = N(__,range(1,2)));
  EVAL2("2D arbitrary index as lvalue with assign-multiply operator", myMatrix23, M, intVector, index, M(index,index) *= 10.0);
  EVAL2("2D arbitrary index as lvalue with aliased right-hand-side", myMatrix23, M, intVector, index, M(index,index) += M(__,range(1,2)));

  HEADING("REDUCTION OPERATIONS"); 
  EVAL2("full reduction", myReal, x, myMatrix23, M, x = sum(M));
  EVAL2("1-dimension reduction", myVector3, v, myMatrix23, M, v = 0.5 * mean(M,0));
  EVAL2("1-dimension reduction", myVector2, z, myMatrix23, M, z = norm2(M,1));
  EVAL2("maxval", myVector2, z, myMatrix23, M, z = maxval(M,1));
  EVAL2("minval", myVector2, z, myMatrix23, M, z = minval(M,1));
  EVAL2("dot product", myReal, x, myVector3, w, x = dot_product(w,w(stride(end,0,-1))));
  //  EVAL2("1D interpolation", myVector3, v, myVector3, w, (v = interp<double,double,true,double>(value(v), w, value(w)/3.0) ));
  EVAL2("1D interpolation", myVector3, v, myVector3, w, v = interp(value(v), w, value(w)/3.0));
  HEADING("CONDITIONAL OPERATIONS");
  EVAL2("where construct, scalar right-hand-side", myMatrix23, M, myMatrix23, N, M.where(N > 20) = 0);
  EVAL2("where construct, expression right-hand-side", myMatrix23, M, myMatrix23, N, M.where(N > 20) = -N);
  EVAL2("where construct, scalar either-or right-hand-side", myMatrix23, M, myMatrix23, N, M.where(N > 20) = either_or(0,1));
  EVAL2("where construct, expression either-or right-hand-side", myMatrix23, M, myMatrix23, N, M.where(N > 20) = either_or(-N,N));
  EVAL("find construct, scalar right-hand-side", myVector3, v, v(find(v > 3.5)) = 0);
  EVAL("find construct, expression right-hand-side", myVector3, v, v(find(v > 3.5)) = -v(range(end,end)));
  EVAL("find construct, multiply-assign right-hand-side", myVector3, v, v(find(v != 5.0)) *= 10.0);

  HEADING("SPECIAL SQUARE MATRICES");
  EVAL2("SymmMatrix assign from fixed matrix", mySymmMatrix, O, myMatrix33, S, O = S);
  EVAL2("DiagMatrix assign from dense matrix", myDiagMatrix, D, myMatrix33, S, D = S);
  EVAL2("TridiagMatrix assign from dense matrix", myTridiagMatrix, T, myMatrix33, S, T = S);
  EVAL2("LowerMatrix assign from dense matrix", myLowerMatrix, L, myMatrix33, S, L = S);
  EVAL2("UpperMatrix assign from dense matrix", myUpperMatrix, U, myMatrix33, S, U = S);
  EVAL2("SymmMatrix as rvalue", myMatrix33, S, mySymmMatrix, O, S = O);
  EVAL2("DiagMatrix as rvalue", myMatrix33, S, myDiagMatrix, D, S = D);
  EVAL2("TridiagMatrix as rvalue", myMatrix33, S, myTridiagMatrix, T, S = T);
  EVAL2("LowerMatrix as rvalue", myMatrix33, S, myLowerMatrix, L, S = L);
  EVAL2("UpperMatrix as rvalue", myMatrix33, S, myUpperMatrix, U, S = U);

  EVAL2("Array submatrix_on_diagonal member function", myMatrix22, B, myMatrix33, S, B = S.submatrix_on_diagonal(1,2));
  EVAL("Array submatrix_on_diagonal member function as lvalue", myMatrix33, S, S.submatrix_on_diagonal(0,1) = 0);

  should_fail = true;
  EVAL2("Array submatrix_on_diagonal member function to non-square matrix", myMatrix22, B, myMatrix33, N, B = N.submatrix_on_diagonal(1,2));
  should_fail = false;

#ifndef MARVEL_STYLE
  if (adept::have_matrix_multiplication()) {
    HEADING("MATRIX MULTIPLICATION");
    EVAL2("Matrix-Matrix multiplication", myMatrix33, S, myMatrix23, M, S = M.T() ** M);
    EVAL2("Matrix-Matrix multiplication with matmul", myMatrix33, S, myMatrix23, M, S = matmul(M.T(), M));

    should_fail = true;
    EVAL2("Matrix-Matrix multiplication with inner dimension mismatch", myMatrix33, S, myMatrix23, M, S = M ** M);
    should_fail = false;
    
    // TESTING!
    EVAL2("Matrix-Matrix-Vector multiplication", myVector3, v, myMatrix33, S, v = S ** S ** v);
    
    EVAL2("Matrix-Matrix-Vector multiplication", myVector3, v, myMatrix33, S, v = S ** log(S) ** S(0,__));
    EVAL2("Vector-Matrix multiplication", myVector3, v, myMatrix33, S, v = v ** S);
    EVAL2("Vector-Matrix multiplication with matmul", myVector3, v, myMatrix33, S, v = matmul(v, S));
    EVAL2("SymmMatrix-Vector multiplication", myVector3, v, mySymmMatrix, O, v = O ** v);
    EVAL2("SymmMatrix-Matrix multiplication", myMatrix33, S, mySymmMatrix, O, S = O ** S);
    EVAL2("Vector-SymmMatrix multiplication", myVector3, v, mySymmMatrix, O, v = v ** O);
    EVAL2("Matrix-SymmMatrix multiplication", myMatrix23, M, mySymmMatrix, O, M = M ** O);
    EVAL2("DiagMatrix-Vector multiplication", myVector3, v, myDiagMatrix, D, v = D ** v);
    EVAL2("TridiagMatrix-Vector multiplication", myVector3, v, myTridiagMatrix, T, v = T ** v);
    EVAL2("TridiagMatrix-Matrix multiplication", myMatrix33, S, myTridiagMatrix, T, S = T ** S);
    EVAL2("Vector-TridiagMatrix multiplication", myVector3, v, myTridiagMatrix, T, v = v ** T);
    EVAL2("Matrix-TridiagMatrix multiplication", myMatrix23, M, myTridiagMatrix, T, M = M ** T);
  }
  else {
    std::cout << "NO MATRIX MULTIPLICATION TESTS PERFORMED BECAUSE ADEPT COMPILED WITHOUT LAPACK\n";
  }

#ifndef ALL_ACTIVE
  if (adept::have_linear_algebra()) {
    HEADING("LINEAR ALGEBRA");
    EVAL2("Solving general linear equations Ax=b", myVector3, v, myMatrix33, S, v = solve(S,v));
    
    EVAL2("Solving general linear equations AX=B", myMatrix23, M, myMatrix33, S, M.T() = solve(S,M.T()));
    EVAL2("Solving linear equations Ax=b with symmetric A", myVector3, v, mySymmMatrix, O, v = solve(O,v));
    EVAL2("Solving linear equations AX=B with symmetric A", myMatrix23, M, mySymmMatrix, O, M.T() = solve(O,M.T()));
    EVAL2("Invert general matrix", myMatrix33, C, myMatrix33, S, C = inv(S));
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


  HEADING("FILLING ARRAYS");
  EVAL("Fill vector with \"<<\"", myVector3, v, (v << 0.1, 0.2));

  should_fail = true;
  EVAL("Overfill vector with \"<<\"", myVector3, v, (v << 0.1, 0.2, 0.3, 0.4));
  should_fail = false;

  EVAL("Underfill matrix with \"<<\"", myMatrix23, M, (M << 0.1, 0.2, 0.3, 0.4, 0.5));
  EVAL("Fill matrix with \"<<\"", myMatrix23, M, (M << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6));

  should_fail = true;
  EVAL("Overfill matrix with \"<<\"", myMatrix23, M, (M << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0));
  should_fail = false;

  EVAL("Fill vector with vectors using \"<<\"", myVector3, v, v << v(range(1,2)) << 0.1);
  EVAL2("Fill matrix with vector using \"<<\"", myMatrix23, M, myVector3, v, M << 0.1 << 0.2 << 0.3 << v);
  EVAL2("Fill matrix with vector using \"<<\"", myMatrix33, S, myVector3, v, S << v << v << v);
  EVAL("Assign array using range", myVector3, v, v = range(3,5));

#ifdef ADEPT_BOUNDS_CHECKING
  HEADING("BOUNDS CHECKING");
  should_fail = true;
  EVAL("Access vector out of bounds", myVector3, v, v(0) = v(4));
  EVAL("Access vector out of bounds", myVector3, v, v(0) = v(end-4));
  EVAL("Access matrix out of bounds", myMatrix23, M, M(0,0) = M(0,-1));
  EVAL("Access matrix out of bounds", myMatrix23, M, M(0,0) = M(end+1,1));
  should_fail = false;
#endif


  std::cout << "====================================================================\n";
  if (anomalous_results > 0) {
    std::cout << "*** In terms of run-time errors, there were " << anomalous_results << " incorrect results\n";
    return 1;
  }
  else {
    std::cout << "In terms of run-time errors, all tests were passed\n";
    return 0;
  }
}
