/* test_packet_operations.cpp

  Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

  This file tests Adept's vectorization capabilities Adept vectors of
  types float and double, and also Packet<float> and Packet<double>
  that encapsulate the underlying intrinsic SIMD vector types.
*/

#include <iostream>
#include "adept_arrays.h"

using namespace adept;

template <typename Type>
Array<1,Type> p2v(internal::Packet<Type> p) {
  Array<1,Type> v(internal::Packet<Type>::size);
  p.put(v.data());
  return v;
}

template <typename Type>
void test_packet_operations() {
  static const int N = internal::Packet<Type>::size;
  std::cout << "\nADEPT PACKET\n";
  std::cout << "Type: " << sizeof(Type) << "-byte floating point numbers\n";
  std::cout << "Packet size: " << N << "\n";
  Array<1,Type> v(N), w(N);
  v = range(1,N);
  w = 2.0;
  internal::Packet<Type> p(v.data());
  internal::Packet<Type> q(w.data());
  std::cout << "p = " << p2v(p) << "\n";
  std::cout << "q = " << p2v(q) << "\n";
  std::cout << "p+q = " << p2v(p+q) << "\n";
  std::cout << "p-q = " << p2v(p-q) << "\n";
  std::cout << "p*q = " << p2v(p*q) << "\n";
  std::cout << "p/q = " << p2v(p/q) << "\n";
  std::cout << "sqrt(p) = " << p2v(sqrt(p)) << "\n";
  std::cout << "fmin(p,q) = " << p2v(fmin(p,q)) << "\n";
  std::cout << "fmax(p,q) = " << p2v(fmax(p,q)) << "\n";
  std::cout << "hsum(p) = " << hsum(p) << "\n";
  std::cout << "hprod(p) = " << hprod(p) << "\n";
  std::cout << "hmin(p) = " << hmin(p) << "\n";
  std::cout << "hmax(p) = " << hmax(p) << "\n";
}
  

template <typename Type>
void test_vector_operations(int N) {
  std::cout << "\nADEPT ARRAY\n";
  std::cout << "Type: " << sizeof(Type) << "-byte floating point numbers\n";
  std::cout << "Packet size: " << internal::Packet<Type>::size << "\n";
  Array<1,Type> v(N), w(N);
  v = range(1,N);
  w = 2.0;
  std::cout << "v = " << v << "\n";
  std::cout << "w = " << w << "\n";
  std::cout << "v+w = " << v+w << "\n";
  std::cout << "v-w = " << v-w << "\n";
  std::cout << "v*w = " << v*w << "\n";
  std::cout << "v/w = " << v/w << "\n";
  std::cout << "sqrt(v) = " << sqrt(v) << "\n";
  std::cout << "fmin(v,w) = " << fmin(v,w) << "\n";
  std::cout << "fmax(v,w) = " << fmax(v,w) << "\n";
  std::cout << "sum(v) = " << sum(v) << "\n";
  std::cout << "product(v) = " << product(v) << "\n";
  std::cout << "minval(v) = " << minval(v) << "\n";
  std::cout << "maxval(v) = " << maxval(v) << "\n";
}

template <typename Type>
void test_unaligned_reduce(int N) {
  std::cout << "\nUNALIGNED REDUCE\n";
  std::cout << "Type: " << sizeof(Type) << "-byte floating point numbers\n";
  std::cout << "Packet size: " << internal::Packet<Type>::size << "\n";
  Array<1,Type> v(N);
  v = range(1,N);
  std::cout << "v = " << v << "\n";
  std::cout << "sum(v(range(1,end-1))) = " << sum(v(range(1,end-1))) << "\n";
}

template <typename Type>
void test_unaligned_assign(int N) {
  std::cout << "\nUNALIGNED ASSIGN\n";
  std::cout << "Type: " << sizeof(Type) << "-byte floating point numbers\n";
  std::cout << "Packet size: " << internal::Packet<Type>::size << "\n";
  Array<1,Type> v(N), w(N), x(N);
  v = range(1,N);
  w = 2.0;
  x = 0.0;
  std::cout << "v = " << v << "\n";
  std::cout << "w = " << w << "\n";
  std::cout << "x = " << x << "\n";
  std::cout << "x(range(1,end-1)) = v(range(1,end-1))+w(range(1,end-1)) ->\n";
  x(range(1,end-1)) = v(range(1,end-1))+w(range(1,end-1));
  std::cout << "x = " << x << "\n";

}

int
main(int argc, const char** argv)
{
  // Vectorization is only carried out on arrays of length twice the
  // packet length or longer
  static const int N = 2*internal::Packet<float>::size;

  test_packet_operations<float>();
  test_packet_operations<double>();

  Packet<double> d(2.0);
  Packet<double> e = fastexp(d);
  std::cout << "e=" << e << "\n";
  
  test_vector_operations<float>(N);
  test_vector_operations<double>(N);

  test_unaligned_reduce<float>(2*N);
  test_unaligned_reduce<double>(2*N);

  test_unaligned_assign<float>(2*N);
  test_unaligned_assign<double>(2*N);

  return 0;
}
