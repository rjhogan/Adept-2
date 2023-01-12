#include <iostream>
#include <adept_arrays.h>
#include "../include/adept/BasicArray.h"

using namespace adept;

typedef BasicArray<double,1> Vectord;
typedef BasicArray<double,2> Matrixd;
typedef BasicArray<double,1,internal::ARRAY_IS_REF> Vectord_ref;
typedef BasicArray<double,2,internal::ARRAY_IS_REF> Matrixd_ref;
typedef BasicArray<const double,1,internal::ARRAY_IS_REF> Vectord_cref;
typedef BasicArray<const double,2,internal::ARRAY_IS_REF> Matrixd_cref;

int main() {
  Vectord x(4);
  std::cerr << "1\n";
  x(0)=1.0;
  std::cerr << "2\n";
  x(1)=3.0;
  std::cerr << "3\n";
  x(2)=5.0;
  std::cerr << "x=" << x+1 << "\n";
  std::cerr << x.info_string() << "\n";
  std::cerr << "x=" << x << "\n";

  Matrixd M(2,3);
  M = 7.0;
  M(1,2) = 10;
  std::cerr << "M=" << M << "\n";
  Vectord_ref vv = M(1,range(0,1));
  //  Vectord_ref vv = M(1,_{0,1});
  std::cerr << "vv=" << vv << "\n";
  std::cerr << M(1,range(0,1)) << "\n";
  /*
  std::cerr << x.info_string() << "\n";
  std::cerr << "x=" << x << "\n";
  */
  return 0;
}
