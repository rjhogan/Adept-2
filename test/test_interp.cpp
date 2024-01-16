/* test_interp.cpp

  Copyright (C) 2024 European Centre for Medium-Range Weather Forecasts

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

  This file tests interpolation operations
*/

#include <iostream>
#include "adept_arrays.h"

using namespace adept;

#define TEST(FUNC)				\
  {						\
    std::cout << #FUNC << " =\n";		\
    std::cout << FUNC << "\n";			\
  }

int
main(int argc, const char** argv)
{
  set_array_print_style(PRINT_STYLE_MATLAB);
  {
    std::cout << "*** 1D interpolation ***\n\n";
    int nx = 3;
    Vector x = pow(linspace(1.0,nx,nx),2.0);
    Vector m = sqrt(linspace(1.0,nx,nx));
    Vector xi = {4.0, 2.0, 0.5, 10.0};
    std::cout << "Coordinate vector and interpolation vector:\n";
    std::cout << "x = " << x << "\n";
    std::cout << "m = " << m << "\n";
    std::cout << "xi = " << xi << "\n";
    std::cout << "...which are:\n"
	      << "  (1) at a point in the interpolation vector,\n"
	      << "  (2) between points in the interpolation vector,\n"
	      << "  (3) off the left of the interpolation vector, and\n"
	      << "  (4) off the right of the interpolation vector.\n\n";
    TEST(interp(x,m,xi));
    TEST(interp(x,m,xi,ADEPT_EXTRAPOLATE_LINEAR));
    TEST(interp(x,m,xi,ADEPT_EXTRAPOLATE_CLAMP));
    TEST(interp(x,m,xi,ADEPT_EXTRAPOLATE_CONSTANT));
    TEST(interp(x,m,xi,ADEPT_EXTRAPOLATE_CONSTANT,-1000.0));
    TEST(interp(x(stride(end,0,-1)),m(stride(end,0,-1)),xi,ADEPT_EXTRAPOLATE_LINEAR));
    TEST(interp(x+0.0,m+0.0,xi+0.0,ADEPT_EXTRAPOLATE_LINEAR));

    Matrix M = spread<1>(m,2);
    std::cout << "\n*** Multiple 1D interpolation ***\n";
    std::cout << "M = " << M << "\n";
    TEST(interp(x,M,xi));

  }

  {
    std::cout << "\n*** 2D interpolation ***\n\n";
    int nx = 4;
    int ny = 3;

    Vector y = pow(linspace(1.0,ny,ny),2.0);
    Vector x = linspace(1.0,nx,nx);
    Matrix M = outer_product(y,x);
    
    Vector yi = {4.0, 2.0, 6.5, 0.5};
    Vector xi = {2.0, 3.8, 0.5, 5.0};

    std::cout << "Coordinate vectors and interpolation matrix:\n";
    std::cout << "y = " << y << "\n";
    std::cout << "x = " << x << "\n";
    std::cout << "M = " << M << "\n";
    std::cout << "\nTo be interpolated to the following points:\n";
    std::cout << "yi = " << yi << "\n";
    std::cout << "xi = " << xi << "\n";
    std::cout << "...which are:\n"
	      << "  (1) at a point in the interpolation matrix,\n"
	      << "  (2) between points in the interpolation matrix,\n"
	      << "  (3) off the left of the matrix, and\n"
	      << "  (4) off the top-right of the matrix.\n\n";
  
    TEST(interp2d(y,x,M,yi,xi));
    TEST(interp2d(y,x,M,yi,xi,ADEPT_EXTRAPOLATE_LINEAR));
    TEST(interp2d(y,x,M,yi,xi,ADEPT_EXTRAPOLATE_CLAMP));
    TEST(interp2d(y,x,M,yi,xi,ADEPT_EXTRAPOLATE_CONSTANT));
    TEST(interp2d(y,x,M,yi,xi,ADEPT_EXTRAPOLATE_CONSTANT,-1000.0));
    TEST(interp2d(y(stride(end,0,-1)),x,M(stride(end,0,-1),__),yi,xi,ADEPT_EXTRAPOLATE_LINEAR));
    TEST(interp2d(y+0.0,x+0.0,M+0.0,yi+0.0,xi+0.0,ADEPT_EXTRAPOLATE_LINEAR));

    Array3D A = spread<2>(M,2);
    std::cout << "\n*** Multiple 2D interpolation ***\n";
    std::cout << "A = " << A << "\n";
    TEST(interp2d(y,x,A,yi,xi));
  }

  {
    std::cout << "\n*** 3D interpolation ***\n\n";
    int nx = 4;
    int ny = 3;
    int nz = 2;

    Vector z = linspace(1.0,nz,nz);
    Vector y = linspace(1.0,ny,ny);
    Vector x = pow(linspace(1.0,nx,nx),2.0);
    Array3D A(nz,ny,nx);
    A(0,__,__) = outer_product(y,x);
    A(1,__,__) = outer_product(y,x)+1.0;

    Vector zi = {2.0, 1.2, 1.5,  5.0};
    Vector yi = {2.0, 2.6, 0.5,  5.0};
    Vector xi = {4.0, 10.0,20.0, 0.5};

    std::cout << "Coordinate vectors and interpolation array:\n";
    std::cout << "z = " << z << "\n";
    std::cout << "y = " << y << "\n";
    std::cout << "x = " << x << "\n";
    std::cout << "A = " << A << "\n";
    std::cout << "\nTo be interpolated to the following points:\n";
    std::cout << "zi = " << zi << "\n";
    std::cout << "yi = " << yi << "\n";
    std::cout << "xi = " << xi << "\n";
    std::cout << "...which are:\n"
	      << "  (1) at a point in the interpolation array,\n"
	      << "  (2) between points in the interpolation array,\n"
	      << "  (3) off the array in two dimension but not the third, and\n"
	      << "  (4) off all dimensions of the array.\n\n";
  
    TEST(interp3d(z,y,x,A,zi,yi,xi));
    TEST(interp3d(z,y,x,A,zi,yi,xi,ADEPT_EXTRAPOLATE_LINEAR));
    TEST(interp3d(z,y,x,A,zi,yi,xi,ADEPT_EXTRAPOLATE_CLAMP));
    TEST(interp3d(z,y,x,A,zi,yi,xi,ADEPT_EXTRAPOLATE_CONSTANT));
    TEST(interp3d(z,y,x,A,zi,yi,xi,ADEPT_EXTRAPOLATE_CONSTANT,-1000.0));
    TEST(interp3d(z,y(stride(end,0,-1)),x,A(__,stride(end,0,-1),__),zi,yi,xi,ADEPT_EXTRAPOLATE_LINEAR));
    TEST(interp3d(z+0.0,y+0.0,x+0.0,A+0.0,zi+0.0,yi+0.0,xi+0.0,ADEPT_EXTRAPOLATE_LINEAR));

  }
  
  return 0;
}
