/* array_shortcuts.h -- Definitions of "shortcut" typedefs for array types

    Copyright (C) 2015-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptArrayShortcuts_H
#define AdeptArrayShortcuts_H

#include <adept/Array.h>
#include <adept/SpecialMatrix.h>
#include <adept/FixedArray.h>

namespace adept {

  // ---------------------------------------------------------------------
  // Pretty typedefs to avoid the need for template arguments
  // ---------------------------------------------------------------------

  typedef Array<1> Vector;
  typedef Array<2> Matrix;
  typedef Array<3> Array3; // Deprecated
  typedef Array<3> Array3D;
  typedef Array<4> Array4D;
  typedef Array<5> Array5D;
  typedef Array<6> Array6D;
  typedef Array<7> Array7D;

  typedef Array<1,Index> IntVector;
  typedef Array<2,Index> IntMatrix;
  typedef Array<3,Index> IntArray3; // Deprecated
  typedef Array<3,Index> IntArray3D;

  typedef Array<1,int> intVector;
  typedef Array<2,int> intMatrix;
  typedef Array<3,int> intArray3; // Deprecated
  typedef Array<3,int> intArray3D;
  typedef Array<4,int> intArray4D;
  typedef Array<5,int> intArray5D;
  typedef Array<6,int> intArray6D;
  typedef Array<7,int> intArray7D;

  typedef Array<1,bool> boolVector;
  typedef Array<2,bool> boolMatrix;
  typedef Array<3,bool> boolArray3; // Deprecated
  typedef Array<3,bool> boolArray3D;
  typedef Array<4,bool> boolArray4D;
  typedef Array<5,bool> boolArray5D;
  typedef Array<6,bool> boolArray6D;
  typedef Array<7,bool> boolArray7D;

  typedef Array<1,float> floatVector;
  typedef Array<2,float> floatMatrix;
  typedef Array<3,float> floatArray3; // Deprecated
  typedef Array<3,float> floatArray3D;
  typedef Array<4,float> floatArray4D;
  typedef Array<5,float> floatArray5D;
  typedef Array<6,float> floatArray6D;
  typedef Array<7,float> floatArray7D;

  typedef SpecialMatrix<Real,internal::SquareEngine<ROW_MAJOR>,
    false> SquareMatrix;
  typedef SpecialMatrix<Real,internal::BandEngine<ROW_MAJOR,0,0>,
    false> DiagMatrix;
  typedef SpecialMatrix<Real,internal::BandEngine<ROW_MAJOR,1,1>,
    false> TridiagMatrix;
  typedef SpecialMatrix<Real,internal::BandEngine<ROW_MAJOR,2,2>,
    false> PentadiagMatrix;
  typedef SpecialMatrix<Real,internal::SymmEngine<ROW_LOWER_COL_UPPER>,
    false> SymmMatrix;
  typedef SpecialMatrix<Real,internal::LowerEngine<ROW_MAJOR>,
    false> LowerMatrix;
  typedef SpecialMatrix<Real,internal::UpperEngine<ROW_MAJOR>,
    false> UpperMatrix;

  typedef FixedArray<Real,false,2> Vector2;
  typedef FixedArray<Real,false,3> Vector3;
  typedef FixedArray<Real,false,4> Vector4;
  typedef FixedArray<Real,false,2,2> Matrix22;
  typedef FixedArray<Real,false,3,3> Matrix33;
  typedef FixedArray<Real,false,4,4> Matrix44;

  // If automatic differentiation is turned off then aVector and
  // friends become identical to their inactive counterparts
#ifdef ADEPT_NO_AUTOMATIC_DIFFERENTIATION
#define ADEPT_IS_ACTIVE false
#else
#define ADEPT_IS_ACTIVE true
#endif

  typedef Array<1,Real,ADEPT_IS_ACTIVE> aVector;
  typedef Array<2,Real,ADEPT_IS_ACTIVE> aMatrix;
  typedef Array<3,Real,ADEPT_IS_ACTIVE> aArray3; // Deprecated
  typedef Array<3,Real,ADEPT_IS_ACTIVE> aArray3D;
  typedef Array<4,Real,ADEPT_IS_ACTIVE> aArray4D;
  typedef Array<5,Real,ADEPT_IS_ACTIVE> aArray5D;
  typedef Array<6,Real,ADEPT_IS_ACTIVE> aArray6D;
  typedef Array<7,Real,ADEPT_IS_ACTIVE> aArray7D;

  typedef SpecialMatrix<Real,internal::SquareEngine<ROW_MAJOR>,
    ADEPT_IS_ACTIVE> aSquareMatrix;
  typedef SpecialMatrix<Real,internal::BandEngine<ROW_MAJOR,0,0>,
    ADEPT_IS_ACTIVE> aDiagMatrix;
  typedef SpecialMatrix<Real,internal::BandEngine<ROW_MAJOR,1,1>,
    ADEPT_IS_ACTIVE> aTridiagMatrix;
  typedef SpecialMatrix<Real,internal::BandEngine<ROW_MAJOR,2,2>,
    ADEPT_IS_ACTIVE> aPentadiagMatrix;
  typedef SpecialMatrix<Real,internal::SymmEngine<ROW_LOWER_COL_UPPER>,
    ADEPT_IS_ACTIVE> aSymmMatrix;
  typedef SpecialMatrix<Real,internal::LowerEngine<ROW_MAJOR>,
    ADEPT_IS_ACTIVE> aLowerMatrix;
  typedef SpecialMatrix<Real,internal::UpperEngine<ROW_MAJOR>,
    ADEPT_IS_ACTIVE> aUpperMatrix;

  typedef FixedArray<Real,ADEPT_IS_ACTIVE,2>   aVector2;
  typedef FixedArray<Real,ADEPT_IS_ACTIVE,3>   aVector3;
  typedef FixedArray<Real,ADEPT_IS_ACTIVE,4>   aVector4;
  typedef FixedArray<Real,ADEPT_IS_ACTIVE,2,2> aMatrix22;
  typedef FixedArray<Real,ADEPT_IS_ACTIVE,3,3> aMatrix33;
  typedef FixedArray<Real,ADEPT_IS_ACTIVE,4,4> aMatrix44;


#undef ADEPT_IS_ACTIVE

} // End namespace adept

#endif
