/* advection_schemes.h - Two test advection algorithms from the Adept paper

  Copyright (C) 2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

// Use templates so that these functions can be easily compiled with
// different automatic differentiation tools in order that the
// performance of these tools can be compared.

#ifndef ADVECTION_SCHEMES_H
#define ADVECTION_SCHEMES_H 1

#include <cmath>

// Use a fixed problem size
#include "nx.h"

// Lax-Wendroff scheme applied to linear advection
template <class aReal, typename Real>
void lax_wendroff(int nt, Real c, const aReal q_init[NX], aReal q[NX]) {
  aReal flux[NX-1];                        // Fluxes between boxes
  for (int i=0; i<NX; i++) q[i] = q_init[i]; // Initialize q 
  for (int j=0; j<nt; j++) {                 // Main loop in time
    for (int i=0; i<NX-1; i++) flux[i] = 0.5*c*(q[i]+q[i+1]+c*(q[i]-q[i+1]));
    for (int i=1; i<NX-1; i++) q[i] += flux[i-1]-flux[i];
    q[0] = q[NX-2]; q[NX-1] = q[1];          // Treat boundary conditions
  }
}

// Toon advection scheme applied to linear advection
template <class aReal, typename Real>
void toon(int nt, Real c, const aReal q_init[NX], aReal q[NX]) {
  aReal flux[NX-1];                        // Fluxes between boxes
  for (int i=0; i<NX; i++) q[i] = q_init[i]; // Initialize q
  for (int j=0; j<nt; j++) {                 // Main loop in time
    for (int i=0; i<NX-1; i++) {
      // Need to check if the difference between adjacent points is
      // not too small or we end up with close to 0/0.  Unfortunately
      // the "fabs" function is not always available in CppAD, hence
      // the following.
      //      aReal bigdiff = (q[i]-q[i+1])*1.0e6;
      //      if (bigdiff > q[i] || bigdiff < -q[i]) {
	flux[i] = (exp(c*log(q[i]/q[i+1]))-1.0)
	  * q[i]*q[i+1] / (q[i]-q[i+1]);
	//      }
	//      else {
	//	flux[i] = c*q[i]; // Upwind scheme
	//      }
    }
    for (int i=1; i<NX-1; i++) q[i] += flux[i-1]-flux[i];
    q[0] = q[NX-2]; q[NX-1] = q[1];          // Treat boundary conditions
  }
}

#include "adept_arrays.h"


template <typename T> struct is_active { static const bool value = false; };
template <> struct is_active<adept::aReal> { static const bool value = true; };

// Lax-Wendroff scheme applied to linear advection
template <typename aReal, typename Real>
void lax_wendroff_vector(int nt, Real c, const aReal q_init[NX], 
			 aReal q[NX]) {
  using namespace adept;
  typedef adept::Array<1,Real,::is_active<aReal>::value> my_vector;
  //  typedef adept::Array<1,Real,true> my_vector;
  my_vector Q(NX);
  my_vector F(NX-1);
  my_vector Qleft = Q(range(0,end-1));
  my_vector Qright = Q(range(1,end));
  my_vector Qcentre = Q(range(1,end-1));
  my_vector Fleft = F(range(0,end-1));
  my_vector Fright = F(range(1,end));
  for (int i=0; i<NX; i++) Q(i) = q_init[i]; // Initialize q 
  for (int j=0; j<nt; j++) {                 // Main loop in time
    F = 0.5*c*(Qleft+Qright+c*(Qleft-Qright));
    Qcentre += Fleft-Fright;
    Q(0) = Q(NX-2);
    Q(NX-1) = Q(1);
  }
  for (int i=0; i<NX; i++) q[i] = Q(i);
}

template <class aReal, typename Real>
void toon_vector(int nt, Real c, const aReal q_init[NX], aReal q[NX]) {
  using namespace adept;
  typedef adept::Array<1,Real,::is_active<aReal>::value> my_vector;
  my_vector Q(NX);
  my_vector F(NX-1);
  my_vector Qleft = Q(range(0,end-1));
  my_vector Qright = Q(range(1,end));
  my_vector Qcentre = Q(range(1,end-1));
  my_vector Fleft = F(range(0,end-1));
  my_vector Fright = F(range(1,end));
  for (int i=0; i<NX; i++) Q(i) = q_init[i]; // Initialize q
  for (int j=0; j<nt; j++) {                 // Main loop in time
    F = (exp(c*log(Qleft/Qright))-1.0)
      * Qleft*Qright / (Qleft-Qright);
    Qcentre += Fleft-Fright;
    Q(0) = Q(NX-2);
    Q(NX-1) = Q(1);
  }
  for (int i=0; i<NX; i++) q[i] = Q(i);
}
#endif
