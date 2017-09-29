/* advection_schemes_AD.h - Header for the hand-coded adjoints

  Copyright (C) 2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#ifndef ADVECTION_SCHEMES_AD_H
#define ADVECTION_SCHEMES_AD_H

#include "nx.h"

// Hand-coded adjoint of Lax-Wendroff advection scheme
template <typename real>
void lax_wendroff_AD(int nt, real c, const real q_init[NX], real q[NX],
		     const real q_AD_const[NX], real q_init_AD[NX]) {
  // Forward pass
  real flux[NX-1];

  for (int i = 0; i < NX; i++)   q[i] = q_init[i];

  // Forward pass
  for (int j = 0; j < nt; j++) {
    for (int i = 0; i < NX-1; i++)  flux[i] = 0.5*c*(q[i]+q[i+1]+c*(q[i]-q[i+1]));
    for (int i = 1; i < NX-1; i++)  q[i] += flux[i-1]-flux[i];
    q[0] = q[NX-2]; q[NX-1] = q[1];  // Treat boundary conditions
  }

  real q_AD[NX];
  real flux_AD[NX-1];
  for (int i = 0; i < NX; i++) q_AD[i] = q_AD_const[i];
  for (int i = 0; i < NX-1; i++) flux_AD[i] = 0.0;
  
  // Reverse pass
  for (int j = nt-1; j >= 0; j--) {
    q_AD[NX-2] += q_AD[0];
    q_AD[0] = 0.0;
    q_AD[1] += q_AD[NX-1];
    q_AD[NX-1] = 0.0;

    for(int i = 1; i < NX-1; i++) {
      flux_AD[i-1] += q_AD[i];
      flux_AD[i] -= q_AD[i];
      //      q_AD[i] = 0.0;
    }
    real factor1 = 0.5*c*(1.0+c);
    real factor2 = 0.5*c*(1.0-c);
    for (int i = 0; i < NX-1; i++) {
      q_AD[i] += factor1*flux_AD[i];
      q_AD[i+1] += factor2*flux_AD[i];
      flux_AD[i] = 0.0;
    }
  }
  for (int i = 0; i < NX; i++) {
    q_init_AD[i] = q_AD[i];
    q_AD[i] = 0.0;
  }
}

// Hand-coded adjoint of Toon advection scheme
template <typename real>
void toon_AD(int nt, real c, const real q_init[NX], real q_out[NX],
	     const real q_AD_const[NX], real q_init_AD[NX]) {
  // Forward pass
  real flux[NX-1];

  real* q_save = new real[NX*(nt+1)];
  //  real q_save[NX*(nt+1)];
  real* q = &(q_save[0]);

  for (int i = 0; i < NX; i++)   q[i] = q_init[i];

  // Forward pass
  for (int j = 0; j < nt; j++) {
    for (int i=0; i<NX-1; i++) flux[i] = (exp(c*log(q[i]/q[i+1]))-1.0) 
                                         * q[i]*q[i+1] / (q[i]-q[i+1]);
    q += NX;
    for (int i = 1; i < NX-1; i++)  q[i] = q[i-NX]+flux[i-1]-flux[i];
    q[0] = q[NX-2]; q[NX-1] = q[1];  // Treat boundary conditions
  }

  for (int i = 0; i < NX; i++) q_out[i] = q[i];

  real q_AD[NX];
  real flux_AD[NX-1];
  for (int i = 0; i < NX; i++) q_AD[i] = q_AD_const[i];
  for (int i = 0; i < NX-1; i++) flux_AD[i] = 0.0;
  
  // Reverse pass
  for (int j = nt-1; j >= 0; j--) {
    q_AD[NX-2] += q_AD[0];
    q_AD[0] = 0.0;
    q_AD[1] += q_AD[NX-1];
    q_AD[NX-1] = 0.0;

    for(int i = 1; i < NX-1; i++) {
      flux_AD[i-1] += q_AD[i];
      flux_AD[i] -= q_AD[i];
      //      q_AD[i] = 0.0;
    }
    q -= NX;
    for (int i = 0; i < NX-1; i++) {
      real factor = exp(c*log(q[i]/q[i+1]));
      real one_over_q_i = 1.0/q[i];
      real one_over_q_i_plus_one = 1.0/q[i+1];
      real one_over_denominator = 1.0/(one_over_q_i+one_over_q_i_plus_one);
      q_AD[i] += one_over_denominator*one_over_q_i
	* (c*factor - (factor-1.0)*one_over_denominator*one_over_q_i)
	* flux_AD[i];
      q_AD[i+1] += one_over_denominator*one_over_q_i_plus_one
	* (- c*factor + (factor-1.0)*one_over_denominator*one_over_q_i_plus_one)
	* flux_AD[i];
      flux_AD[i] = 0.0;
    }
  }
  for (int i = 0; i < NX; i++) {
    q_init_AD[i] = q_AD[i];
    q_AD[i] = 0.0;
  }

  delete[] q_save;
}

#endif
