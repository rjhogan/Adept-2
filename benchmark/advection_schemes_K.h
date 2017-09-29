/* advection_schemes_K.h - Header for hand-coded Jacobians

  Copyright (C) 2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#ifndef ADVECTION_SCHEMES_K_H
#define ADVECTION_SCHEMES_K_H

#include <cmath>
#include <iostream>

#include "nx.h"

// Lax-Wendroff scheme applied to linear advection
template <typename real>
void lax_wendroff_K(int nt, real c, const real q_init[NX],
		    real q[NX], real jacobian[NX*NX]) {
  real flux[NX-1];                         // Fluxes between boxes
  real flux_K[NX-1][NX];                   // Flux Jacobian (dflux/dq_init)
  //  real (&q_K)[NX][NX] = *reinterpret_cast<real(*)[NX][NX]>(jacobian);
  real q_K[NX][NX];
  real coeff1 = 0.5*c*(1.0+c);
  real coeff2 = 0.5*c*(1.0-c);

  for (int i=0; i<NX; i++) {
    q[i] = q_init[i];                        // Initialize q 
    for (int k=0; k<NX; k++) {
      q_K[i][k] = 0.0;                       // Initialize Jacobian
    }
    q_K[i][i] = 1.0;
  }
  for (int j=0; j<nt; j++) {                 // Main loop in time
    for (int i=0; i<NX-1; i++) {
      flux[i] = 0.5*c*(q[i]+q[i+1]+c*(q[i]-q[i+1]));
      for (int k=0; k<NX; k++) {
	flux_K[i][k] = coeff1*q_K[i][k] + coeff2*q_K[i+1][k];
      }
    }
    for (int i=1; i<NX-1; i++) {
      q[i] += flux[i-1]-flux[i];
      for (int k=0; k<NX; k++) {
	q_K[i][k] += flux_K[i-1][k]-flux_K[i][k];
      }
    }
    q[0] = q[NX-2]; q[NX-1] = q[1];          // Treat boundary conditions
    for (int k=0; k<NX; k++) {
      q_K[0][k] = q_K[NX-2][k];
      q_K[NX-1][k] = q_K[1][k];
    }
  }

  // Transpose the result
  for (int i = 0, index = 0; i < NX; i++) {
    for (int j = 0; j < NX; j++, index++) {
      jacobian[index] = q_K[j][i];
    }
  }

}


// Toon advection scheme applied to linear advection
template <typename real>
void toon_K(int nt, real c, const real q_init[NX], real q[NX],
	    real jacobian[NX*NX]) {
  real flux[NX-1];                        // Fluxes between boxes
  real flux_K[NX-1][NX];
  real q_K[NX][NX];

  for (int i=0; i<NX; i++) {
    q[i] = q_init[i]; // Initialize q
    for (int k=0; k<NX; k++) {
      q_K[i][k] = 0.0;                       // Initialize Jacobian
    }
    q_K[i][i] = 1.0;
  }
  for (int j=0; j<nt; j++) {                 // Main loop in time
    for (int i=0; i<NX-1; i++) {
      real coeff1, coeff2;
      // Ought to check if the difference between adjacent points is
      // not too small or we end up with close to 0/0, but this leads
      // to different results from the automatic differentiation
      //      if (fabs(q[i]-q[i+1]) > q[i]*1.0e-6) {
	real factor = exp(c*log(q[i]/q[i+1]));
	real one_over_denominator = 1.0/(q[i]-q[i+1]);
	coeff1 = one_over_denominator*q[i+1]
	  * (c*factor + (factor-1.0)*(1.0-q[i]*one_over_denominator));
	coeff2 = one_over_denominator*q[i]
	  * (- c*factor + (factor-1.0)*(1.0+q[i+1]*one_over_denominator));
	flux[i] = (factor-1.0) * q[i]*q[i+1]*one_over_denominator;
	/*
      }
      else {
	flux[i] = c*q[i]; // Upwind scheme
	coeff1 = c;
	coeff2 = 0.0;
      }
	*/
      for (int k=0; k<NX; k++) {
	flux_K[i][k] = coeff1*q_K[i][k] + coeff2*q_K[i+1][k];
      }
    }

    for (int i=1; i<NX-1; i++) {
      q[i] += flux[i-1]-flux[i];
      for (int k=0; k<NX; k++) {
	q_K[i][k] += flux_K[i-1][k]-flux_K[i][k];
      }
    }
    q[0] = q[NX-2]; q[NX-1] = q[1];          // Treat boundary conditions
    for (int k=0; k<NX; k++) {
      q_K[0][k] = q_K[NX-2][k];
      q_K[NX-1][k] = q_K[1][k];
    }
  }

  // Transpose the result
  for (int i = 0, index = 0; i < NX; i++) {
    for (int j = 0; j < NX; j++, index++) {
      jacobian[index] = q_K[j][i];
    }
  }
}

#endif
