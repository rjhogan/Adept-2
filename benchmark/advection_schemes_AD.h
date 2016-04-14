/* advection_schemes_AD.h - Header for the hand-coded adjoints

  Copyright (C) 2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include "nx.h"

// Hand-coded adjoint of Lax-Wendroff advection scheme
void lax_wendroff_AD(int nt, double c, const double q_init[NX], double q[NX],
		     const double q_AD_const[NX], double q_init_AD[NX]);

// Hand-coded adjoint of Toon advection scheme
void toon_AD(int nt, double c, const double q_init[NX], double q_out[NX],
	     const double q_AD_const[NX], double q_init_AD[NX]);
