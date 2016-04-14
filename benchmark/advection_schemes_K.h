/* advection_schemes_K.h - Header for hand-coded Jacobians

  Copyright (C) 2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include "nx.h"

void lax_wendroff_K(int nt, double c, const double q_init[NX],
		    double q[NX], double jacobian[NX*NX]);
void toon_K(int nt, double c, const double q_init[NX],
	    double q[NX], double jacobian[NX*NX]);
