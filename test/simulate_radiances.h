/* simulate_radiances.h - a function taking inactive arguments that returns also Jacobian matrices

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

void simulate_radiances(int n, // Size of temperature array
			// Input variables:
			double surface_temperature, 
			const double* temperature,
			// Output variables:
			double radiance[2],
			// Output Jacobians:
			double dradiance_dsurface_temperature[2],
			double* dradiance_dtemperature);
