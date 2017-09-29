/* simulate_radiances.h - a function taking inactive arguments that returns also Jacobian matrices

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include <adept.h>

void simulate_radiances(int n, // Size of temperature array
			// Input variables:
			adept::Real surface_temperature, 
			const adept::Real* temperature,
			// Output variables:
			adept::Real radiance[2],
			// Output Jacobians:
			adept::Real dradiance_dsurface_temperature[2],
			adept::Real* dradiance_dtemperature);
