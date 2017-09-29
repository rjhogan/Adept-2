/* simulate_radiances.cpp - provides a function taking inactive arguments that returns also Jacobian matrices

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include "adept.h"
#include "simulate_radiances.h"

using adept::aReal;
using adept::Real;

// Simulate a single radiance (W sr-1 m-3) given the wavelength (m),
// emissivity profile, surface temperature (K) and temperature profile
// (K), where the profile data are located at n points with spacing
// 1000 m. This function uses active arguments. It is accessible only
// from within this file; the public interface is the
// simulate_radiance function.
static
aReal
simulate_radiance_private(int n,
			  Real wavelength,
			  const Real* emissivity,
			  const aReal& surface_temperature,
			  const aReal* temperature)
{
  static const Real BOLTZMANN_CONSTANT = 1.380648813e-23;
  static const Real SPEED_OF_LIGHT = 299792458.0;

  int i;
  aReal bt = surface_temperature; // Brightness temperature in K
  // Loop up through the atmosphere working out the contribution from
  // each layer
  for (i = 0; i < n; i++) {
    bt = bt*(1.0-emissivity[i]) + emissivity[i]*temperature[i];
  }
  // Convert from brightness temperature to radiance using
  // Rayleigh-Jeans approximation
  return 2.0*SPEED_OF_LIGHT*BOLTZMANN_CONSTANT*bt
    /(wavelength*wavelength*wavelength*wavelength);
}

// Simulate two radiances (W sr-1 m-3) given the surface temperature
// (K) and temperature profile (K), where the profile data are located
// at n points with spacing 1000 m. This function uses inactive
// arguments.
void
simulate_radiances(int n, // Size of temperature array
		   // Input variables:
		   Real surface_temperature, 
		   const Real* temperature,
		   // Output variables:
		   Real radiance[2],
		   // Output Jacobians:
		   Real dradiance_dsurface_temperature[2],
		   Real* dradiance_dtemperature)
{
  // First temporarily deactivate any existing Adept stack used by the
  // calling function
  adept::Stack* caller_stack = adept::active_stack();
  if (caller_stack != 0) {
    caller_stack->deactivate();
  }

  // Within the scope of these curly brackets, another Adept stack
  // will be used
  {
    // Ficticious oxygen channels around 60 GHz: wavelength in m
    static const Real wavelength[2] = {0.006, 0.0061}; 
    // Mass absorption coefficient of oxygen in m2 kg-1
    static const Real mass_abs_coefft[2] = {3.0e-5, 3.0e-3};
    // Layer thickness in m
    static const Real dz = 1000.0;

    // Density of oxygen in kg m-3
    std::vector<Real> density_oxygen(n);
    // Emissivity at a particular microwave wavelength
    std::vector<Real> emissivity(n);

    // Start a new stack
    adept::Stack s;

    // Create local active variables: surface temperature, temperature
    // and radiance
    aReal st = surface_temperature;
    std::vector<aReal> t(n);
    aReal r[2];

    // Initialize the oxygen density and temperature
    for (int i = 0; i < n; i++) {
      Real altitude = i*dz;
      // Oxygen density uses an assumed volume mixing ratio with air
      // of 21%, molecular mass of 16 (compared to 29 for air), a
      // surface air density of 1.2 kg m-3 and an atmospheric scale
      // height of 8000 m
      density_oxygen[i] = 1.2*0.21*(16.0/29.0)*exp(-altitude/8000.0);
      t[i] = temperature[i];
    }

    // Start recording derivative information
    s.new_recording();

    // Loop through the two channels
    for (int ichan = 0; ichan < 2; ichan++) {
      // Compute the emissivity profile
      for (int i = 0; i < n; i++) {
	emissivity[i] = 1.0-exp(-density_oxygen[i]*mass_abs_coefft[ichan]*dz);
      }
      // Simulate the radiance
      r[ichan] = simulate_radiance_private(n, wavelength[ichan], 
					   &emissivity[0], st, &t[0]);
      // Copy the aReal variable to the Real variable
      radiance[ichan] = r[ichan].value();
    }

    // Declare independent (x) and dependent (y) variables for
    // Jacobian matrix
    s.independent(st);
    s.independent(&t[0], n);
    s.dependent(r, 2);
    
    // Compute Jacobian matrix
    std::vector<Real> jacobian((n+1)*2);
    s.jacobian(&jacobian[0]);

    // Copy elements of Jacobian matrix into the calling arrays
    for (int ichan = 0; ichan < 2; ichan++) {
      dradiance_dsurface_temperature[ichan] = jacobian[ichan];
      for (int i = 0; i < n; i++) {
	dradiance_dtemperature[i*2+ichan] = jacobian[2+i*2+ichan];
      }
    }

    // At the following curly bracket, the local Adept stack will be
    // destructed
  }

  // Reactivate the Adept stack of the calling function
  if (caller_stack != 0) {
    caller_stack->activate();
  }
}
