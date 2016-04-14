/* test_radiances.cpp - "main" function for Test 3

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include "adept.h"
#include "simulate_radiances.h"

using adept::adouble;

// This function provides an Adept interface to the simulate_radiances
// function
void simulate_radiances_wrapper(int n,
				const adouble& surface_temperature,
				const adouble* temperature,
				adouble radiance[2]) {
  // Create inactive (double) versions of the active (adouble) inputs
  double st = value(surface_temperature);
  std::vector<double> t(n);
  for (int i = 0; i < n; ++i) t[i] = value(temperature[i]);
  
  // Declare variables to hold the inactive outputs and their Jacobians
  double r[2];
  double dr_dst[2];
  std::vector<double> dr_dt(2*n);
  
   // Call the function with the non-Adept interface
  simulate_radiances(n, st, &t[0], &r[0], dr_dst, &dr_dt[0]);
  
  // Copy the results into the active variables, but use set_value in order
  // not to write any equivalent derivative statement to the Adept stack
  radiance[0].set_value(r[0]);
  radiance[1].set_value(r[1]);
  
  // Loop over the two radiances and add the derivative statements to
  // the Adept stack
  for (int i = 0; i < 2; ++i) {
    // Add the first term on the right-hand-side of Equation 1 in the text
    radiance[i].add_derivative_dependence(surface_temperature, dr_dst[i]);
    // Now append the second term on the right-hand-side of Equation
    // 1. The third argument "n" of the following function says that
    // there are n terms to be summed, and the fourth argument "2"
    // says to take only every second element of the Jacobian dr_dt,
    // since the derivatives with respect to the two radiances have
    // been interlaced.  If the fourth argument is omitted then
    // relevant Jacobian elements will be assumed to be contiguous in
    // memory.
    radiance[i].append_derivative_dependence(temperature, &dr_dt[i], n, 2);
  }

  for (int i = 0; i < 2; ++i) {
    std::cout << "Channel " << i << "\n";
    std::cout << "d[radiance]/d[surface_temperature] = " << dr_dst[i] << "\n";
    std::cout << "d[radiance]/d[temperature] =";
    for (int j = 0; j < n; ++j) {
      std::cout << " " << dr_dt[i+j*2];
    }
    std::cout << "\n\n";
  }

}


int
main(int argc, char** argv)
{
  // Temperature (K) at 1000-m intervals from the mid-latitude summer
  // standard atmosphere
  static const int N_POINTS = 25;
  static const double temperature_profile[N_POINTS+1]
    = {294.0, 290.0, 285.0, 279.0, 273.0, 267.0, 261.0, 255.0,
       248.0, 242.0, 235.0, 229.0, 222.0, 216.0, 216.0, 216.0,
       216.0, 216.0, 216.0, 217.0, 218.0, 219.0, 220.0, 222.0,
       223.0, 224.0};

  // Start the Adept stack
  adept::Stack s;
  
  // Copy the temperature profile information into active variables
  adouble surface_temperature = temperature_profile[0];
  adouble temperature[N_POINTS];
  for (int i = 0; i < N_POINTS; i++) {
    temperature[i] = temperature_profile[i+1];
  }

  // The simulated radiances will be put here...
  adouble sim_radiance[2];

  // ...and compared to the observed radiances here with their 1-sigma
  // error
  double obs_radiance[2] = {0.00189, 0.00140};
  double radiance_error = 2.0e-5;

  // Start recording derivative information
  s.new_recording();

  // Simulate the radiances for the input surface temperature and
  // atmospheric temperature
  simulate_radiances_wrapper(N_POINTS, surface_temperature,
			     temperature, sim_radiance);

  std::cout << "Simulated radiances = "
	    << sim_radiance[0].value() << " "
	    << sim_radiance[1].value() << "\n";

  // Compute a "cost function" (or "penalty function") expressing the
  // sum of the squared number of error standard deviations the
  // simulated radiances are from the observed radiances
  adouble cost_function = 0.0;
  for (int ichan = 0; ichan < 2; ichan++) {
    cost_function
      += (sim_radiance[ichan] - obs_radiance[ichan])
       * (sim_radiance[ichan] - obs_radiance[ichan])
      / (radiance_error*radiance_error);
  }
  
  std::cout << "Cost function = " << cost_function << "\n";

  // We want the computed adjoints to be gradients of the cost
  // function with respect to the surface temperature or atmospheric
  // temperature
  cost_function.set_gradient(1.0);

  // Reverse-mode automatic differentiation
  s.reverse();

  // Extract the gradients  
  double dcost_dsurface_temperature = 0;
  double dcost_dtemperature[N_POINTS];
  surface_temperature.get_gradient(dcost_dsurface_temperature);
  adept::get_gradients(temperature, N_POINTS, dcost_dtemperature);


  std::cout << "d[cost_function]/d[surface_temperature] = "
	    << dcost_dsurface_temperature << "\n";
  std::cout << "d[cost_function]/d[temperature] =";
  for (int i = 0; i < N_POINTS; i++) {
    std::cout << " " << dcost_dtemperature[i];
  }
  std::cout << "\n";


}
