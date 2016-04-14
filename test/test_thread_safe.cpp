/* test_thread_safe.cpp - Tests that Adept is thread-safe

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.

  This program tests the thread-safety of the Adept library: compile
  with and without ADEPT_STACK_THREAD_UNSAFE defined, and run with
  -serial and -parallel command-line arguments.  It should crash only
  if ADEPT_STACK_THREAD_UNSAFE is defined AND -parallel is selected.
*/

#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

// Test what happens if thread safety is disabled by uncommenting the
// following
//#define ADEPT_STACK_THREAD_UNSAFE 1
#include "adept.h"

using adept::adouble;

// Number of points in spatial grid of simulation
#define NX 128

// "Toon" advection scheme applied to linear advection in a 1D
// periodic domain - see Adept paper for details
static
void
toon(int nt, double c, const adouble q_init[NX], adouble q[NX]) {
  adouble flux[NX-1];                        // Fluxes between boxes
  for (int i=0; i<NX; i++) q[i] = q_init[i]; // Initialize q
  for (int j=0; j<nt; j++) {                 // Main loop in time
    for (int i=0; i<NX-1; i++) flux[i] = (exp(c*log(q[i]/q[i+1]))-1.0) 
                                         * q[i]*q[i+1] / (q[i]-q[i+1]);
    for (int i=1; i<NX-1; i++) q[i] += flux[i-1]-flux[i];
    q[0] = q[NX-2]; q[NX-1] = q[1];          // Treat boundary conditions
  }
}

// Perform a simulation and compute the Jacobian two ways - this is to
// be run in parallel to test thread safety
static
bool
compute(int i, int nt, double dt, double q_init_save[NX])
{
  bool error_occurred = false; // Return value

  // Start an Adept stack before the first adouble object is
  // constructed
  adept::Stack s;

  adouble q_init[NX];  // Initial values of field as adouble array
  adouble q[NX];       // Final values 
  
  // Copy initial values
  for (int j = 0; j < NX; j++) {
    q_init[j] = q_init_save[j];
  }

  // Do something to the data specific to the loop
  q_init[i+5] = q_init[i+5] + 1.0;

  // Start a new recording of derivative statements; note that this
  s.new_recording();

  // Run the simulation with nt timesteps
  toon(nt, dt, q_init, q);

  s.independent(q_init, NX); // Declare independents
  s.dependent(q, NX);        // Declare dependents
  double jac_for[NX*NX];     // Where the Jacobian will be stored from forward computation
  double jac_rev[NX*NX];     // Where the Jacobian will be stored from reverse computation
  // Compute Jacobian two ways
  s.jacobian_forward(jac_for);
  s.jacobian_reverse(jac_rev);
    
  double rmsd = 0.0;
  for (int j = 0; j < NX*NX; j++) {
    if (jac_for[j] != jac_rev[j]) {
      double diff = jac_for[j]-jac_rev[j];
      rmsd += diff*diff;
    }
    }
  rmsd = sqrt(rmsd / (NX*NX));
    
#pragma omp critical
  {
    std::cout.flush();
    
#ifdef _OPENMP
    std::cout << "*** Iteration " << i << " executed by thread " << omp_get_thread_num() 
	      << " (stack address " << adept::active_stack() << "):\n";
#else
    std::cout << "*** Iteration " << i 
	      << " (stack address " << adept::active_stack() << "):\n";
#endif
      
    std::cout << "Used maximum of " << s.max_jacobian_threads() << " thread(s) for Jacobian calculation\n";
    
    if (rmsd > 1.0e-5) {
      std::cout << "*** ERROR: Jacobian from forward and reverse computations disagree (RMSD = "
		<< rmsd << ")\n";
      error_occurred = true;
    }
    else {
      std::cout << "CORRECT BEHAVIOUR: Jacobians from forward and reverse computations agree within tolerance\n";
    }
    
    if (i == 0) {
      // Print information about the data held in the stack
	std::cout << "Stack status for iteration 0:\n"
		  << s;
	// Print memory information
	std::cout << "Memory usage: " << s.memory() << " bytes\n\n";
    }
    
    std::cout << "\n";
  }
  
  return error_occurred;
}


int
main(int argc, char** argv)
{
  using adept::adouble;

  bool error_occurred = false;

  const double pi = 4.0*atan(1.0);

  // Edit these variables to change properties of simulation
  const int nt = 2000;        // Number of timesteps
  const double dt = 0.125;   // Timestep (actually a Courant number)
  const int ncomputations = 16;

  // Initial values of field as a double array
  double q_init_save[NX];

  bool is_parallel = true;

  if (argc > 1) {
    if (std::string("-serial") == argv[1]) {
      is_parallel = false;
    }
    else if (std::string("-parallel") == argv[1]) {
      is_parallel = true;
    }
    else {
      std::cout << "Usage: " << argv[0] << " [-serial|-parallel]\n";
      return 1;
    }
  }


  std::cout << "Running " << argv[0] << "...\n";
  
#ifdef ADEPT_STACK_THREAD_UNSAFE
  std::cout << "  Compiled to be THREAD UNSAFE\n";
#else
  std::cout << "  Compiled to be THREAD SAFE\n";
#endif

#ifdef _OPENMP
  std::cout << "  " << omp_get_num_procs() << " processors available running maximum of "
	    << omp_get_max_threads() << " threads\n";
  if (is_parallel) {
    std::cout << "  Performing " << ncomputations << " parallel computations,\n";
    std::cout << "    within which Jacobian (" << NX << "x" << NX << " matrix) calculations will be serial\n";
#ifdef ADEPT_STACK_THREAD_UNSAFE
    if (omp_get_max_threads() > 1) {
      std::cout << "*** You should expect this program to crash now!\n";
    }
#endif
  }
  else {
    std::cout << "  Performing " << ncomputations << " serial computations,\n";
    std::cout << "    within which Jacobian (" << NX << "x" << NX << " matrix) calculations will be in parallel\n";
  }
#else
  std::cout << "  Compiled with no OpenMP support\n";
#endif

  std::cout << "\n";
  std::cout.flush();


  // Initialize the field
  for (int i = 0; i < NX; i++) {
    q_init_save[i] = (0.5+0.5*sin((i*2.0*pi)/(NX-1.5)))+0.0001;
  }

  if (is_parallel) {
#pragma omp parallel for
    for (int i = 0; i < ncomputations; i++) {
      if (compute(i, nt, dt, q_init_save)) {
	error_occurred = true;
      }
    }
  }
  else {
    for (int i = 0; i < ncomputations; i++) {
      if (compute(i, nt, dt, q_init_save)) {
	error_occurred = true;
      }
    }
  }

  if (error_occurred) {
    std::cout << "An error occurred\n";
  }

  return error_occurred;
    
}
