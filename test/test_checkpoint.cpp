/* test_checkpoint.cpp - Test manual checkpointing of a simulation

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include <iostream>
#include <cmath>

#include "adept.h"
// This header file is in the same directory as adept.h in the Adept
// package
#include "Timer.h"

using adept::adouble;

// Number of points in spatial grid of simulation
#define NX 100

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

// Main program to test checkpointing
int
main(int argc, char** argv)
{
  Timer timer;
  timer.print_on_exit(true);

  const double pi = 4.0*atan(1.0);

  // Edit these variables to change properties of simulation
  const int nblocks = 100;   // Number of checkpoints
  const int nt = 100;        // Number of timesteps between checkpoints
  const double dt = 0.125;   // Timestep (actually a Courant number)

  // Initial values of field as a double array
  double q_init_save[NX];

  // First initialize the field - note that the Toon function does not
  // like identical values next to each other
  for (int i = 0; i < NX; i++) {
    q_init_save[i] = (0.5+0.5*sin((i*2.0*pi)/(NX-1.5)))+0.0001;
  }

  // We perform the simulation twice, once without checkpointing and
  // once with
  int full_id = timer.new_activity("Non-checkpointed simulation");
  int checkpointed_id = timer.new_activity("Checkpointed simulation");


  // PART 1: NON-CHECKPOINTED SIMULATION
  timer.start(full_id);
  { 
    // Note that we run each test in a pair of curly brackets so that
    // the Adept stack goes out of scope and is destructed before the
    // next test is performed
    std::cout << "*** NON-CHECKPOINTED SIMULATION ***\n";

    adept::Stack stack;

    adouble q_init[NX];  // Initial values of field as adouble array
    adouble q[NX];       // Final values 

    // Rate of change of cost function with respect to initial values
    // of the field
    double dJ_dq[NX];

    // Copy initial values
    for (int i = 0; i < NX; i++) {
      q_init[i] = q_init_save[i];
    }

    // Run a simulation with nt*nblocks timesteps
    stack.new_recording();
    toon(nt*nblocks, dt, q_init, q);

    // Define a "cost function" J that is the sum of squared
    // differences between the final field and the initial field
    adouble J = 0.0;
    for (int i = 0; i < NX; i++) {
      J += (q[i]-q_init_save[i])*(q[i]-q_init_save[i]);
    }

    // In order to get the gradients of the cost function with respect
    // to the initial field, we first set the seed gradient of the
    // cost function to unity
    J.set_gradient(1.0);

    // Perform adjoint calculation
    stack.reverse();

    // Extract the gradients
    adept::get_gradients(q_init, NX, dJ_dq);
  
    // Print out the results
    std::cout << "J=" << J << "\n";
    std::cout << "q_final=[";
    for (int i = 0; i < NX; i++) {
      std::cout << " " << q[i];
    }
    std::cout << "]\n";
    std::cout << "dJ_dq=[";
    for (int i = 0; i < NX; i++) {
      std::cout << " " << dJ_dq[i];
    }
    std::cout << "]\n";
    std::cout << stack;
  }


  // PART 2: CHECKPOINTED SIMULATION
  timer.start(checkpointed_id);
  {
    std::cout << "*** CHECKPOINTED SIMULATION ***\n";
    adept::Stack stack;

    // We save the field at each checkpoint, where 0 corresponds to
    // the initial values and nblocks-1 corresponds to the final
    // checkpoint (which is not the very final set of values of the
    // field).  Note that this will only work if nblocks is non-const
    // if you use gcc, which has a C++ extension to allow C99-style
    // variable-length arrays.
    adouble q_save[nblocks][NX];

    // This will be the very final set of values of the field
    adouble q[NX];

    // Rate of change of cost function with respect to initial values
    // of the field
    double dJ_dq[NX];

    // Copy initial values
    for (int i = 0; i < NX; i++) {
      q_save[0][i] = q_init_save[i];
    }

    // Run simulation in a set of blocks, saving the results each
    // time. Note that this step does not need to be automatically
    // differentiated, hence the use of pause_recording and
    // continue_recording.
    for (int i = 0; i < nblocks-1; i++) {
      stack.pause_recording();
      toon(nt, dt, q_save[i], q_save[i+1]);
      stack.continue_recording();
    }

    // Now we rerun the simulations multiple times with automatic
    // differentiation, each time stepping back to the previous block.
    // The first simulation is treated separately since this is the
    // one in which the gradient of the cost function is computed.
    stack.new_recording();
    toon(nt, dt, q_save[nblocks-1], q);

    // Define a "cost function" J that is the sum of squared
    // differences between the final field "q" and the initial field
    adouble J = 0.0;
    for (int i = 0; i < NX; i++) {
      J += (q[i]-q_init_save[i])*(q[i]-q_init_save[i]);
    }

    // In order to get the gradients of the cost function with respect
    // to the initial field, we first set the seed gradient of the
    // cost function to unity
    J.set_gradient(1.0);

    // Perform adjoint calculation
    stack.reverse();

    // Extract the gradients of the cost function with respect to the
    // values at the final checkpoint
    adept::get_gradients(q_save[nblocks-1], NX, dJ_dq);

    // Print out the simulation results (not yet the gradients)
    std::cout << "J=" << J << "\n";
    std::cout << "q_final=[";
    for (int i = 0; i < NX; i++) {
      std::cout << " " << q[i];
    }
    std::cout << "]\n";

    // Now we repeat the simulation starting one checkpoint earlier
    // each time, with the final simulation being performed starting
    // at the initial values of the field
    for (int i = nblocks-2; i >= 0; i--) {
      stack.new_recording();
      toon(nt, dt, q_save[i], q);

      // This time we use the set of gradients output from the previous
      // simulation (which can be thought of as dJ/dq_save[i+1]) as
      // the input gradients for the next
      adept::set_gradients(q, NX, dJ_dq);

      // Perform adjoint calculation
      stack.reverse();

      // Extract the next set of gradients (which can be thought of as
      // dJ/dq_save[i]) and place in dJ_dq ready for the next
      // iteration
      adept::get_gradients(q_save[i], NX, dJ_dq);
    }

    // Print out the gradients
    std::cout << "dJ_dq=[";
    for (int i = 0; i < NX; i++) {
      std::cout << " " << dJ_dq[i];
    }
    std::cout << "]\n";
    std::cout << stack;
  }
  timer.stop();

  return 0;

}
