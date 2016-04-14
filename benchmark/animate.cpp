/* animate.cpp - Visualize the advection

  Copyright (C) 2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include <string>
#include <iostream>
#include <time.h>

#include "advection_schemes.h"

int
main(int argc, char** argv)
{
  double q1_save[NX];
  double q2_save[NX];
  double* q1 = q1_save;
  double* q2 = q2_save;
  double pi = 4.0*atan(1.0);

  double min_q = -0.2;
  double max_q = 1.2;
  double dq = 0.05;

  double dt = 0.125;
  int nt = 8;
  int cycles = 5;

  int j_min = min_q/dq;
  int j_max = max_q/dq;

  std::string line;
  line.resize(NX);

  timespec t;
  t.tv_sec = 0;
  t.tv_nsec = 20000000;

  for (int i = 0; i < NX; i++) q1[i] = (0.5+0.5*sin((i*2.0*pi)/(NX-1.5)))+0.0001;
  for (int k = 0; k < cycles*NX/(nt*dt); k++) {
    std::cout << "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n";

    for (int j = j_max; j > 0; j--) {
      double q_thresh = j*dq;
      for (int i = 0; i < NX; i++) {
	if (q1[i] > q_thresh) {
	  line[i] = '#';
	}
	else {
	  line[i] = ' ';
	}
      }
      std::cout << line << "\n";
    }
    for (int i = 0; i < NX; i++) {
      line[i] = '-';
    }
    std::cout << line << "\n";
    for (int j = -1; j > j_min; j--) {
      double q_thresh = j*dq;
      for (int i = 0; i < NX; i++) {
	if (q1[i] <= q_thresh) {
	  line[i] = '$';
	}
	else {
	  line[i] = ' ';
	}
      }
      std::cout << line << "\n";
      std::cout.flush();
    }
    nanosleep(&t, 0);
    //toon(nt, dt, q1, q2);
    lax_wendroff(nt, dt, q1, q2);
    double* tmp = q1;
    q2 = q1;
    q1 = tmp;  
  }
  return 0;
}
