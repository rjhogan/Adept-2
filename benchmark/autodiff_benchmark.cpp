/* autodiff_benchmark.cpp - Program to benchmark different automatic differentiation tools

  Copyright (C) 2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#include <sstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <valarray>

#include "differentiator.h"

#include <adept.h>
using adept::Real;

static
Real
rms(const std::vector<Real>& a, const std::vector<Real>&b)
{
  if (a.size() != b.size()) {
    throw differentiator_exception("Attempt to compute RMS difference between vectors of different size");
  }
  Real sum = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    sum += (a[i]-b[i])*(a[i]-b[i]);
  }
  return sqrt(sum/a.size());
}

static
void
usage(const char* argv0)
{
  std::cout << "Usage: " << argv0 << " [OPTIONS] where OPTIONS can be\n";
  std::cout << "  -h|--help          Print this message\n";
  std::cout << "  -a|--algorithm  s  Use test algorithms specified by string s which may be\n";
  std::cout << "                     \"all\" or a comma separated list with possible entries\n";
  std::cout << "                     " << test_algorithms() << "\n";
  std::cout << "  -t|--tool       s  Use automatic differentiation tools specified by string\n";
  std::cout << "                     s which may be \"all\" or a comma separated list with\n";
  std::cout << "                     possible entries " << autodiff_tools() << "\n";    
  std::cout << "  -r|--repeat     n  Benchmark repeats the simulation n times\n";
  std::cout << "  -j|--jrepeat    n  Repeat the Jacobian simulation n times\n";
  std::cout << "  -n|--timesteps  n  Simulation uses n timesteps\n";
  std::cout << "  --print-result     Print the final output from the simulation(s)\n";
  std::cout << "  --print-adjoint    Print the hand-coded adjoint\n";
  std::cout << "  --print-jacobian   Print the hand-coded Jacobian matrix\n";
  std::cout << "  --no-openmp        Don't use OpenMP to speed up Adept\n";
  std::cout << "  --jacobian-forward Force use of forward-mode Jacobian\n";
  std::cout << "  --jacobian-reverse Force use of reverse-mode Jacobian\n";
  std::cout << "  --tolerance     x  Agreement with hand-coded requires RMS difference < x\n";
  std::cout << "  --verify-only      No benchmark: only verify correctness of results\n";
  std::cout << "Return code: 0 if all automatic differentiation tools produce adjoints and\n"
    "  Jacobians whose RMS difference with the values from hand-coded\n"
    "  differentiation is less than the required tolerance; 1 otherwise.\n";
}

int
main(int argc, char** argv)
{
  int nt = 2000;
  int nr = 100;
  int nr_jacobian = nr/10;
  Real dt = 0.125;
  Real tolerance = 1.0e-5;
  int force_jacobian = 0;

  bool verbose = false;
  bool print_result = false;
  bool print_adjoint = false;
  bool print_jacobian = false;
  bool no_openmp = false;
  bool verify_only = false;

  std::valarray<bool> use_tool(N_AUTODIFF_TOOLS);
  std::valarray<bool> use_algorithm(N_TEST_ALGORITHMS);

  use_tool = true;
  use_algorithm = true;

  int iarg = 1;

  while (iarg < argc) {
    if (std::string("-h") == argv[iarg]
	|| std::string("--help") == argv[iarg]) {
      usage(argv[0]);
      return 0;
    }
    if (std::string("-v") == argv[iarg]
	|| std::string("--verbose") == argv[iarg]) {
      verbose = true;
    }
    else if (std::string("--print-result") == argv[iarg]) {
      print_result = true;
    }
    else if (std::string("--print-adjoint") == argv[iarg]) {
      print_adjoint = true;
    }
    else if (std::string("--print-jacobian") == argv[iarg]) {
      print_jacobian = true;
    }
    else if (std::string("--jacobian-forward") == argv[iarg]) {
      force_jacobian = +1;
    }
    else if (std::string("--jacobian-reverse") == argv[iarg]) {
      force_jacobian = -1;
    }
    else if (std::string("--no-openmp") == argv[iarg]) {
      no_openmp = true;
    }
    else if (std::string("--verify-only") == argv[iarg]) {
      verify_only = true;
    }
    else if (std::string("-a") == argv[iarg]
	     || std::string("--algorithm") == argv[iarg]) {
      if (++iarg < argc) {
	if (std::string(argv[iarg]) != "all") {
	  use_algorithm = false;
	  std::istringstream ss(argv[iarg]);
	  std::string alg;
	  while (std::getline(ss, alg, ',')) {
	    bool found = false;
	    for (int i = 0; i < N_TEST_ALGORITHMS; i++) {
	      if (alg == test_algorithm_string[i]) {
		use_algorithm[i] = true;
		found = true;
		break;
	      }
	    }
	    if (!found) {
	      std::cout << "Test algorithm \""
			<< alg << "\" not available; available algorithms are "
			<< test_algorithms() << "\n";
	    }
	  }
	}
      }
      else {
	std::cout << "Arguments \"-a\" or \"--algorithm\" need to be followed by a string containing a comma-separated list of algorithms\n";
	return 1;
      }
    }
    else if (std::string("-t") == argv[iarg]
	     || std::string("--tool") == argv[iarg]) {
      if (++iarg < argc) {
	if (std::string(argv[iarg]) != "all") {
	  use_tool = false;
	  std::istringstream ss(argv[iarg]);
	  std::string tool;
	  while (std::getline(ss, tool, ',')) {
	    bool found = false;
	    for (int i = 0; i < N_AUTODIFF_TOOLS; i++) {
	      if (tool == autodiff_tool_string[i]) {
		use_tool[i] = true;
		found = true;
		break;
	      }
	    }
	    if (!found) {
	      std::cout << "Automatic differentiation tool \""
			<< tool << "\" not available; available tools are "
			<< autodiff_tools() << "\n";
	    }
	  }
	}
      }
      else {
	std::cout << "Arguments \"-a\" or \"--algorithm\" need to be followed by a string containing a comma-separated list of algorithms\n";
	return 1;
      }
    }
    else if (std::string("-r") == argv[iarg]
	     || std::string("--repeat") == argv[iarg]) {
      if (++iarg < argc) {
	std::stringstream ss(argv[iarg]);
	if (ss >> nr) {
	  if (nr <= 0) { 
	    std::cout << "Number of repeats must be greater than zero\n";
	    return 1;
	  }
	}
	else {
	  std::cout << "Failed to read \""
		    << argv[iarg]
		    << "\"as an integer\n";
	  return 1;
	}
      }
      else {
	throw differentiator_exception("Arguments \"-r\" or \"--repeat\" need to be followed by a number");
      }
    }
    else if (std::string("-j") == argv[iarg]
	     || std::string("--jrepeat") == argv[iarg]) {
      if (++iarg < argc) {
	std::stringstream ss(argv[iarg]);
	if (ss >> nr_jacobian) {
	  if (nr <= 0) { 
	    throw differentiator_exception("Number of repeats must be greater than zero");
	  }
	}
	else {
	  std::string msg = "Failed to read \"";
	  msg += argv[iarg];
	  msg += "\"as an integer";
	  throw differentiator_exception(msg.c_str());
	}
      }
      else {
	throw differentiator_exception("Arguments \"-j\" or \"--jrepeat\" need to be followed by a number");
      }
    }
    else if (std::string("-n") == argv[iarg]
	     || std::string("--timesteps") == argv[iarg]) {
      if (++iarg < argc) {
	std::stringstream ss(argv[iarg]);
	if (ss >> nt) {
	  if (nt < 0) { 
	    throw differentiator_exception("Number of timesteps must be greater than or equal to zero");
	  }
	}
	else {
	  std::string msg = "Failed to read \"";
	  msg += argv[iarg];
	  msg += "\"as an integer";
	  throw differentiator_exception(msg.c_str());
	}
      }
      else {
	throw differentiator_exception("Arguments \"-n\" or \"--timesteps\" need to be followed by a number");
      }
    }
    else if (std::string("--tolerance") == argv[iarg]) {
      if (++iarg < argc) {
	std::stringstream ss(argv[iarg]);
	if (ss >> tolerance) {
	  if (tolerance < 0) { 
	    throw differentiator_exception("Tolerance must be greater than or equal to zero");
	  }
	}
	else {
	  std::string msg = "Failed to read \"";
	  msg += argv[iarg];
	  msg += "\"as a Real";
	  throw differentiator_exception(msg.c_str());
	}
      }
      else {
	throw differentiator_exception("Arguments \"-j\" or \"--jrepeat\" need to be followed by a number");
      }
    }
    else {
      std::string msg = "Argument \"";
      msg += argv[iarg];
      msg += "\" not understood\n";
      std::cout << msg;
      usage(argv[0]);
      return 1;
    }
    iarg++;
  }

  Real pi = 4.0*atan(1.0);
  std::vector<Real> q_init(NX);
  std::vector<Real> q(NX);
  std::vector<Real> q_AD(NX);
  std::vector<Real> q_init_AD(NX);
  std::vector<Real> q_init_AD_reference(NX);
  std::vector<Real> jac(NX*NX);
  std::vector<Real> jac_reference(NX*NX);

  int nr_warm_up = nr/10;
  int nr_jacobian_warm_up = nr_jacobian/10;
  if (nr_warm_up < 1) {
    nr_warm_up = 1;
  }
  if (nr_jacobian_warm_up < 1) {
    nr_jacobian_warm_up = 1;
  }

  if (verify_only) {
    nr = 0;
    nr_jacobian = 0;
    nr_warm_up = 1;
    nr_jacobian_warm_up = 1;
  }

  for (int i = 0; i < NX; i++) q_init[i] = (0.5+0.5*sin((i*2.0*pi)/(NX-1.5)))+1;
  for (int i = 0; i < NX; i++) q_AD[i] = 0.0;

  bool verify_error = false;

  Timer timer;

  std::cout << "Automatic differentiation benchmark and verification\n";
  std::cout << "   Automatic differentiation tools = ";
  bool is_first = true;
  for (int i = 0; i < N_AUTODIFF_TOOLS; i++) {
    if (use_tool[i]) {
      if (!is_first) {
	std::cout << ", ";
      }
      else {
	is_first = false;
      }
      std::cout << autodiff_tool_long_string[i];
    }
  }
  std::cout << "\n";

  std::cout << "   Test algorithms = ";
  is_first = true;
  for (int i = 0; i < N_TEST_ALGORITHMS; i++) {
    if (use_algorithm[i]) {
      if (!is_first) {
	std::cout << ", ";
      }
      else {
	is_first = false;
      }
      std::cout << test_algorithm_long_string[i];
    }
  }
  std::cout << "\n";

  std::cout << "   Number of x points = " << NX << "\n";
  std::cout << "   Number of timesteps = " << nt << ", Courant number = " << dt << "\n";
  if (!verify_only) {
    std::cout << "   Algorithm repeats = " << nr << ", warm-up repeats = " << nr_warm_up << "\n";
    std::cout << "   Jacobian repeats = " << nr_jacobian << ", warm-up repeats = " << nr_jacobian_warm_up << "\n";
  }
  else {
    std::cout << "   Verifying results only: no repeats\n";
  }

  std::cout << adept::configuration();

  // Loop through test algorithms
  for (int ialg = 0; ialg < N_TEST_ALGORITHMS; ialg++) {
    if (use_algorithm[ialg]) {

      std::string algorithm_string = test_algorithm_long_string[ialg];
      std::cout << "\nRunning test algorithm \"" << algorithm_string << "\":\n";
      
      TestAlgorithm ta = static_cast<TestAlgorithm>(ialg);
      
      std::cout << "   Hand coded (forward-mode Jacobian only)\n";
      
      HandCodedDifferentiator hand_coded_differentiator(timer, algorithm_string);
      hand_coded_differentiator.initialize(nt, dt);
      for (int i = 0; i < nr_warm_up; i++) {
	hand_coded_differentiator.func(ta, q_init, q);
	hand_coded_differentiator.adjoint(ta, q_init, q, q_AD, q_init_AD_reference);
	hand_coded_differentiator.jacobian(ta, q_init, q, jac_reference);
      }
      hand_coded_differentiator.reset_timings();
      for (int i = 0; i < nr; i++) {
	hand_coded_differentiator.func(ta, q_init, q);
	hand_coded_differentiator.adjoint(ta, q_init, q, q_AD, q_init_AD_reference);
	hand_coded_differentiator.jacobian(ta, q_init, q, jac_reference);
      }
      
      if (print_result) {
	std::cout << "      result = [" << q[0];
	for (int i = 1; i < NX; i++) {
	  std::cout << ", " << q[i];
	}
	std::cout << "]\n";
      }
      
      if (print_adjoint) {
	std::cout << "adjoint = [" << q[0];
	for (int i = 1; i < NX; i++) {
	  std::cout << ", " << q[i];
	}
	std::cout << "]\n";
      }
      if (print_jacobian) {
	Real (&q_K)[NX][NX]
	  = *reinterpret_cast<Real(*)[NX][NX]>(&jac_reference[0]);
	std::cout << "jacobian = [\n";
	for (int i = 0; i < NX; i++) {
	  std::cout << q_K[i][0];
	  for (int j = 1; j < NX; j++) {
	    std::cout << ", " << q_K[i][j];
	}
	  std::cout << "\n";
	}
	std::cout << "]\n";
      }
      
      Real base_time = timer.timing(hand_coded_differentiator.base_timer_id());
      
      if (!verify_only) {
	std::cout << "      Time of original algorithm: " << base_time << " seconds\n";
	std::cout << "      Absolute time of adjoint: " 
		  << timer.timing(hand_coded_differentiator.adjoint_compute_timer_id())
		  << " s\n";
	std::cout << "      Relative time of adjoint: " 
		  << timer.timing(hand_coded_differentiator.adjoint_compute_timer_id())
	  / base_time << "\n";
	std::cout << "      Absolute time of Jacobian: " 
		  << timer.timing(hand_coded_differentiator.jacobian_timer_id())
		  << " s\n";
	std::cout << "      Relative time of Jacobian: " 
		  << timer.timing(hand_coded_differentiator.jacobian_timer_id())
	  / base_time << "\n";
      }
      
      for (int itool = 0; itool < N_AUTODIFF_TOOLS; itool++) {
	if (use_tool[itool]) {
	  Differentiator* differentiator
	    = new_differentiator(static_cast<AutoDiffTool>(itool),
				 timer, algorithm_string);
	  if (!differentiator) {
	    if (verbose) std::cout << "Automatic differentiation tool with code " << itool << " not available\n";
	    continue;
	  }
	  
	  differentiator->initialize(nt, dt);
	  if (no_openmp) {
	    differentiator->no_openmp();
	  }
	  
	  std::cout << "   " << differentiator->name() << "\n";
	  
	  for (int i = 0; i < nr_warm_up; i++) {
	    differentiator->adjoint(ta, q_init, q, q_AD, q_init_AD);
	  }
	  Real rms_verify = rms(q_init_AD, q_init_AD_reference);
	  if (rms_verify > tolerance) {
	    std::cout << "      *** Adjoint RMS difference with hand-coded of " << rms_verify << " is greater than tolerance of " << tolerance << " ***\n";
	    verify_error = true;
	  }
	  else {
	    std::cout << "      Adjoint RMS difference with hand-coded of " << rms_verify << " is within tolerance of " << tolerance << "\n";
	  }

	  for (int i = 0; i < nr_jacobian_warm_up; i++) {
	    differentiator->jacobian(ta, q_init, q, jac, force_jacobian);
	  }
	  rms_verify = rms(jac, jac_reference);
	  if (rms_verify > tolerance) {
	    std::cout << "      *** Jacobian RMS difference with hand-coded of " << rms_verify << " is greater than tolerance of " << tolerance << " ***\n";
	    verify_error = true;
	  }
	  else {
	    std::cout << "      Jacobian RMS difference with hand-coded of " << rms_verify << " is within tolerance of " << tolerance << "\n";
	  }
	  
	  
	  if (!verify_only) {
	    differentiator->reset_timings();
	    for (int i = 0; i < nr; i++) {
	      differentiator->adjoint(ta, q_init, q, q_AD, q_init_AD);
	    }

	    Real relative_record_time = timer.timing(differentiator->base_timer_id())
	      / base_time;
	    Real relative_adjoint_time
	      = timer.timing(differentiator->adjoint_compute_timer_id())
	      / base_time;
	    Real relative_adjoint_prep_time
	      = timer.timing(differentiator->adjoint_prep_timer_id())
	      / base_time;

	    std::cout << "      Absolute time of adjoint: "
		      << timer.timing(differentiator->base_timer_id())
	      + timer.timing(differentiator->adjoint_compute_timer_id())
	      + timer.timing(differentiator->adjoint_prep_timer_id())
		      << " s (" 
		      << timer.timing(differentiator->base_timer_id())
		      << " s + ";
	    if (relative_adjoint_prep_time > 0.0) {
	      std::cout << timer.timing(differentiator->adjoint_prep_timer_id()) 
			<< " s + ";
	    }
	    std::cout <<  timer.timing(differentiator->adjoint_compute_timer_id())
		      << " s)\n";
	    std::cout << "      Relative time of adjoint: "
		      << relative_record_time + relative_adjoint_prep_time
	      + relative_adjoint_time
		      << " (" << relative_record_time << " + ";
	    if (relative_adjoint_prep_time > 0.0) {
	      std::cout << relative_adjoint_prep_time << " + ";
	    }
	    std::cout << relative_adjoint_time << ")\n";
	    differentiator->reset_timings();
	  }
	  
	  for (int i = 0; i < nr_jacobian; i++) {
	    differentiator->jacobian(ta, q_init, q, jac, force_jacobian);
	  }
	  
	  if (print_jacobian) {
	    Real (&q_K)[NX][NX]
	      = *reinterpret_cast<Real(*)[NX][NX]>(&jac[0]);
	    std::cout << "jacobian_auto = [\n";
	    for (int i = 0; i < NX; i++) {
	      std::cout << q_K[i][0];
	      for (int j = 1; j < NX; j++) {
		std::cout << ", " << q_K[i][j];
	      }
	      std::cout << "\n";
	    }
	    std::cout << "]\n";
	  }
	  
	  if (!verify_only) {
	    Real relative_record_time = (nr*timer.timing(differentiator->base_timer_id()))
	      /(nr_jacobian*base_time);
	    Real relative_jacobian_time = (nr*timer.timing(differentiator->jacobian_timer_id()))
	      /(nr_jacobian*base_time);
	    Real relative_adjoint_prep_time = (nr*timer.timing(differentiator->adjoint_prep_timer_id()))
	      /(nr_jacobian*base_time);
	    std::cout << "      Absolute time of Jacobian: "
		      << timer.timing(differentiator->base_timer_id())
	      + timer.timing(differentiator->adjoint_prep_timer_id())
	      + timer.timing(differentiator->jacobian_timer_id())
		      << " s ("
		      << timer.timing(differentiator->base_timer_id()) 
		      << " s + ";
	    if (relative_adjoint_prep_time > 0.0) {
	      std::cout << timer.timing(differentiator->adjoint_prep_timer_id())
			<< " s + ";
	    }
	    std::cout << timer.timing(differentiator->jacobian_timer_id())
		      << " s)\n";
	    std::cout << "      Relative time of Jacobian: "
		      << relative_record_time + relative_adjoint_prep_time + relative_jacobian_time
		      << " (" << relative_record_time << " + ";
	    if (relative_adjoint_prep_time > 0.0) {
	      std::cout << relative_adjoint_prep_time << " + ";
	    }
	    std::cout << relative_jacobian_time << ")\n";
	  }
	  	  differentiator->print();
	  delete differentiator;
	}
      }
    }
  }
  if (verify_error) {
    std::cout << "\nEXITING WITH ERROR CODE 1: ONE OR MORE OF THE AUTOMATIC DIFFERENTIATION\n"
	      << "TOOLS DID NOT REPRODUCE THE HAND-CODING RESULT\n";
    return 1;
  }
  else {
    std::cout << "\nAll tests were passed within tolerance\n";
    return 1;
  }
}

