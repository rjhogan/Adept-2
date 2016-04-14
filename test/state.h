/* state.h - An object-oriented interface to an Adept-based minimizer

  Copyright (C) 2012-2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#ifndef STATE_H
#define STATE_H 1
#include <vector>
#include "adept.h"
class State {
public:
  // Construct a state with n state variables
  State(int n) { active_x_.resize(n); }
  // Minimize the function, returning true if minimization
  // successful, false otherwise
  bool minimize();
  // Get copy of state variables after minimization
  void x(std::vector<double>& x_out) const;
  // For input state variables x, compute the function J(x) and
  // return it
  double calc_function_value(const double* x);
  // For input state variables x, compute function and put its
  // gradient in dJ_dx
  double calc_function_value_and_gradient(const double* x, double* dJ_dx);
  // Return the size of the state vector
  unsigned int nx() const { return active_x_.size(); }
protected:
  // Active version of the function: the algorithm is contained in
  // the definition of this function (in
  // rosenbrock_banana_function.cpp)
  adept::adouble calc_function_value(const adept::adouble* x);
  // DATA
  adept::Stack stack_;                    // Adept stack object
  std::vector<adept::adouble> active_x_;  // Active state variables
};
#endif
