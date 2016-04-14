#include <iostream>
#include <adept_arrays.h>
#include "Timer.h"

using namespace adept;

int main()
{
  Timer timer;
  timer.print_on_exit();
  int n = 50;
  static const int rep = 1000;
  //  static const int rep = 10;

  Stack stack;

  aMatrix M(n,n), P(n,n), Q(n,n);
  //  Array<2,adouble,false> M(n,n), P(n,n), Q(n,n);
  adouble Mc[n][n], Pc[n][n], Qc[n][n];

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      P(i,j) = Pc[i][j] = Q(i,j) = Qc[i][j] = 0.01 * (i-j);
      M(i,j) = Mc[i][j] = 0.0;
    }
  }

  int t_c_style_w = timer.new_activity("C-style for loops (warm-up)");
  int t_c_style = timer.new_activity("C-style for loops");
  int t_adept_w = timer.new_activity("Adept (warm-up)");
  int t_adept = timer.new_activity("Adept");
  int t_adept_container_w = timer.new_activity("Adept container only (warm-up)");
  int t_adept_container = timer.new_activity("Adept container only");
  int t_jacobian_w = timer.new_activity("Jacobian (warm-up)");
  int t_jacobian = timer.new_activity("Jacobian");
  int t_jacobian_array_w = timer.new_activity("Jacobian array-op (warm-up)");
  int t_jacobian_array = timer.new_activity("Jacobian array-op");

  stack.new_recording();
  timer.start(t_c_style_w);
  for (int irep = 0; irep < rep; ++irep) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
	Mc[i][j] += Pc[i][j]/(Qc[i][j]+0.5);
      }
    }
  }
  timer.stop();
  std::cout << stack;

  stack.new_recording();
  timer.start(t_c_style);
  for (int irep = 0; irep < rep; ++irep) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
	Mc[i][j] += Pc[i][j]/(Qc[i][j]+0.5);
      }
    }
  }
  timer.stop();
  std::cout << stack;

  stack.independent(&Pc[0][0], n*n);
  stack.dependent(&Mc[0][0], n*n);

  timer.start(t_jacobian_w);
  double* jac;
  jac = new double[n*n*n*n];

  stack.jacobian_forward(jac);
  timer.stop();
  timer.start(t_jacobian);
  stack.jacobian_forward(jac);
  timer.stop();


  //  std::cout << Mc[0][0] << " " << Mc[10][10] << "\n";

  stack.new_recording();
  timer.start(t_adept_w);
  for (int irep = 0; irep < rep; ++irep) {
    //    M += noalias(P/(Q+0.5));
    M += P/(Q+0.5);
    //    M = P/(Q);
  }
  timer.stop();
  std::cout << stack;

  stack.new_recording();
  timer.start(t_adept);
  for (int irep = 0; irep < rep; ++irep) {
    //    M += noalias(P/(Q+0.5));
    M += P/(Q+0.5);
  }
  timer.stop();
  std::cout << stack;

  stack.clear_independents();
  stack.clear_dependents();
  stack.independent(P);
  stack.dependent(Q);
  //  stack.independent(P.data(), n*n);
  //  stack.dependent(M.data(), n*n);

  std::cout << stack;

  timer.start(t_jacobian_array_w);
  stack.jacobian_forward(jac);
  timer.stop();
  timer.start(t_jacobian_array);
  stack.jacobian_forward(jac);
  timer.stop();

  stack.new_recording();
  timer.start(t_adept_container_w);
  for (int irep = 0; irep < rep; ++irep) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
	M(i,j) += P(i,j)/(Q(i,j)+0.5);
      }
    }
  }
  timer.stop();
  std::cout << stack;
  //  std::cout << M;

  stack.new_recording();
  timer.start(t_adept_container);
  for (int irep = 0; irep < rep; ++irep) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
	M(i,j) += P(i,j)/(Q(i,j)+0.5);
      }
    }
  }
  timer.stop();
  std::cout << stack;
}
