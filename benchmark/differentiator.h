/* differentiator.h

  Copyright (C) 2014 The University of Reading

  Copying and distribution of this file, with or without modification,
  are permitted in any medium without royalty provided the copyright
  notice and this notice are preserved.  This file is offered as-is,
  without any warranty.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <iostream>
#include <vector>
#include <exception>
#include <cmath>
#include <string>

#include "Timer.h"

#include "adept.h"

using adept::Real;

#ifdef HAVE_ADOLC
#include "adolc/adolc.h"
#endif

#ifdef HAVE_CPPAD
#include "cppad/cppad.hpp"
#endif

#ifdef HAVE_SACADO
#include "Sacado.hpp"
#endif

#include "advection_schemes.h"
#include "advection_schemes_AD.h"
#include "advection_schemes_K.h"


enum TestAlgorithm {
  TEST_ALGORITHM_LAX_WENDROFF = 0,
  TEST_ALGORITHM_TOON = 1,
  TEST_ALGORITHM_LAX_WENDROFF_VECTOR = 2,
  TEST_ALGORITHM_TOON_VECTOR = 3,
  N_TEST_ALGORITHMS
};

const char* test_algorithm_long_string[] = {"Lax-Wendroff", "Toon et al.",
					    "Lax-Wendroff vector", "Toon et al. vector"};
const char* test_algorithm_string[] = {"lw","toon","lw_vector", "toon_vector"};

inline
std::string
test_algorithms()
{
  std::string algs = test_algorithm_string[0];
  for (int i = 1; i < N_TEST_ALGORITHMS; i++) {
    algs += ",";
    algs += test_algorithm_string[i];
  }
  return algs;
}


class differentiator_exception : public std::exception {
public:
  differentiator_exception(const char* message = "An error occurred in differentiator.h")
  { message_ = message; }
  virtual const char* what() const throw()
  { return message_; }
protected:
  const char* message_;
};

class Differentiator {
public:
  Differentiator(Timer& timer) 
    : timer_(timer) {
    initialize(2000, 0.125); 
  }

  virtual ~Differentiator() { }

  virtual void print() { }

  void initialize(int nt, Real c) {
    nt_ = nt;
    c_ = c;
  }

  template <class ActiveRealType>
  void func(TestAlgorithm test_algorithm,
	    const std::vector<ActiveRealType>& x,
	    std::vector<ActiveRealType>& y) {
    timer_.start(base_timer_id_);
    if (test_algorithm == TEST_ALGORITHM_LAX_WENDROFF) {
      lax_wendroff(nt_, c_, &x[0], &y[0]);
    }
    else if (test_algorithm == TEST_ALGORITHM_TOON) {
      toon(nt_, c_, &x[0], &y[0]);
    }
    else if (test_algorithm == TEST_ALGORITHM_LAX_WENDROFF_VECTOR) {
      lax_wendroff_vector(nt_, c_, &x[0], &y[0]);
    }
    else if (test_algorithm == TEST_ALGORITHM_TOON_VECTOR) {
      toon_vector(nt_, c_, &x[0], &y[0]);
    }
    timer_.stop();
  }

  virtual bool adjoint(TestAlgorithm test_algorithm,
		       const std::vector<Real>& x,
		       std::vector<Real>& y,
		       const std::vector<Real>& y_AD,
		       std::vector<Real>& x_AD) {
    return false;
  }

  virtual bool jacobian(TestAlgorithm test_algorithm,
			const std::vector<Real>& x,
			std::vector<Real>& y,
			std::vector<Real>& jac,
			int force_jacobian = 0) {
    return false;
  }

  void reset_timings() {
    timer_.reset(base_timer_id_);
    timer_.reset(adjoint_prep_timer_id_);
    timer_.reset(adjoint_compute_timer_id_);
    timer_.reset(jacobian_timer_id_);
  }

  virtual std::string name() const = 0; //{ return "GENERIC"; }

  virtual void no_openmp() { }

  int base_timer_id() const { return base_timer_id_; }
  int adjoint_prep_timer_id() const { return adjoint_prep_timer_id_; }
  int adjoint_compute_timer_id() const { return adjoint_compute_timer_id_; }
  int jacobian_timer_id() const { return jacobian_timer_id_; }

protected:
  void init_timer(const std::string name_) {
    base_timer_id_ = timer_.new_activity(name() + " | " + name_ + " | record");
    adjoint_prep_timer_id_ = timer_.new_activity(name() + " | " + name_ + " | adjoint prep");
    adjoint_compute_timer_id_ = timer_.new_activity(name() + " | " + name_ + " | adjoint compute");
    jacobian_timer_id_ = timer_.new_activity(name() + " | " + name_ + " | Jacobian");
  }

protected:
  Timer& timer_;
  int nt_; // Number of timesteps to run
  Real c_;  // Courant number
  int base_timer_id_;
  int adjoint_prep_timer_id_;
  int adjoint_compute_timer_id_;
  int jacobian_timer_id_;
};

// ================= HAND CODED ===========================
#include "advection_schemes_AD.h"

class HandCodedDifferentiator
  : public Differentiator {
public:
  HandCodedDifferentiator(Timer& timer, const std::string& name_)
    : Differentiator(timer) {
    init_timer(name_);
  }

  virtual bool adjoint(TestAlgorithm test_algorithm,
		       const std::vector<Real>& x,
		       std::vector<Real>& y,
		       const std::vector<Real>& y_AD,
		       std::vector<Real>& x_AD) {
    if (test_algorithm == TEST_ALGORITHM_LAX_WENDROFF) {
      timer_.start(adjoint_compute_timer_id_);
      lax_wendroff_AD(nt_, c_, &x[0], &y[0], &y_AD[0], &x_AD[0]);
      timer_.stop();
    }
    else if (test_algorithm == TEST_ALGORITHM_TOON) {
      timer_.start(adjoint_compute_timer_id_);
      toon_AD(nt_, c_, &x[0], &y[0], &y_AD[0], &x_AD[0]);
      timer_.stop();
    }
    else if (test_algorithm == TEST_ALGORITHM_LAX_WENDROFF_VECTOR) {
      timer_.start(adjoint_compute_timer_id_);
      lax_wendroff_AD(nt_, c_, &x[0], &y[0], &y_AD[0], &x_AD[0]);
      timer_.stop();
    }
    else if (test_algorithm == TEST_ALGORITHM_TOON_VECTOR) {
      timer_.start(adjoint_compute_timer_id_);
      toon_AD(nt_, c_, &x[0], &y[0], &y_AD[0], &x_AD[0]);
      timer_.stop();
    }
    else {
      std::cerr << "Algorithm not found: " << test_algorithm << "\n";
      return false;
    }
    return true;
  }

  virtual bool jacobian(TestAlgorithm test_algorithm,
			const std::vector<Real>& x,
			std::vector<Real>& y,
			std::vector<Real>& jac,
			int force_jacobian = 0) {
    jac.resize(NX*NX);
    if (test_algorithm == TEST_ALGORITHM_LAX_WENDROFF) {
      timer_.start(jacobian_timer_id_);
      lax_wendroff_K(nt_, c_, &x[0], &y[0], &jac[0]);
      timer_.stop();
    }
    else if (test_algorithm == TEST_ALGORITHM_TOON) {
      timer_.start(jacobian_timer_id_);
      toon_K(nt_, c_, &x[0], &y[0], &jac[0]);
      timer_.stop();
    }
    else if (test_algorithm == TEST_ALGORITHM_LAX_WENDROFF_VECTOR) {
      timer_.start(jacobian_timer_id_);
      lax_wendroff_K(nt_, c_, &x[0], &y[0], &jac[0]);
      timer_.stop();
    }
    else if (test_algorithm == TEST_ALGORITHM_TOON_VECTOR) {
      timer_.start(jacobian_timer_id_);
      toon_K(nt_, c_, &x[0], &y[0], &jac[0]);
      timer_.stop();
    }
    else {
      std::cerr << "Algorithm not found: " << test_algorithm << "\n";
      return false;
    }
    return true;
  }

  virtual std::string name() const { return "Hand coded"; }
};



// ================= ADEPT ================================ 

class AdeptDifferentiator
  : public Differentiator {
public:
  AdeptDifferentiator(Timer& timer, const std::string& name_)
    : Differentiator(timer) { init_timer(name_); }

  virtual ~AdeptDifferentiator() { }

  virtual bool adjoint(TestAlgorithm test_algorithm,
		       const std::vector<Real>& x,
		       std::vector<Real>& y,
		       const std::vector<Real>& y_AD,
		       std::vector<Real>& x_AD) {
    if (x.size() != NX || y_AD.size() != NX) {
      throw differentiator_exception("One of input vectors not of size NX in call to AdeptDifferentiator::adjoint");
    }
    y.resize(NX);
    x_AD.resize(NX);

    std::vector<adept::aReal> q_init(NX);
    std::vector<adept::aReal> q(NX);

    adept::set_values(&q_init[0], NX, &x[0]);

    stack_.new_recording();
    func(test_algorithm, q_init, q);

    timer_.start(adjoint_compute_timer_id_);

    adept::set_gradients(&q[0], NX, &y_AD[0]);
    stack_.compute_adjoint();
    adept::set_gradients(&q_init[0], NX, &x_AD[0]);

    timer_.stop();

    return true;
  }


  virtual bool jacobian(TestAlgorithm test_algorithm,
			const std::vector<Real>& x,
			std::vector<Real>& y,
			std::vector<Real>& jac,
			int force_jacobian = 0) {
    if (x.size() != NX) {
      throw differentiator_exception("Input vector x not of size NX in call to AdeptDifferentiator::jacobian");
    }
    y.resize(NX);
    jac.resize(NX*NX);

    std::vector<adept::aReal> q_init(NX);
    std::vector<adept::aReal> q(NX);

    adept::set_values(&q_init[0], NX, &x[0]);

    stack_.new_recording();
    func(test_algorithm, q_init, q);

    stack_.independent(&q_init[0], NX);
    stack_.dependent(&q[0], NX);

    timer_.start(jacobian_timer_id_);
    if (force_jacobian > 0) {
      stack_.jacobian_forward(&jac[0]);
    }
    else if (force_jacobian < 0) {
      stack_.jacobian_reverse(&jac[0]);
    }
    else {
      stack_.jacobian(&jac[0]);
    }
    timer_.stop();
    return true;
  }

  virtual std::string name() const {
    std::stringstream name_;
    name_ << "Adept";
    int nthread = stack_.max_jacobian_threads();
    if (nthread > 1) {
      name_ << " (Jacobian using up to " << nthread << " OpenMP threads)";
    }
    else {
      name_ << " (single threaded)";
    }
    return name_.str(); 
  }

  virtual void no_openmp() { 
    stack_.set_max_jacobian_threads(1);
  }

  virtual void print() {
    std::cout << stack_;
  }

private:
  adept::Stack stack_;
};

 

#ifdef HAVE_ADOLC

// ================= ADOLC ================================ 

class AdolcDifferentiator
  : public Differentiator {
public:
  AdolcDifferentiator(Timer& timer, const std::string& name_)
    : Differentiator(timer), jac(0), I(0), result(0) { init_timer(name_); }

  virtual ~AdolcDifferentiator() {
    if (I) {
      myfreeI2(NX, I);
    }
    if (jac) {
      myfree2(jac);
    }
    if (result) {
      myfree1(result);
    }
  }

  virtual bool adjoint(TestAlgorithm test_algorithm,
		       const std::vector<Real>& x,
		       std::vector<Real>& y,
		       const std::vector<Real>& y_AD,
		       std::vector<Real>& x_AD) {
    if (x.size() != NX || y_AD.size() != NX) {
      throw differentiator_exception("One of input vectors not of size NX in call to AdolcDifferentiator::adjoint");
    }
    y.resize(NX);
    x_AD.resize(NX);

    std::vector<aReal> q_init(NX);
    std::vector<aReal> q(NX);

    trace_on(1,1);

    for (int i = 0; i < NX; i++) {
      q_init[i] <<= x[i];
    }

    func(test_algorithm, q_init, q);

    for (int i = 0; i < NX; i++) {
      q[i] >>= y[i];
    }

    trace_off();

    timer_.start(adjoint_compute_timer_id_);

    reverse(1, NX, NX, 0, const_cast<Real*>(&y_AD[0]), &x_AD[0]);                                                

    timer_.stop();
    return true;
  }


  virtual bool jacobian(TestAlgorithm test_algorithm,
			const std::vector<Real>& x,
			std::vector<Real>& y,
			std::vector<Real>& jac_,
			int force_jacobian = 0) {
    if (x.size() != NX) {
      throw differentiator_exception("Input vector x not of size NX in call to AdolcDifferentiator::jacobian");
    }
    y.resize(NX);
    jac_.resize(NX*NX);

    std::vector<aReal> q_init(NX);
    std::vector<aReal> q(NX);

    trace_on(1,1);

    for (int i = 0; i < NX; i++) {
      q_init[i] <<= x[i];
    }

    func(test_algorithm, q_init, q);

    for (int i = 0; i < NX; i++) {
      q[i] >>= y[i];
    }

    trace_off();

    if (!jac) {
      jac = myalloc2(NX,NX);
      I = myallocI2(NX);
      result = myalloc1(NX);
    }

    timer_.start(jacobian_timer_id_);

    if (force_jacobian < 0) {
      int rc = zos_forward(1, NX, NX, 1, &x[0], result);
      if (rc < 0) {
	throw differentiator_exception("Error occurred ADOL-C's zos_forward()");
      }
      MINDEC(rc,fov_reverse(1, NX, NX, NX, I, jac));
    }
    else if (force_jacobian > 0) {
      int rc = fov_forward(1, NX, NX, NX, &x[0], I, result, jac);
      if (rc < 0) {
	throw differentiator_exception("Error occurred ADOL-C's fov_forward()");
      }
    }
    else {
      ::jacobian(1, NX, NX, &x[0], jac);
    }

    timer_.stop();

    for (int j=0, index=0; j < NX; j++) {
      for (int i=0; i < NX; i++, index++) {
	jac_[index] = jac[i][j];
      }
    }
    return true;
  }

  virtual std::string name() const { return "ADOL-C"; }

private:
  Real** jac;
  Real** I;
  Real* result;
};

#endif // HAVE_ADOLC


#ifdef HAVE_CPPAD

// ================= CPPAD ================================ 

class CppadDifferentiator
  : public Differentiator {
public:
  typedef CppAD::AD<Real> aReal;

  CppadDifferentiator(Timer& timer, const std::string& name_)
    : Differentiator(timer) {
    init_timer(name_); 
    CppAD::thread_alloc::hold_memory(true);
  }
    
  virtual ~CppadDifferentiator() { }
  
  virtual bool adjoint(TestAlgorithm test_algorithm,
		       const std::vector<Real>& x,
		       std::vector<Real>& y,
		       const std::vector<Real>& y_AD,
		       std::vector<Real>& x_AD) {
    if (x.size() != NX || y_AD.size() != NX) {
      throw differentiator_exception("One of input vectors not of size NX in call to CppadDifferentiator::adjoint");
    }
    y.resize(NX);
    x_AD.resize(NX);

    std::vector<aReal> q_init(NX);
    std::vector<aReal> q(NX);

    for (int i = 0; i < NX; i++) {
      q_init[i] = x[i];
    }

    CppAD::Independent(q_init);

    func(test_algorithm, q_init, q);

    for (int i = 0; i < NX; i++) {
      y[i] = CppAD::Value(q[i]);
    }

    timer_.start(adjoint_prep_timer_id_);
    CppAD::ADFun<Real> f(q_init, q);

    timer_.start(adjoint_compute_timer_id_);
    x_AD = f.Reverse(1, y_AD);
    timer_.stop();

    return true;
  }

  virtual bool jacobian(TestAlgorithm test_algorithm,
			const std::vector<Real>& x,
			std::vector<Real>& y,
			std::vector<Real>& jac,
			int force_jacobian = 0) {
    if (x.size() != NX) {
      throw differentiator_exception("Input vector x not of size NX in call to CppadDifferentiator::jacobian");
    }
    y.resize(NX);
    jac.resize(NX*NX);
    jac_transpose_.resize(NX*NX);

    std::vector<aReal> q_init(NX);
    std::vector<aReal> q(NX);

    for (int i = 0; i < NX; i++) {
      q_init[i] = x[i];
    }

    CppAD::Independent(q_init);

    func(test_algorithm, q_init, q);

    for (int i = 0; i < NX; i++) {
      y[i] = CppAD::Value(q[i]);
    }

    timer_.start(adjoint_prep_timer_id_);
    CppAD::ADFun<Real> f(q_init, q);

    timer_.start(jacobian_timer_id_);

    if (force_jacobian < 0) {
      CppAD::JacobianRev(f, x, jac_transpose_);
    }
    else if (force_jacobian > 0) {
      CppAD::JacobianFor(f, x, jac_transpose_);
    } 
    else {
      jac_transpose_ = f.Jacobian(x);
    }

    // Transpose Jacobian because CppAD uses the opposite convention to the other tools
    Real (&jac_transpose2)[NX][NX]
      = *reinterpret_cast<Real(*)[NX][NX]>(&jac_transpose_[0]);
    for (int i = 0, index = 0; i < NX; i++) {
      for (int j = 0; j < NX; j++, index++) {
	jac[index] = jac_transpose2[j][i];
      }
    }

    return true;
  }

  virtual std::string name() const { return "CppAD"; }

private:
  std::vector<Real> jac_transpose_;
};

#endif // HAVE_CPPAD


#ifdef HAVE_SACADO

// ================= SACADO ================================ 

template<> int Sacado::Rad::ADmemblock<Real>::n_blocks = 0;

class SacadoDifferentiator
  : public Differentiator {
public:
  typedef Sacado::Rad::ADvar<Real> aReal;
  typedef Sacado::ELRFad::DFad<Real> aReal_fad;

  SacadoDifferentiator(Timer& timer, const std::string& name_)
    : Differentiator(timer) { init_timer(name_); }
    
  virtual ~SacadoDifferentiator() { }
  
  virtual bool adjoint(TestAlgorithm test_algorithm,
		       const std::vector<Real>& x,
		       std::vector<Real>& y,
		       const std::vector<Real>& y_AD,
		       std::vector<Real>& x_AD) {
    if (x.size() != NX || y_AD.size() != NX) {
      throw differentiator_exception("One of input vectors not of size NX in call to SacadoDifferentiator::adjoint");
    }
    y.resize(NX);
    x_AD.resize(NX);

    std::vector<aReal> q_init(NX);
    std::vector<aReal> q(NX);

    for (int i = 0; i < NX; i++) {
      q_init[i] = x[i];
    }

    func(test_algorithm, q_init, q);

    for (int i = 0; i < NX; i++) {
      y[i] = q[i].val();
    }

    timer_.start(base_timer_id_);
    aReal objective_func = 0.0;
    for (int i = 0; i < NX; i++) {
      objective_func += q[i] * y_AD[i];
    }

    timer_.start(adjoint_compute_timer_id_);
    Sacado::Rad::ADvar<Real>::Gradcomp();
    for (int i = 0; i < NX; i++) { 
      x_AD[i] = q_init[i].adj();
    }
    timer_.stop();

    return true;
  }  


  virtual bool jacobian(TestAlgorithm test_algorithm,
			const std::vector<Real>& x,
			std::vector<Real>& y,
			std::vector<Real>& jac,
			int force_jacobian = 0) {
    if (x.size() != NX) {
      throw differentiator_exception("Input vector x not of size NX in call to SacadoDifferentiator::jacobian");
    }
    y.resize(NX);
    jac.resize(NX*NX);

    std::vector<aReal_fad> q_init(NX);
    std::vector<aReal_fad> q(NX);

    for (int i = 0; i < NX; i++) {
      q_init[i] = x[i];
      q_init[i].resize(NX);
      q[i].resize(NX);
      q_init[i].fastAccessDx(i) = 1.0;
    }

    func(test_algorithm, q_init, q);

    for (int i = 0; i < NX; i++) {
      y[i] = q[i].val();
    }
            
    int index = 0;
    for (int i = 0; i < NX; i++) { 
      for (int k = 0; k < NX; k++, index++) {
	jac[index] = q[k].dx(i);
      }
    }
    return true;
  }

  virtual std::string name() const { return "Sacado (::Rad for adjoint, forward-mode only ::ELRFad for Jacobian)"; }
};

#endif // HAVE_SACADO




// The following enum is designed to be used in a "for" loop to loop
// through the available automatic differentiaion tools
enum AutoDiffTool {
  AUTODIFF_TOOL_ADEPT = 0
#ifdef HAVE_ADOLC
  , AUTODIFF_TOOL_ADOLC
#endif
#ifdef HAVE_CPPAD
  , AUTODIFF_TOOL_CPPAD
#endif
#ifdef HAVE_SACADO
  , AUTODIFF_TOOL_SACADO
#endif
  , N_AUTODIFF_TOOLS
};

const char* autodiff_tool_string[] = {
  "adept"
#ifdef HAVE_ADOLC
  , "adolc"
#endif
#ifdef HAVE_CPPAD
  , "cppad"
#endif
#ifdef HAVE_SACADO
  , "sacado"
#endif
};

const char* autodiff_tool_long_string[] = {
  "Adept"
#ifdef HAVE_ADOLC
  , "ADOL-C"
#endif
#ifdef HAVE_CPPAD
  , "CppAD"
#endif
#ifdef HAVE_SACADO
  , "Sacado"
#endif
};

inline
std::string
autodiff_tools()
{
  std::string tools = autodiff_tool_string[0];
  for (int i = 1; i < N_AUTODIFF_TOOLS; i++) {
    tools += ",";
    tools += autodiff_tool_string[i];
  }
  return tools;
}


// Return pointer to a virtual base object Differentiator
inline
Differentiator* 
new_differentiator(AutoDiffTool auto_diff_tool, Timer& timer, const std::string& name_)
{
  if (auto_diff_tool == AUTODIFF_TOOL_ADEPT) {
    return new AdeptDifferentiator(timer, name_);
  }
#ifdef HAVE_ADOLC
  else if (auto_diff_tool == AUTODIFF_TOOL_ADOLC) {
    return new AdolcDifferentiator(timer, name_);
  }
#endif
#ifdef HAVE_CPPAD
  else if (auto_diff_tool == AUTODIFF_TOOL_CPPAD) {
    return new CppadDifferentiator(timer, name_);
  }
#endif
#ifdef HAVE_SACADO
  else if (auto_diff_tool == AUTODIFF_TOOL_SACADO) {
    return new SacadoDifferentiator(timer, name_);
  }
#endif
  else {
    return 0;
  }
}
