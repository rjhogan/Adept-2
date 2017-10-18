/* exception.h -- Exceptions thrown by Adept library

    Copyright (C) 2012-2014 University of Reading
    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   Adept functions can throw exceptions that are all derived either
   from the adept::autodiff_exception or adept::array_exception types,
   themselves inherited from the adept::exception type.  All implement
   the "what()" function to return an error message.

*/

#ifndef AdeptException_H
#define AdeptException_H 1

#include <exception>
#include <string>
#include <sstream>


namespace adept {

  // -------------------------------------------------------------------
  // adept::exception class from which all others are derived
  // -------------------------------------------------------------------
  class exception : public std::exception {
  public:
    virtual const char* what() const throw() { return message_.c_str(); }
    virtual ~exception() throw() { }
  protected:
    std::string message_;
  };

  class feature_not_available : public adept::exception {
  public:
    feature_not_available(const std::string& message = "Feature not available")
    { message_ = message; }
  };

  // -------------------------------------------------------------------
  // autodiff_exception and child classes
  // -------------------------------------------------------------------

  // The autodiff_exception type is only used as a base for more
  // specific exceptions
  class autodiff_exception : public adept::exception { };

  // Now we define the various specific autodiff exceptions that can
  // be thrown.
  class gradient_out_of_range : public autodiff_exception {
  public:
    gradient_out_of_range(const std::string& message 
	  = "Gradient index out of range: probably aReal objects have been created after a set_gradient(s) call")
    { message_ = message; }
  };

  class gradients_not_initialized : public autodiff_exception {
  public:
    gradients_not_initialized(const std::string& message 
	      = "Gradients not initialized: at least one call to set_gradient(s) is needed before a forward or reverse pass")
    { message_ = message; }
  };

  class stack_already_active : public autodiff_exception {
  public:
    stack_already_active(const std::string& message 
	 = "Attempt to activate an adept::Stack when one is already active in this thread")
    { message_ = message; }
  };

  class dependents_or_independents_not_identified : public autodiff_exception {
  public:
    dependents_or_independents_not_identified(const std::string& message 
	 = "Dependent or independent variables not identified before a Jacobian computation")
    { message_ = message; }
  };

  class wrong_gradient : public autodiff_exception {
  public:
    wrong_gradient(const std::string& message
	  = "Wrong gradient: append_derivative_dependence called on a different aReal object from the most recent add_derivative_dependence call")
    { message_ = message; }
  };

  class non_finite_gradient : public autodiff_exception {
  public:
    non_finite_gradient(const std::string& message
	= "A non-finite gradient has been computed")
    { message_ = message; }
  };


  // -------------------------------------------------------------------
  // array_exception and child classes
  // -------------------------------------------------------------------

  // The array_exception type
  class array_exception : public adept::exception { 
  public:
    array_exception(const std::string& message
		    = "A misuse of arrays occurred")
    { message_ = message; }
  };

  class size_mismatch : public array_exception {
  public:
    size_mismatch(const std::string& message
		  = "Array sizes do not match in array expression")
    { message_ = message; }
  };

  class inner_dimension_mismatch : public array_exception {
  public:
    inner_dimension_mismatch(const std::string& message
	  = "Inner dimensions don't agree in matrix multiplication")
    { message_ = message; }
  };

  class empty_array : public array_exception {
  public:
    empty_array(const std::string& message
	= "Use of empty array where non-empty array required")
    { message_ = message; }
  };

  class invalid_dimension : public array_exception {
  public:
    invalid_dimension(const std::string& message
	= "Attempt to create array with invalid dimension")
    { message_ = message; }
  };

  class index_out_of_bounds : public array_exception {
  public:
    index_out_of_bounds(const std::string& message
	= "Array index is out of bounds")
    { message_ = message; }
  };

  class invalid_operation : public array_exception {
  public:
    invalid_operation(const std::string& message
      = "Operation not permitted for this type of array")
    { message_ = message; }
  };

  class matrix_ill_conditioned : public array_exception {
  public:
    matrix_ill_conditioned(const std::string& message
      = "Matrix ill conditioned")
    { message_ = message; }
  };


  // -------------------------------------------------------------------
  // Provide location of where exception was thrown
  // -------------------------------------------------------------------

  // The following enables the file name and line number to be reported
  // with something like 
  //   throw array_exception("Bad matrix" ADEPT_EXCEPTION_LOCATION)
#define ADEPT_EXCEPTION_LOCATION \
  +adept::internal::exception_location(__FILE__,__LINE__)

  // A string with location information to append to the error message
  namespace internal {
    inline
    std::string exception_location(const char* file, int line) {
      std::stringstream s;
      s << " (in " << file << ":" << line << ")";
      return s.str();      
    }
  }

} // End namespace adept

#endif
