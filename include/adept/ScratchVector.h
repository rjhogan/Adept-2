/* ScratchVector.h -- Class for holding temporary real data

    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   The ScratchVector class is used to store a temporary vector of real
   numbers (the type "Real") for use in optimally evaluating an
   expression and computing its derivative.  Certain parts of the
   expression need to store their numerical value when first computed
   since it will be needed again in the derivative computation.  In
   Adept 1.x such data were stored in the expression objects
   themselves, e.g. in adept::Multiply, but now that such an object
   represents array operations that might be computed in parallel, the
   storage for such scratch data must be held externally so that it
   can be different for each thread, and then be passed by reference
   to the evaluation member functions.

*/

#ifndef AdeptScratchVector_H
#define AdeptScratchVector_H

#include <adept/base.h>

namespace adept {

  namespace internal {

    // Definition of ScratchVector class
    template <int Size>
    class ScratchVector {
    public:
      // Constructors

      // By default no initialization is done
      ScratchVector() { }

      // Set all dimensions to the same value
      ScratchVector(Real x) {
	set_all(x);
      }

      // Specify the values of all elements
      ScratchVector(Real x[Size]) {
	for (int i = 0; i < Size; ++i) {
	  val[i] = x[i];
	}
      }

      // Assume copy constructor will copy elements of val
    
      // Set all to specified value
      void set_all(Real x) {
	for (int i = 0; i < Size; ++i) {
	  val[i] = x;
	}
      }

      // Copy from a ScratchVector object of the same rank
      void copy(const ScratchVector& d) {
	for (int i = 0; i < Size; ++i) {
	  val[i] = d[i];
	}
      }
      // ...or pointer to raw data
      void copy(const Real* d) {
	for (int i = 0; i < Size; ++i) {
	  val[i] = d[i];
	}
      }

      // Write out contents for debugging
      std::ostream& write(std::ostream& os) const {
	os << "{" << val[0];
	for (int i = 1; i < Size; i++) {
	  os << "," << val[i];
	}
	return os << "}\n";
      }

      // Const and non-const access to elements
      Real& operator[](int i) { return val[i]; }

      const Real& operator[](int i) const { return val[i]; }

      // Data
    private:
      Real val[Size];
    };
  
    // Specialization for scalars (zero-rank arrays) known at compile
    // time
    template <>
    class ScratchVector<0> {
    public:
      ScratchVector() { }
      ScratchVector(Real x) { }
      std::ostream& write(std::ostream& os) const {
	return os << "{}\n";
      }
    };

    // Write out all elements for debugging
    template <int Size>
    inline
    std::ostream& operator<<(std::ostream& os, const ScratchVector<Size>& s) {
      return s.write(os);
    }
   
 
  } // End namespace internal

} // End namespace adept

#endif // AdeptScratchVector_H
