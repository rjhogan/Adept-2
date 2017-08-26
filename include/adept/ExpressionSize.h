/* ExpressionSize.h -- Class for describing array sizes

    Copyright (C) 2014-2017 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   The ExpressionSize class is used to pass information between
   components of an expression on the dimensions (e.g. rows/columns,
   but works in any number of dimensions) of that part of an
   expression, and to check that the dimensions match.  Since
   ExpressionSize objects can be used to index arrays, they may be
   useful to users and so are not placed in the "adept::internal"
   namespace.

*/

#ifndef AdeptExpressionSize_H
#define AdeptExpressionSize_H

#include <string>
#include <sstream>

#include <adept/base.h>
#include <adept/traits.h>

namespace adept {

  // Definition of ExpressionSize class
  template <int Rank>
  class ExpressionSize {
  public:
    // Constructors
    ExpressionSize() { } // By default no initialization is done

    ExpressionSize(Index j) {
      if (j >= 0) {
	// Set all dimensions to the same value - usually 0 (empty
	// array) or 1 (scalar)
	set_all(j);
      }
      else {
	// Set just the first dimension to j; usually this would be
	// less than 0 to indicate an invalid expression
	dim[0] = j;
      }
    }

    ExpressionSize(Index j0, Index j1) { dim[0] = j0; dim[1] = j1; }

    // Assume copy constructor will copy elements of dim
    
    // An "invalid" expression is one involving a mismatch of array
    // sizes, and is conveyed by a negative first element
    bool invalid_expression() const { return (dim[0] < 0); }

    // Set all to specified value
    void set_all(Index j) {
      for (int i = 0; i < Rank; ++i) {
	dim[i] = j;
      }
    }

    // Copy from an ExpressionSize object of the same rank
    void copy(const ExpressionSize& d) {
      for (int i = 0; i < Rank; ++i) {
	dim[i] = d[i];
      }
    }
    // ...or pointer to raw data
    void copy(const Index* d) {
      for (int i = 0; i < Rank; ++i) {
	dim[i] = d[i];
      }
    }

    // Copy dissimilar ExpressionSize object, filling the remaining
    // dimensions with 1
    template <int MyRank>
    void copy_dissimilar(const ExpressionSize<MyRank>& d) {
      int rank = MyRank > Rank ? Rank : MyRank;
      for (int i = 0; i < rank; ++i) {
	dim[i] = d[i];
      }
      for (int i = rank; i < Rank; ++i) {
	dim[i] = 1;
      }
    }

    // String representation
    std::string str() const {
      std::stringstream s;
      s << "[" << dim[0];
      for (int i = 1; i < Rank; ++i) {
	s << "," << dim[i];
      }
      s << "]";
      return s.str();
    }

    // Get the total number of elements
    Index size() const {
      Index prod;
      if (Rank == 0) {
	prod = 1;
      }
      else {
	prod = dim[0];
	for (int i = 1; i < Rank; ++i) {
	  prod *= dim[i];
	}
      }
      return prod;
    }

    ExpressionSize& operator++() {
      for (int i = 0; i < Rank; ++i) {
	++dim[i];
      }
      return *this;
    }
    ExpressionSize& operator+=(Index inc) {
      for (int i = 0; i < Rank; ++i) {
	dim[i] += inc;
      }
      return *this;
    }


    bool operator==(const ExpressionSize<Rank>& rhs) const {
      for (int i = 0; i < Rank; i++) {
	if (dim[i] != rhs[i]) {
	  return false;
	}
      }
      return true;
    }
    bool operator!=(const ExpressionSize<Rank>& rhs) const {
      return !(*this == rhs);
    }

    // Const and non-const access to elements
    Index& operator[](int i) { return dim[i]; }
    const Index& operator[](int i) const { return dim[i]; }
  private:
    Index dim[Rank];
  };

  // Specialization for scalars (zero-rank arrays) known at compile
  // time
  template <>
  class ExpressionSize<0> {
  public:
    ExpressionSize() { }
    ExpressionSize(Index j) { }
    bool invalid_expression() const { return false; }
    std::string str() { return ""; }
    void set_all(Index) const { }
    bool operator==(const ExpressionSize<0>&) const { return true; }
    bool operator!=(const ExpressionSize<0>&) const { return false; }
    bool operator[](int) const { return 0; }
    template <int MyRank>
    void copy_dissimilar(const ExpressionSize<MyRank>&) { }
  };

  // Send the size of an expression to a stream
  template <int Rank>
  inline
  std::ostream& operator<<(std::ostream& os, const ExpressionSize<Rank>& s) {
    if (Rank > 0) {
      os << "(" << s[0];
      for (int i = 1; i < Rank; i++) {
	os << "," << s[i];
      }
      return os << ")";
    }
  }
 

  namespace internal {
    // The following are only used within the Adept library

    // Check whether the size of one expression is compatible with
    // that of another for arithmetic operations: this is "true" if
    // the rank is the same and the dimensions match, or if one of the
    // expressions is a scalar (zero rank).  If the ranks don't match
    // and neither is zero then the program won't compile.
    template <int LRank, int RRank>
    inline
    typename enable_if<LRank==RRank && (LRank>1), bool>::type
    compatible(const ExpressionSize<LRank>& l, const ExpressionSize<RRank>& r) {
      bool result = (l[0] == r[0]);
      for (int i = 1; i < RRank; ++i) {
	result = result && (l[i] == r[i]);
      }
      return result;
    }

    template <int LRank, int RRank>
    inline
    typename enable_if<LRank==1 && RRank==1, bool>::type
    compatible(const ExpressionSize<LRank>& l, const ExpressionSize<RRank>& r) {
      return l[0] == r[0];
    }

    template <int LRank, int RRank>
    inline
    typename enable_if<LRank==0 || RRank==0, bool>::type
    compatible(const ExpressionSize<LRank>& l, const ExpressionSize<RRank>& r) {
      return true;
    }

    // Return an ExpressionSize object of specified rank that expresses
    // an invalid expression
    template <int Rank>
    inline
    ExpressionSize<Rank> invalid_expression_size() {
      return ExpressionSize<Rank>(-1);
    }

  } // End namespace internal

  inline ExpressionSize<1> expression_size(Index j0)
  { return ExpressionSize<1>(j0); }

  inline ExpressionSize<2> expression_size(Index j0, Index j1)
  { return ExpressionSize<2>(j0, j1); }

} // End namespace adept

#endif // AdeptExpressionSize_H
