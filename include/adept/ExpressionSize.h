/* ExpressionSize.h -- Class for describing array sizes

    Copyright (C) 2014- European Centre for Medium-Range Weather Forecasts

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

#ifdef ADEPT_CXX11_FEATURES
#include <array>
#endif

namespace adept {

  // In principle ExpressionSize could be replaced by std::array, but
  // this means RankType must be unsigned std::size_t which causes
  // problems as rank is assumed to be signed in many places.
  
  //  template <std::size_t Rank>
  //  using ExpressionSize = std::array<Index, Rank>;


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
	fill(j);
      }
      else {
	// Set just the first dimension to j; usually this would be
	// less than 0 to indicate an invalid expression
	dim[0] = j;
      }
    }

    ExpressionSize(Index j0, Index j1)
    { dim[0]=j0; dim[1]=j1; }
    ExpressionSize(Index j0, Index j1, Index j2)
    { dim[0]=j0; dim[1]=j1; dim[2]=j2; }
    ExpressionSize(Index j0, Index j1, Index j2, Index j3)
    { dim[0]=j0; dim[1]=j1; dim[2]=j2; dim[3]=j3; }
    ExpressionSize(Index j0, Index j1, Index j2, Index j3, Index j4)
    { dim[0]=j0; dim[1]=j1; dim[2]=j2; dim[3]=j3; dim[4]=j4; }
    ExpressionSize(Index j0, Index j1, Index j2, Index j3, Index j4, Index j5)
    { dim[0]=j0; dim[1]=j1; dim[2]=j2; dim[3]=j3; dim[4]=j4; dim[5]=j5; }
    ExpressionSize(Index j0, Index j1, Index j2, Index j3, Index j4, Index j5, Index j6)
    { dim[0]=j0; dim[1]=j1; dim[2]=j2; dim[3]=j3; dim[4]=j4; dim[5]=j5; dim[6]=j6; }

    // Assume copy constructor will copy elements of dim
    
    // An "invalid" expression is one involving a mismatch of array
    // sizes, and is conveyed by a negative first element
    bool invalid_expression() const { return (dim[0] < 0); }

    // Set all to specified value (as std::array)
    void fill(Index j) {
      for (int i = 0; i < Rank; ++i) {
	dim[i] = j;
      }
    }

    // Copy from pointer to raw data; note that copying from an
    // ExpressionSize object is no longer supported as we have the
    // copy assignment operator (=) for that
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

    /*
    // String representation; in case a future versions of Adept
    // replaces ExpressionSize with std::array, this is now a free
    // function internal::str
    std::string str() const {
      std::stringstream s;
      s << "[" << dim[0];
      for (int i = 1; i < Rank; ++i) {
	s << "," << dim[i];
      }
      s << "]";
      return s.str();
    }
    */

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

#ifdef ADEPT_CXX11_FEATURES
    /*
    ExpressionSize& operator=(const std::array<Index,Rank>& arr) {
      for (int i = 0; i < Rank; ++i) {
	dim[i] = arr[i];
      }
      return *this;
    }
    */
#endif
    
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

#ifdef ADEPT_MOVE_SEMANTICS
    friend void swap(ExpressionSize<Rank>& l, 
		     ExpressionSize<Rank>& r) noexcept {
      for (int i = 0; i < Rank; ++i) {
	Index tmp = l.dim[i];
	l.dim[i] = r.dim[i];
	r.dim[i] = tmp;
      }
    }
#endif

    // Const and non-const access to elements
    Index& operator[](int i) { return dim[i]; }
    const Index& operator[](int i) const { return dim[i]; }
  private:
    static const std::size_t Rank_uint = Rank;
    union {
      Index dim[Rank];
      std::array<Index,Rank_uint> dim_as_std_array;
    };
  };

  // Specialization for scalars (zero-rank arrays) known at compile
  // time
  template <>
  class ExpressionSize<0> {
  public:
    ExpressionSize() { }
    ExpressionSize(Index j) { }
    bool invalid_expression() const { return false; }
    //    std::string str() const { return ""; }
    void fill(Index) const { }
    bool operator==(const ExpressionSize<0>&) const { return true; }
    bool operator!=(const ExpressionSize<0>&) const { return false; }
    bool operator[](int) const { return 0; }
    template <int MyRank>
    void copy_dissimilar(const ExpressionSize<MyRank>&) { }
  };


  // Send the size of an expression to a stream
  template <RankType Rank>
  inline
  std::ostream& operator<<(std::ostream& os, const ExpressionSize<Rank>& s) {
    if (Rank > 0) {
      os << "(" << s[0];
      for (RankType i = 1; i < Rank; i++) {
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
    template <RankType LRank, RankType RRank>
    inline
    typename enable_if<LRank==RRank && (LRank>1), bool>::type
    compatible(const ExpressionSize<LRank>& l, const ExpressionSize<RRank>& r) {
      bool result = (l[0] == r[0]);
      for (RankType i = 1; i < RRank; ++i) {
	result = result && (l[i] == r[i]);
      }
      return result;
    }

    template <RankType LRank, RankType RRank>
    inline
    typename enable_if<LRank==1 && RRank==1, bool>::type
    compatible(const ExpressionSize<LRank>& l, const ExpressionSize<RRank>& r) {
      return l[0] == r[0];
    }

    template <RankType LRank, RankType RRank>
    inline
    typename enable_if<LRank==0 || RRank==0, bool>::type
    compatible(const ExpressionSize<LRank>& l, const ExpressionSize<RRank>& r) {
      return true;
    }

    // Return an ExpressionSize object of specified rank that expresses
    // an invalid expression
    template <RankType Rank>
    inline
    ExpressionSize<Rank> invalid_expression_size() {
      return ExpressionSize<Rank>(-1);
    }

    template <RankType Rank>
    std::string str(ExpressionSize<Rank> dim) {
      std::stringstream s;
      s << "[" << dim[0];
      for (RankType i = 1; i < Rank; ++i) {
	s << "," << dim[i];
      }
      s << "]";
      return s.str();
    }

  } // End namespace internal

 
  // Deprecated
  inline ExpressionSize<1> expression_size(Index j0)
  { return ExpressionSize<1>(j0); }
  inline ExpressionSize<2> expression_size(Index j0, Index j1)
  { return ExpressionSize<2>(j0, j1); }

  // Use this instead
  inline ExpressionSize<1> dimensions(Index j0)
  { return ExpressionSize<1>(j0); }
  inline ExpressionSize<2> dimensions(Index j0, Index j1)
  { return ExpressionSize<2>(j0, j1); }
  inline ExpressionSize<3> dimensions(Index j0, Index j1, Index j2)
  { return ExpressionSize<3>(j0, j1, j2); }
  inline ExpressionSize<4> dimensions(Index j0, Index j1, Index j2,
				      Index j3)
  { return ExpressionSize<4>(j0, j1, j2, j3); }
  inline ExpressionSize<5> dimensions(Index j0, Index j1, Index j2,
				      Index j3, Index j4)
  { return ExpressionSize<5>(j0, j1, j2, j3, j4); }
  inline ExpressionSize<6> dimensions(Index j0, Index j1, Index j2,
				      Index j3, Index j4, Index j5)
  { return ExpressionSize<6>(j0, j1, j2, j3, j4, j5); }
  inline ExpressionSize<7> dimensions(Index j0, Index j1, Index j2,
				      Index j3, Index j4, Index j5, Index j6)
  { return ExpressionSize<7>(j0, j1, j2, j3, j4, j5, j6); }

  /*
  // This is how we could do the above with later versions of C++
  template <typename... T>
  std::array<Index,sizeof...(T)> dimensions(T... j) {
    return std::array{j...};
  }
  */

} // End namespace adept

#endif // AdeptExpressionSize_H
