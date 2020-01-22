/* adept_fortran.h -- Interoperability between Adept and Fortran-90 arrays

    Copyright (C) 2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   Fortran-90 introduced multi-dimensional arrays with essentially the
   same basic capabilities as passive Adept arrays, including the
   ability to index strided data in memory. The improved
   interoperability features of Fortran 2018 enable Fortran array data
   to be passed to and from C/C++. This header file enables passive
   Adept arrays to be passed to and from Fortran.

   PASSING ARRAYS FROM FORTRAN TO C++

   A C++ subroutine callable from Fortran could be declared in C++ as:

     extern "C"
     void adept_subroutine(adept::FortranArray* int_arr,
                           adept::FortranArray* dbl_arr);

   where FortranArray is a C++ class wrapping the CFI_cdesc_t type
   that contains the Fortran array descriptor. Within the definition
   of this function, Adept arrays may be associated with the Fortran
   data as follows:

     adept::intMatrix imat;
     adept::associate(imat, int_arr);
     imat >>= int_arr; // Alternative form

   In this example, the matrix of integers "imat" shares its data with
   the Fortran array int_arr. An exception will be thrown if the
   Fortran array is not of type integer and rank 2. Note that the
   array indexing of imat will be in the standard C/C++ convention,
   zero-based and with the final index varying fastest as memory is
   traversed. This is opposite to the way the array is accessed in
   Fortran.  The ">>=" provides a more succinct way to do the same
   thing.

   Consider the following: 

     adept::Matrix dmat;
     adept::associate(dmat, dbl_arr, true);

   Here, the third argument "true" indicates that the array strides of
   dmat are to be configured so that the array indices are the same as
   in Fortran (although still zero based). This will impede
   optimization of some array expressions using dmat, since the second
   dimension of dmat will not be contiguous in memory, and this is the
   dimension that Adept attempts to vectorize.

   PASSING ARRAYS FROM ADEPT TO FORTRAN

   A Fortran-implemented subroutine could be declared in C++ as
   follows:

     extern "C"
     void fort_subroutine(adept::FortranArray* int_arr,
                          adept::FortranArray* dbl_arr);

   To call this routine from C++, passing Adept arrays "imat" and
   "dmat" as the arguments, we can do simply:

     fort_subroutine(FortranArray(imat), FortranArray(dmat));

*/


#ifndef AdeptFortran_H
#define AdeptFortran_H 1

#include <complex>
#include <adept_arrays.h>

// Load the Fortran array interface into the global namespace
#include <ISO_Fortran_binding.h>

namespace adept {

  namespace internal {
    // Helper types such that cfi_type<X>::type returns the integer
    // type of "X", or fails to compile if it is not possible to send
    // an array of type X to Fortran
    template <typename Type> struct cfi_type
    { }; // Fails to compile if attempt to access "type"
    template <> struct cfi_type<char>
    { static const CFI_type_t type = CFI_type_signed_char; };
    template <> struct cfi_type<short>
    { static const CFI_type_t type = CFI_type_short; };
    template <> struct cfi_type<int>
    { static const CFI_type_t type = CFI_type_int; };
    template <> struct cfi_type<long>
    { static const CFI_type_t type = CFI_type_long; };
    template <> struct cfi_type<long long>
    { static const CFI_type_t type = CFI_type_long_long; };
    template <> struct cfi_type<bool>
    { static const CFI_type_t type = CFI_type_Bool; };
    template <> struct cfi_type<float>
    { static const CFI_type_t type = CFI_type_float; };
    template <> struct cfi_type<double>
    { static const CFI_type_t type = CFI_type_double; };
    template <> struct cfi_type<long double>
    { static const CFI_type_t type = CFI_type_long_double; };
    template <> struct cfi_type<std::complex<float> >
    { static const CFI_type_t type = CFI_type_float_Complex; };
    template <> struct cfi_type<std::complex<double> >
    { static const CFI_type_t type = CFI_type_long_double_Complex; };
    template <> struct cfi_type<std::complex<long double> >
    { static const CFI_type_t type = CFI_type_long_double_Complex; };
  }

  // This class is essentially a wrapper around the CFI_cdesc_t type
  // which stores a Fortran array descriptor which could be for an
  // array of any rank or type
  class FortranArray {

  protected:
    // Data: the Fortran array descriptor CFI_cdesc_t type, but the
    // version configured for the maximum allowable Fortran rank
    CFI_CDESC_T(CFI_MAX_RANK) ad;

  public:
    // This class either exists as a pointer to a Fortran array passed
    // in from a Fortran routine, or as an object pointing to an Adept
    // array that is about to be passed into a Fortran routine.
    // Therefore it can only be constructed from an existing Adept
    // array.
    FortranArray() = delete;
    
    // Initialize from Adept array. By default, the dimensions will
    // need to be accessed in opposite order in Fortran than in
    // C++/Adept, reflecting the default column-major array access of
    // the former and row-major array access of the latter. But by
    // providing preserve_dim_order=true, the dimension access order
    // will be preserved between the two.
    template <int Rank, typename Type>
    FortranArray(adept::Array<Rank,Type>& a,
		 bool preserve_dim_order = false) {
      init(a, preserve_dim_order);
    }
    // No way to ensure that Fortran cannot modify an array,
    // unfortunately, so we need to cast away the const-ness
    template <int Rank, typename Type>
    FortranArray(const adept::Array<Rank,Type>& a,
		 bool preserve_dim_order = false) {
      init(const_cast<adept::Array<Rank,Type>&>(a), preserve_dim_order);
    }

  protected:
    // Constructor implementation: initialize CFI_cdesc_t elements
    // from Adept array
    template <int Rank, typename Type>
    void init(adept::Array<Rank,Type>& a, bool preserve_dim_order) {
      ADEPT_STATIC_ASSERT(Rank <= CFI_MAX_RANK, ARRAY_RANK_EXCEEDS_FORTRAN_MAXIMUM);
      ad.base_addr = static_cast<void*>(a.data());
      ad.elem_len  = sizeof(Type);
      ad.version   = CFI_VERSION;
      ad.rank      = Rank;
      ad.attribute = CFI_attribute_other;
      ad.type      = internal::cfi_type<Type>::type;
      if (!preserve_dim_order) {
	for (int irank = 0; irank < Rank; ++irank) {
	  ad.dim[irank].lower_bound = 0;
	  ad.dim[irank].extent = a.dimension(Rank-irank-1);
	  ad.dim[irank].sm = a.offset(Rank-irank-1)*sizeof(Type);
	}
      }
      else {
	for (int irank = 0; irank < Rank; ++irank) {
	  ad.dim[irank].lower_bound = 0;
	  ad.dim[irank].extent = a.dimension(irank);
	  ad.dim[irank].sm = a.offset(irank)*sizeof(Type);
	}
      }
    }

  public:
    // Query the rank and type of the Fortran array
    int rank() const { return ad.rank; }
    int type_code() const { return ad.type; }

    // Return "true" if the rank or type equal the template parameters
    // Rank and Type
    template <int Rank>
    bool is_rank() const {
      return (Rank == ad.rank);
    }
    template <typename Type>
    bool is_type() const {
      return (internal::cfi_type<Type>::type == ad.type
	      && sizeof(Type) == ad.elem_len);
    }

    // Return the length or stride in memory of a particular dimension
    CFI_index_t dimension(int idim) const { return ad.dim[idim].extent; }
    CFI_index_t offset(int idim) const { return ad.dim[idim].sm/ad.elem_len; }
    
    // Throw an exception if the rank or type differ from the template
    // parameters Rank and Type
    template <int Rank, typename Type>
    void verify() const {
      if (!is_rank<Rank>()) {
	throw fortran_interoperability_error(
           "Rank of Fortran array does not match expected rank");
      }
      else if (!is_type<Type>()) {
	throw fortran_interoperability_error(
           "Type of Fortran array does not match expected type");
      }
    }

    // Return a pointer to the underlying data casting to the
    // specified Type
    template <typename Type>
    Type* data() {
      return static_cast<Type*>(ad.base_addr);
    }

    // Allow this object to be passed to a function expecting a
    // pointer
    operator CFI_cdesc_t*() { return reinterpret_cast<CFI_cdesc_t*>(&ad); }
    operator FortranArray*() { return this; }
    
  };

  // Associate Adept array "a" with Fortran array "fa" so that
  // subsequent changes to the elements of "a" will be seen within
  // Fortran when the C++ routine returns.
  template <int Rank, typename Type>
  void associate(Array<Rank,Type>& a, FortranArray* fa,
		 bool preserve_dim_order = false) {
    fa->verify<Rank,Type>(); // Verify rank and type
    ExpressionSize<Rank> dims, offs;
    if (!preserve_dim_order) {
      for (int irank = 0; irank < Rank; ++irank) {
	dims[Rank-irank-1] = fa->dimension(irank);
	offs[Rank-irank-1] = fa->offset(irank);
      }
    }
    else {
      for (int irank = 0; irank < Rank; ++irank) {
	dims[irank] = fa->dimension(irank);
	offs[irank] = fa->offset(irank);
      }
    }
    a.clear();
    a = Array<Rank,Type>(static_cast<Type*>(fa->data<Type>()), 0, dims, offs);
  }

  // Associate Adept array "a" with a general Fortran array descriptor
  // "cd", noting that we only verify that the rank and type match
  // when the "associate" function above is called.
  template <int Rank, typename Type>
  void associate(Array<Rank,Type>& a, CFI_cdesc_t* cd,
		 bool preserve_dim_order = false) {
    FortranArray* fa = reinterpret_cast<FortranArray*>(cd);
    associate(a, fa, preserve_dim_order);
  }

  // Enable link of an Adept array to a Fortran array using the >>=
  // operator
  template<int Rank, typename Type>
  void operator>>=(adept::Array<Rank,Type>& a, FortranArray* fa) {
    associate(a,fa);
  }
  template<int Rank, typename Type>
  void operator>>=(adept::Array<Rank,Type>& a, CFI_cdesc_t* cd) {
    associate(a,cd);
  }

} // End namespace adept

#endif
