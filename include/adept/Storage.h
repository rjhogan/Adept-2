/* Storage.h -- store array of active or inactive data

    Copyright (C) 2012-2014 University of Reading
    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.


   The Storage class manages the data underlying array objects, and
   uses a model of reference counting so that multiple objects can
   refer to the same data.  This enables arrays that are actually
   subsets of another array to be treated as normal array objects.

*/

#ifndef AdeptStorage_H
#define AdeptStorage_H 1

#include <string>
#include <sstream>
#include <new>
#ifdef __unix__
// Define _POSIX_VERSION
#include <unistd.h>
#endif
#include <stdlib.h>

// Windows needs _aligned_malloc
#ifdef _MSC_VER
#include <malloc.h>
#endif

#include <adept/exception.h>
#include <adept/base.h>
#include <adept/Stack.h>
#include <adept/Packet.h>

namespace adept {

  // -------------------------------------------------------------------
  // Global variables
  // -------------------------------------------------------------------
  namespace internal {
    // To check for memory leaks, we keep a running total of the number
    // of Storage objects that are created and destroyed
    extern Index n_storage_objects_created_;
    extern Index n_storage_objects_deleted_;

    // -------------------------------------------------------------------
    // Aligned allocation and freeing of memory
    // -------------------------------------------------------------------
    template <typename Type>
    inline
    Type* alloc_aligned(Index n) {
      int n_align = Packet<Type>::alignment_bytes;
      if (n_align <= 1) {
	return new Type[n];
      }
      else {
	int status;
	Type* result;
#ifdef _POSIX_VERSION
#if _POSIX_VERSION >= 200112L
	if (posix_memalign(reinterpret_cast<void**>(&result), n_align, n*sizeof(Type)) != 0) {
	  throw std::bad_alloc();
	}
#else
	result = new Type[n];
#endif
#elif defined(_MSC_VER)
	result = reinterpret_cast<Type*>(_aligned_malloc(n*sizeof(Type), n_align));
	if (result == 0) {
	  throw std::bad_alloc();
	}
#else
	result = new Type[n];	
#endif
      return result;
      }
    }
    
    template <typename Type>
    inline
    void free_aligned(Type* data) {
      if (Packet<Type>::alignment_bytes <= 1) {
	delete[] data;
      }
      else { 
#ifdef _POSIX_VERSION
#if _POSIX_VERSION >= 200112L   
	free(data);
#else
	delete[] data;
#endif
#elif defined(_MSC_VER)
	_aligned_free(ptr);
#endif
      }
    }

  }

  // -------------------------------------------------------------------
  // Definition of Storage class
  // -------------------------------------------------------------------
  template <typename Type>
  class Storage {
  public:
    // -------------------------------------------------------------------
    // Storage: 1. Constructors and destructor
    // -------------------------------------------------------------------

    // The only way to construct this object is by passing it an
    // integer indicating the size, and optionally for active objects,
    // an integer representing the index to the gradients stored in
    // the stack.
    Storage(Index n, bool IsActive = false)
      : n_(n), n_links_(1), gradient_index_(-1) {
      data_ = internal::alloc_aligned<Type>(n);
      internal::n_storage_objects_created_++; 
      if (IsActive) {
	gradient_index_ = ADEPT_ACTIVE_STACK->register_gradients(n);
      }
    }
    
  protected:
    // Only allow the class to destroy itself by putting in "protected"
    ~Storage() {
      internal::free_aligned(data_);
      if (gradient_index_ >= 0) {
	ADEPT_ACTIVE_STACK->unregister_gradients(gradient_index_, n_);
      }
      internal::n_storage_objects_deleted_++; }


    // Null initialization, copy and assignment methods that are
    // "protected" to prevent them being used
    Storage() { }
    Storage(Storage& storage) { };
    void operator=(Storage& storage) { };


    // -------------------------------------------------------------------
    // Storage: 2. Public member functions
    // -------------------------------------------------------------------  
  public:
    // Add link to an existing storage object
    void add_link() const
    { n_links_++; } 
    
    // Remove link as follows
    void remove_link() const {
      if (n_links_ == 0) {
	throw invalid_operation("Attempt to remove more links to a storage object than set"
				ADEPT_EXCEPTION_LOCATION);
      }
      else if (--n_links_ == 0) {
	delete this;
      }
    }

    // Return the number of elements allocated
    Index n_allocated() const
    { return n_; }

    // Return the number of links to an object
    int n_links() const
    { return n_links_; }

    Index gradient_index() const
    { return gradient_index_; }

    // Return pointer to the start of the data
    Type*
    data()
    { return data_; }
    const Type*
    data() const
    { return data_; }

    // Return a string of information
    std::string
    info_string() const {
      std::stringstream x;
      x << n_ << " " << sizeof(Type) << "-byte elements allocated with "
	<< n_links_ << " links";
      return x.str();
    }

    // -------------------------------------------------------------------
    // Storage: 3. Data
    // -------------------------------------------------------------------  
  private:
    // Pointer to the start of the data
    Type* data_;
    // Number of elements allocated
    Index n_;
    // Number of links to the storage object allowing for arrays and
    // array slices to point to the same data. If this falls to zero
    // the Storage object will destruct itself
    mutable int n_links_;
    // For active variables, this s the gradient index of the first
    // element.  It would be better to only store this if Type is
    // floating point.
    Index gradient_index_;

  }; // End of Storage class
  

  // -------------------------------------------------------------------
  // Helper functions
  // -------------------------------------------------------------------
  inline Index n_storage_objects()
  { return internal::n_storage_objects_created_
      - internal::n_storage_objects_deleted_; }

  inline Index n_storage_objects_created()
  { return internal::n_storage_objects_created_; }
  
  inline Index n_storage_objects_deleted()
  { return internal::n_storage_objects_deleted_; }
  
} // End namespace adept

#endif
