/* base.h -- Basic definitions 

    Copyright (C) 2012-2014 University of Reading
    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/


#ifndef AdeptBase_H
#define AdeptBase_H 1

#include <cstddef>


// ---------------------------------------------------------------------
// 0: Adept version number
// ---------------------------------------------------------------------

// The version of the Adept library is specified both as a string and
// an integer
#define ADEPT_VERSION      10909
#define ADEPT_VERSION_STR "1.9.9"


// ---------------------------------------------------------------------
// 1: Defines not requiring a library recompile
// ---------------------------------------------------------------------

// The following can either be changed here, or define them just
// before including this header file in your code, or define using the
// -Dxxx compiler option.  These options to not need the library to be
// recompiled.

// A globally accessible stack needs to be present for arithmetic
// statements to access; by default this is thread safe but if you
// know you are running a single-threaded application then slightly
// faster performance may be achieved by defining this. Note that in
// section 4 of this header file, ADEPT_STACK_THREAD_UNSAFE is
// explicitly defined on the Mac OS platform, since the executable
// format used typically does not support thread-local storage.
//#define ADEPT_STACK_THREAD_UNSAFE 1

// Define this to check whether the "multiplier" is zero before it is
// placed on the operation stack. This makes the forward pass slower
// and the reverse pass slightly faster, and is only worthwhile if
// many reverse passes will be carried out per forward pass (or if you
// have good reason to believe many variables in your code are zero).
// #define ADEPT_REMOVE_NULL_STATEMENTS 1

// If copy constructors for aReal objects are only used in the return
// values for functions then defining this will lead to slightly
// faster code, because it will be assumed that when a copy
// constructor is called the gradient_offset can simply be copied
// because the object being copied will shortly be destructed. You
// need to be sure that the code does not contain these constructions:
//   aReal x = y;
//   aReal x(y);
// where y is an aReal object.
//#define ADEPT_COPY_CONSTRUCTOR_ONLY_ON_RETURN_FROM_FUNCTION 1

// If using the same code for both forward-only and
// forward-and-reverse calculations, then it is useful to be able to
// dynamically control whether or not gradient information is computed
// by expressions in the forward pass using the pause_recording() and
// continue_recording() functions. To enable this feature uncomment
// the following, but note that it slows down the forward pass a
// little.  
//#define ADEPT_RECORDING_PAUSABLE 1

// Often when you first convert a code for automatic differentiation
// the gradients computed contain NaNs or infinities: uncommenting the
// following will check for these and throw an error when they are
// found, so that by running the program in a debugger and looking at
// the backtrace, you can locate the source.
//#define ADEPT_TRACK_NON_FINITE_GRADIENTS 1

// If this is defined then each mathematical operation does not
// involve a check whether more memory needs to be allocated; rather
// the user first specifies how much memory to allocate to hold the
// entire algorithm via the preallocate_statements and
// preallocate_operations functions. This is a little faster, but is
// obviously risky if you don't anticipate correctly how much memory
// will be needed.
//#define ADEPT_MANUAL_MEMORY_ALLOCATION 1

// Do we check array bounds when indexing arrays?
//#define ADEPT_BOUNDS_CHECKING 1


// The initial size of the stacks, which can be grown if required
#ifndef ADEPT_INITIAL_STACK_LENGTH
#define ADEPT_INITIAL_STACK_LENGTH 1000
#endif


// The statement and operation stacks
#ifndef ADEPT_STACK_BLOCK_LENGTH
#define ADEPT_STACK_BLOCK_LENGTH 1048576
#endif

//#define ADEPT_SUPPORT_HUGE_ARRAYS 1




// ---------------------------------------------------------------------
// 2: Defines requiring a library recompile
// ---------------------------------------------------------------------

// The "stack" containing derivative information can be implemented in
// two ways: if ADEPT_STACK_STORAGE_STL is defined then C++ STL
// containers are used, otherwise dynamically allocated arrays are
// used.  Experience says that dynamically allocated arrays are faster.
//#define ADEPT_STACK_STORAGE_STL 1

// The number of rows/columns of a Jacobian that are calculated at
// once. The optimum value depends on platform, the size of your
// Jacobian and the number of OpenMP threads available.
#ifndef ADEPT_MULTIPASS_SIZE
//#define ADEPT_MULTIPASS_SIZE 1
//#define ADEPT_MULTIPASS_SIZE 2
#define ADEPT_MULTIPASS_SIZE 4
//#define ADEPT_MULTIPASS_SIZE 8
//#define ADEPT_MULTIPASS_SIZE 15
//#define ADEPT_MULTIPASS_SIZE 16
//#define ADEPT_MULTIPASS_SIZE 32
//#define ADEPT_MULTIPASS_SIZE 64
#endif

// If ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK then the
// Jacobian calculation will try to remove redundant loops involving
// zeros; note that this may inhibit auto-vectorization
#define ADEPT_MULTIPASS_SIZE_ZERO_CHECK 64
#define PACKET_SIZE_ZERO_CHECK 64

// By default the precision of differentiated expressions is "double".
// To override this, define ADEPT_FLOATING_POINT_TYPE to the type
// required.
//#define ADEPT_FLOATING_POINT_TYPE float
//#define ADEPT_FLOATING_POINT_TYPE long double

// Thread-local storage is used for the global Stack pointer to ensure
// thread safety.  Thread-local variables are declared in different
// ways by different compilers, the most common ones being detected in
// section 4 below.  Some platforms (particularly some Mac platforms)
// do not implement thread-local storage, and therefore on Mac
// thread-local storage is disabled. If you want to manually specify
// how thread-local storage is declared, you may do it here,
// e.g. using the C++11 keyword "thread_local".  If thread-local
// storage is not available on your platform but is not detected in
// section 4, and consequently you cannot get the code to compile,
// then you can make an empty declaration here.
//#define ADEPT_THREAD_LOCAL thread_local

// Define the following if you wish to use OpenMP to accelerate array
// expressions
//#define ADEPT_OPENMP_ARRAY_OPERATIONS 1

// Do we disable automatic alias checking in array operations?
//#define ADEPT_NO_ALIAS_CHECKING


// This cannot be changed without rewriting the Adept library
#define ADEPT_MAX_ARRAY_DIMENSIONS 7

// ---------------------------------------------------------------------
// 4: Miscellaneous
// ---------------------------------------------------------------------

// The following attempt to align the data to facilitate SSE2
// vectorization did not work so is disabled
#ifdef __GNUC__
//#define ADEPT_SSE2_ALIGNED __attribute__ ((aligned (16)))
#define ADEPT_SSE2_ALIGNED
#else
#define ADEPT_SSE2_ALIGNED
#endif

// The way thread-local variables are specified pre-C++11 is compiler
// specific.  You can specify this manually by defining the
// ADEPT_THREAD_LOCAL preprocessor variable (e.g. to "thread_local");
// otherwise it is defined here depending on your compiler
#ifndef ADEPT_THREAD_LOCAL
#if defined(__APPLE__)
// Thread-local storage typically does not work on Mac OS X so we turn
// it off and provide a blank definition of ADEPT_THREAD_LOCAL
#define ADEPT_STACK_THREAD_UNSAFE 1
#define ADEPT_THREAD_LOCAL
#elif defined(_WIN32)
// Windows has a different way to specify thread-local storage from
// the GCC/Intel/Sun/IBM compilers.  Note that this is unified with
// C++11 but the older formats are still supported and it would be
// more complicated to check for C++11 support.
#define ADEPT_THREAD_LOCAL __declspec(thread)
#else
// The following should work on GCC/Intel/Sun/IBM compilers
#define ADEPT_THREAD_LOCAL __thread
#endif
#endif

// If we use OpenMP to parallelize array expressions then some
// variables local to active operation structures (Multiply etc) need
// to be made thread-local
#ifdef ADEPT_OPENMP_ARRAY_OPERATIONS
#define ADEPT_THREAD_LOCAL_IF_OPENMP ADEPT_THREAD_LOCAL
#else
#define ADEPT_THREAD_LOCAL_IF_OPENMP
#endif

// Various C++11 features
#if __cplusplus > 199711L
// We can optimize the returning of Arrays from functions with move
// semantics
#define ADEPT_MOVE_SEMANTICS 1
#define ADEPT_CXX11_FEATURES 1
#endif


// ---------------------------------------------------------------------
// 5: Define basic floating-point and integer types
// ---------------------------------------------------------------------
namespace adept {

  // By default everything is double precision, but this precision can
  // be changed by defining ADEPT_FLOATING_POINT_TYPE
#ifdef ADEPT_FLOATING_POINT_TYPE
#undef ADEPT_FLOATING_POINT_TYPE
#error ADEPT_FLOATING_POINT_TYPE is deprecated: use ADEPT_REAL_TYPE_SIZE instead
#endif

#ifndef ADEPT_REAL_TYPE_SIZE
#define ADEPT_REAL_TYPE_SIZE 8
#endif

#if ADEPT_REAL_TYPE_SIZE == 4
  typedef float Real;
#elif ADEPT_REAL_TYPE_SIZE == 8
  typedef double Real;
#elif ADEPT_REAL_TYPE_SIZE == 16
  typedef long double Real;
#else
#undef ADEPT_REAL_TYPE_SIZE
#error If defined, ADEPT_REAL_TYPE_SIZE must be 4 (float), 8 (double) or 16 (long double)
#endif

  // By default sizes of arrays, indices to them, and indices in the
  // automatic differentiation stack are stored as 4-byte integers,
  // but for very large arrays and algorithms, larger types may be
  // needed.  Remember that on 32-bit platforms this will have no
  // effect.
#ifdef ADEPT_SUPPORT_HUGE_ARRAYS
  typedef std::size_t  uIndex; // Unsigned
  typedef std::ptrdiff_t Index;  // Signed
#else
  //  typedef unsigned int uIndex;
  typedef int uIndex;
  typedef int Index;
#endif

} // End namespace adept

#endif
