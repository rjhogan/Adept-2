/* jacobian.cpp -- Computation of Jacobian matrix

    Copyright (C) 2012-2014 University of Reading
    Copyright (C) 2015-2016 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifdef _OPENMP
#include <omp.h>
#endif

#include "adept/Stack.h"
#include "adept/Packet.h"
#include "adept/traits.h"

namespace adept {

  namespace internal {
    static const int MULTIPASS_SIZE = ADEPT_REAL_PACKET_SIZE == 1 ? ADEPT_MULTIPASS_SIZE : ADEPT_REAL_PACKET_SIZE;
  }

  using namespace internal;

  template <typename T>
  T _check_long_double() {
    // The user may have requested Real to be of type "long double" by
    // specifying ADEPT_REAL_TYPE_SIZE=16. If the present system can
    // only support double then sizeof(long double) will be 8, but
    // Adept will not be emitting the best code for this, so it is
    // probably better to fail forcing the user to specify
    // ADEPT_REAL_TYPE_SIZE=8.
    ADEPT_STATIC_ASSERT(ADEPT_REAL_TYPE_SIZE != 16 || ADEPT_REAL_TYPE_SIZE == sizeof(Real),
			COMPILER_DOES_NOT_SUPPORT_16_BYTE_LONG_DOUBLE);
    return 1;
  }

  /*
  void
  Stack::jacobian_forward_kernel(Real* gradient_multipass_b) const
  {
    static const int MULTIPASS_SIZE = Packet<Real>::size;

    // Loop forward through the derivative statements
    for (uIndex ist = 1; ist < n_statements_; ist++) {
      const Statement& statement = statement_[ist];
      // We copy the LHS to "a" in case it appears on the RHS in any
      // of the following statements
      Block<MULTIPASS_SIZE,Real> a; // Initialized to zero automatically
      
      // Loop through operations
      for (uIndex iop = statement_[ist-1].end_plus_one;
	   iop < statement.end_plus_one; iop++) {
	Real* __restrict grad = gradient_multipass_b+index_[iop]*MULTIPASS_SIZE;
	// Loop through columns within this block; we hope the
	// compiler can optimize this loop. Note that it is faster
	// to always use MULTIPASS_SIZE, always known at
	// compile time, than to use block_size, which is not, even
	// though in the last iteration this may involve redundant
	// computations.
	if (multiplier_[iop] == 1.0) {
	  //	    if (__builtin_expect(multiplier_[iop] == 1.0,0)) {
	  for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	    //	      for (uIndex i = 0; i < block_size; i++) {
	    a[i] += grad[i];
	  }
	}
	else {
	  for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	    //	      for (uIndex i = 0; i < block_size; i++) {
	    a[i] += multiplier_[iop]*grad[i];
	  }
	}
      }
      // Copy the results
      for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	gradient_multipass_b[statement.index*MULTIPASS_SIZE+i] = a[i];
      }
    } // End of loop over statements
  }    
  */

#if ADEPT_REAL_PACKET_SIZE > 1
  void
  Stack::jacobian_forward_kernel(Real* __restrict gradient_multipass_b) const
  {

    // Loop forward through the derivative statements
    for (uIndex ist = 1; ist < n_statements_; ist++) {
      const Statement& statement = statement_[ist];
      // We copy the LHS to "a" in case it appears on the RHS in any
      // of the following statements
      Packet<Real> a; // Zeroed automatically
      // Loop through operations
      for (uIndex iop = statement_[ist-1].end_plus_one;
	   iop < statement.end_plus_one; iop++) {
	Packet<Real> g(gradient_multipass_b+index_[iop]*MULTIPASS_SIZE);
	Packet<Real> m(multiplier_[iop]);
	a += m * g;
      }
      // Copy the results
      a.put(gradient_multipass_b+statement.index*MULTIPASS_SIZE);
    } // End of loop over statements
  }    
#else
  void
  Stack::jacobian_forward_kernel(Real* __restrict gradient_multipass_b) const
  {

    // Loop forward through the derivative statements
    for (uIndex ist = 1; ist < n_statements_; ist++) {
      const Statement& statement = statement_[ist];
      // We copy the LHS to "a" in case it appears on the RHS in any
      // of the following statements
      Block<MULTIPASS_SIZE,Real> a; // Zeroed automatically
      // Loop through operations
      for (uIndex iop = statement_[ist-1].end_plus_one;
	   iop < statement.end_plus_one; iop++) {
	for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	  a[i] += multiplier_[iop]*gradient_multipass_b[index_[iop]*MULTIPASS_SIZE+i];
	}
      }
      // Copy the results
      for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	gradient_multipass_b[statement.index*MULTIPASS_SIZE+i] = a[i];
      }
    } // End of loop over statements
  }    
#endif

  void
  Stack::jacobian_forward_kernel_extra(Real* __restrict gradient_multipass_b,
				       uIndex n_extra) const
  {

    // Loop forward through the derivative statements
    for (uIndex ist = 1; ist < n_statements_; ist++) {
      const Statement& statement = statement_[ist];
      // We copy the LHS to "a" in case it appears on the RHS in any
      // of the following statements
      Block<MULTIPASS_SIZE,Real> a; // Zeroed automatically
      // Loop through operations
      for (uIndex iop = statement_[ist-1].end_plus_one;
	   iop < statement.end_plus_one; iop++) {
	for (uIndex i = 0; i < n_extra; i++) {
	  a[i] += multiplier_[iop]*gradient_multipass_b[index_[iop]*MULTIPASS_SIZE+i];
	}
      }
      // Copy the results
      for (uIndex i = 0; i < n_extra; i++) {
	gradient_multipass_b[statement.index*MULTIPASS_SIZE+i] = a[i];
      }
    } // End of loop over statements
  }    



  // Compute the Jacobian matrix, parallelized using OpenMP. Normally
  // the user would call the jacobian or jacobian_forward functions,
  // and the OpenMP version would only be called if OpenMP is
  // available and the Jacobian matrix is large enough for
  // parallelization to be worthwhile.  Note that jacobian_out must be
  // allocated to be of size m*n, where m is the number of dependent
  // variables and n is the number of independents. The independents
  // and dependents must have already been identified with the
  // functions "independent" and "dependent", otherwise this function
  // will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED. In the
  // resulting matrix, the "m" dimension of the matrix varies
  // fastest. This is implemented using a forward pass, appropriate
  // for m>=n.
  void
  Stack::jacobian_forward_openmp(Real* jacobian_out) const
  {

    // Number of blocks to cycle through, including a possible last
    // block containing fewer than MULTIPASS_SIZE variables
    int n_block = (n_independent() + MULTIPASS_SIZE - 1)
      / MULTIPASS_SIZE;
    uIndex n_extra = n_independent() % MULTIPASS_SIZE;
    
    int iblock;
    
#pragma omp parallel
    {
      //      std::vector<Block<MULTIPASS_SIZE,Real> > 
      //	gradient_multipass_b(max_gradient_);
      uIndex gradient_multipass_size = max_gradient_*MULTIPASS_SIZE;
      Real* __restrict gradient_multipass_b 
	= alloc_aligned<Real>(gradient_multipass_size);
      
#pragma omp for schedule(static)
      for (iblock = 0; iblock < n_block; iblock++) {
	// Set the index to the dependent variables for this block
	uIndex i_independent =  MULTIPASS_SIZE * iblock;
	
	uIndex block_size = MULTIPASS_SIZE;
	// If this is the last iteration and the number of extra
	// elements is non-zero, then set the block size to the number
	// of extra elements. If the number of extra elements is zero,
	// then the number of independent variables is exactly divisible
	// by MULTIPASS_SIZE, so the last iteration will be the
	// same as all the rest.
	if (iblock == n_block-1 && n_extra > 0) {
	  block_size = n_extra;
	}
	
	// Set the initial gradients all to zero
	for (std::size_t i = 0; i < gradient_multipass_size; i++) {
	  gradient_multipass_b[i] = 0.0;
	}
	// Each seed vector has one non-zero entry of 1.0
	for (uIndex i = 0; i < block_size; i++) {
	  gradient_multipass_b[independent_index_[i_independent+i]*MULTIPASS_SIZE+i] = 1.0;
	}

	jacobian_forward_kernel(gradient_multipass_b);

	// Copy the gradients corresponding to the dependent variables
	// into the Jacobian matrix
	for (uIndex idep = 0; idep < n_dependent(); idep++) {
	  for (uIndex i = 0; i < block_size; i++) {
	    jacobian_out[(i_independent+i)*n_dependent()+idep]
	      = gradient_multipass_b[dependent_index_[idep]*MULTIPASS_SIZE+i];
	  }
	}
      } // End of loop over blocks
      free_aligned(gradient_multipass_b);
    } // End of parallel section
  } // End of jacobian function


  // Compute the Jacobian matrix; note that jacobian_out must be
  // allocated to be of size m*n, where m is the number of dependent
  // variables and n is the number of independents. The independents
  // and dependents must have already been identified with the
  // functions "independent" and "dependent", otherwise this function
  // will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED. In the
  // resulting matrix, the "m" dimension of the matrix varies
  // fastest. This is implemented using a forward pass, appropriate
  // for m>=n.
  void
  Stack::jacobian_forward(Real* jacobian_out)
  {
    if (independent_index_.empty() || dependent_index_.empty()) {
      throw(dependents_or_independents_not_identified());
    }
#ifdef _OPENMP
    if (have_openmp_ 
	&& !openmp_manually_disabled_
	&& n_independent() > MULTIPASS_SIZE
	&& omp_get_max_threads() > 1) {
      // Call the parallel version
      jacobian_forward_openmp(jacobian_out);
      return;
    }
#endif

    // For optimization reasons, we process a block of
    // MULTIPASS_SIZE columns of the Jacobian at once; calculate
    // how many blocks are needed and how many extras will remain
    uIndex n_block = n_independent() / MULTIPASS_SIZE;
    uIndex n_extra = n_independent() % MULTIPASS_SIZE;

    ///gradient_multipass_.resize(max_gradient_);
    uIndex gradient_multipass_size = max_gradient_*MULTIPASS_SIZE;
    Real* __restrict gradient_multipass_b 
      = alloc_aligned<Real>(gradient_multipass_size);

    // Loop over blocks of MULTIPASS_SIZE columns
    for (uIndex iblock = 0; iblock < n_block; iblock++) {
      // Set the index to the dependent variables for this block
      uIndex i_independent =  MULTIPASS_SIZE * iblock;

      // Set the initial gradients all to zero
      ///zero_gradient_multipass();
      for (std::size_t i = 0; i < gradient_multipass_size; i++) {
	gradient_multipass_b[i] = 0.0;
      }

      // Each seed vector has one non-zero entry of 1.0
      for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	gradient_multipass_b[independent_index_[i_independent+i]*MULTIPASS_SIZE+i] = 1.0;
      }

      jacobian_forward_kernel(gradient_multipass_b);

      // Copy the gradients corresponding to the dependent variables
      // into the Jacobian matrix
      for (uIndex idep = 0; idep < n_dependent(); idep++) {
	for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	  jacobian_out[(i_independent+i)*n_dependent()+idep] 
	    = gradient_multipass_b[dependent_index_[idep]*MULTIPASS_SIZE+i];
	}
      }
      i_independent += MULTIPASS_SIZE;
    } // End of loop over blocks
    
    // Now do the same but for the remaining few columns in the matrix
    if (n_extra > 0) {
      uIndex i_independent =  MULTIPASS_SIZE * n_block;
      ///zero_gradient_multipass();
      for (std::size_t i = 0; i < gradient_multipass_size; i++) {
	gradient_multipass_b[i] = 0.0;
      }

      for (uIndex i = 0; i < n_extra; i++) {
	gradient_multipass_b[independent_index_[i_independent+i]*MULTIPASS_SIZE+i] = 1.0;
      }

      jacobian_forward_kernel_extra(gradient_multipass_b, n_extra);

      for (uIndex idep = 0; idep < n_dependent(); idep++) {
	for (uIndex i = 0; i < n_extra; i++) {
	  jacobian_out[(i_independent+i)*n_dependent()+idep] 
	    = gradient_multipass_b[dependent_index_[idep]*MULTIPASS_SIZE+i];
	}
      }
    }

    free_aligned(gradient_multipass_b);
  }


  // Compute the Jacobian matrix, parallelized using OpenMP.  Normally
  // the user would call the jacobian or jacobian_reverse functions,
  // and the OpenMP version would only be called if OpenMP is
  // available and the Jacobian matrix is large enough for
  // parallelization to be worthwhile.  Note that jacobian_out must be
  // allocated to be of size m*n, where m is the number of dependent
  // variables and n is the number of independents. The independents
  // and dependents must have already been identified with the
  // functions "independent" and "dependent", otherwise this function
  // will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED. In the
  // resulting matrix, the "m" dimension of the matrix varies
  // fastest. This is implemented using a reverse pass, appropriate
  // for m<n.
  void
  Stack::jacobian_reverse_openmp(Real* jacobian_out) const
  {

    // Number of blocks to cycle through, including a possible last
    // block containing fewer than MULTIPASS_SIZE variables
    int n_block = (n_dependent() + MULTIPASS_SIZE - 1)
      / MULTIPASS_SIZE;
    uIndex n_extra = n_dependent() % MULTIPASS_SIZE;
    
    int iblock;

    // Inside the OpenMP loop, the "this" pointer may be NULL if the
    // adept::Stack pointer is declared as thread-local and if the
    // OpenMP memory model uses thread-local storage for private
    // data. If this is the case then local pointers to or copies of
    // the following members of the adept::Stack object may need to be
    // made: dependent_index_ n_statements_ statement_ multiplier_
    // index_ independent_index_ n_dependent() n_independent().
    // Limited testing implies this is OK though.

#pragma omp parallel
    {
      std::vector<Block<MULTIPASS_SIZE,Real> > 
	gradient_multipass_b(max_gradient_);
      
#pragma omp for schedule(static)
      for (iblock = 0; iblock < n_block; iblock++) {
	// Set the index to the dependent variables for this block
	uIndex i_dependent =  MULTIPASS_SIZE * iblock;
	
	uIndex block_size = MULTIPASS_SIZE;
	// If this is the last iteration and the number of extra
	// elements is non-zero, then set the block size to the number
	// of extra elements. If the number of extra elements is zero,
	// then the number of independent variables is exactly divisible
	// by MULTIPASS_SIZE, so the last iteration will be the
	// same as all the rest.
	if (iblock == n_block-1 && n_extra > 0) {
	  block_size = n_extra;
	}

	// Set the initial gradients all to zero
	for (std::size_t i = 0; i < gradient_multipass_b.size(); i++) {
	  gradient_multipass_b[i].zero();
	}
	// Each seed vector has one non-zero entry of 1.0
	for (uIndex i = 0; i < block_size; i++) {
	  gradient_multipass_b[dependent_index_[i_dependent+i]][i] = 1.0;
	}

	// Loop backward through the derivative statements
	for (uIndex ist = n_statements_-1; ist > 0; ist--) {
	  const Statement& statement = statement_[ist];
	  // We copy the RHS to "a" in case it appears on the LHS in any
	  // of the following statements
	  Real a[MULTIPASS_SIZE];
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
	  // For large blocks, we only process the ones where a[i] is
	  // non-zero
	  uIndex i_non_zero[MULTIPASS_SIZE];
#endif
	  uIndex n_non_zero = 0;
	  for (uIndex i = 0; i < block_size; i++) {
	    a[i] = gradient_multipass_b[statement.index][i];
	    gradient_multipass_b[statement.index][i] = 0.0;
	    if (a[i] != 0.0) {
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
	      i_non_zero[n_non_zero++] = i;
#else
	      n_non_zero = 1;
#endif
	    }
	  }

	  // Only do anything for this statement if any of the a values
	  // are non-zero
	  if (n_non_zero) {
	    // Loop through the operations
	    for (uIndex iop = statement_[ist-1].end_plus_one;
		 iop < statement.end_plus_one; iop++) {
	      // Try to minimize pointer dereferencing by making local
	      // copies
	      Real multiplier = multiplier_[iop];
	      Real* __restrict gradient_multipass 
		= &(gradient_multipass_b[index_[iop]][0]);
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
	      // For large blocks, loop over only the indices
	      // corresponding to non-zero a
	      for (uIndex i = 0; i < n_non_zero; i++) {
		gradient_multipass[i_non_zero[i]] += multiplier*a[i_non_zero[i]];
	      }
#else
	      // For small blocks, do all indices
	      for (uIndex i = 0; i < block_size; i++) {
	      //	      for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
		gradient_multipass[i] += multiplier*a[i];
	      }
#endif
	    }
	  }
	} // End of loop over statement
	// Copy the gradients corresponding to the independent
	// variables into the Jacobian matrix
	for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	  for (uIndex i = 0; i < block_size; i++) {
	    jacobian_out[iindep*n_dependent()+i_dependent+i] 
	      = gradient_multipass_b[independent_index_[iindep]][i];
	  }
	}
      } // End of loop over blocks
    } // end #pragma omp parallel
  } // end jacobian_reverse_openmp


  // Compute the Jacobian matrix; note that jacobian_out must be
  // allocated to be of size m*n, where m is the number of dependent
  // variables and n is the number of independents. The independents
  // and dependents must have already been identified with the
  // functions "independent" and "dependent", otherwise this function
  // will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED. In the
  // resulting matrix, the "m" dimension of the matrix varies
  // fastest. This is implemented using a reverse pass, appropriate
  // for m<n.
  void
  Stack::jacobian_reverse(Real* jacobian_out)
  {
    if (independent_index_.empty() || dependent_index_.empty()) {
      throw(dependents_or_independents_not_identified());
    }
#ifdef _OPENMP
    if (have_openmp_ 
	&& !openmp_manually_disabled_
	&& n_dependent() > MULTIPASS_SIZE
	&& omp_get_max_threads() > 1) {
      // Call the parallel version
      jacobian_reverse_openmp(jacobian_out);
      return;
    }
#endif

    //    gradient_multipass_.resize(max_gradient_);
    std::vector<Block<MULTIPASS_SIZE,Real> > 
      gradient_multipass_b(max_gradient_);

    // For optimization reasons, we process a block of
    // MULTIPASS_SIZE rows of the Jacobian at once; calculate
    // how many blocks are needed and how many extras will remain
    uIndex n_block = n_dependent() / MULTIPASS_SIZE;
    uIndex n_extra = n_dependent() % MULTIPASS_SIZE;
    uIndex i_dependent = 0; // uIndex of first row in the block we are
			    // currently computing
    // Loop over the of MULTIPASS_SIZE rows
    for (uIndex iblock = 0; iblock < n_block; iblock++) {
      // Set the initial gradients all to zero
      //      zero_gradient_multipass();
      for (std::size_t i = 0; i < gradient_multipass_b.size(); i++) {
	gradient_multipass_b[i].zero();
      }

      // Each seed vector has one non-zero entry of 1.0
      for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	gradient_multipass_b[dependent_index_[i_dependent+i]][i] = 1.0;
      }
      // Loop backward through the derivative statements
      for (uIndex ist = n_statements_-1; ist > 0; ist--) {
	const Statement& statement = statement_[ist];
	// We copy the RHS to "a" in case it appears on the LHS in any
	// of the following statements
	Real a[MULTIPASS_SIZE];
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
	// For large blocks, we only process the ones where a[i] is
	// non-zero
	uIndex i_non_zero[MULTIPASS_SIZE];
#endif
	uIndex n_non_zero = 0;
	for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	  a[i] = gradient_multipass_b[statement.index][i];
	  gradient_multipass_b[statement.index][i] = 0.0;
	  if (a[i] != 0.0) {
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
	    i_non_zero[n_non_zero++] = i;
#else
	    n_non_zero = 1;
#endif
	  }
	}
	// Only do anything for this statement if any of the a values
	// are non-zero
	if (n_non_zero) {
	  // Loop through the operations
	  for (uIndex iop = statement_[ist-1].end_plus_one;
	       iop < statement.end_plus_one; iop++) {
	    // Try to minimize pointer dereferencing by making local
	    // copies
	    Real multiplier = multiplier_[iop];
	    Real* __restrict gradient_multipass 
	      = &(gradient_multipass_b[index_[iop]][0]);
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
	    // For large blocks, loop over only the indices
	    // corresponding to non-zero a
	    for (uIndex i = 0; i < n_non_zero; i++) {
	      gradient_multipass[i_non_zero[i]] += multiplier*a[i_non_zero[i]];
	    }
#else
	    // For small blocks, do all indices
	    for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	      gradient_multipass[i] += multiplier*a[i];
	    }
#endif
	  }
	}
      } // End of loop over statement
      // Copy the gradients corresponding to the independent variables
      // into the Jacobian matrix
      for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	  jacobian_out[iindep*n_dependent()+i_dependent+i] 
	    = gradient_multipass_b[independent_index_[iindep]][i];
	}
      }
      i_dependent += MULTIPASS_SIZE;
    } // End of loop over blocks
    
    // Now do the same but for the remaining few rows in the matrix
    if (n_extra > 0) {
      for (std::size_t i = 0; i < gradient_multipass_b.size(); i++) {
	gradient_multipass_b[i].zero();
      }
      //      zero_gradient_multipass();
      for (uIndex i = 0; i < n_extra; i++) {
	gradient_multipass_b[dependent_index_[i_dependent+i]][i] = 1.0;
      }
      for (uIndex ist = n_statements_-1; ist > 0; ist--) {
	const Statement& statement = statement_[ist];
	Real a[MULTIPASS_SIZE];
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
	uIndex i_non_zero[MULTIPASS_SIZE];
#endif
	uIndex n_non_zero = 0;
	for (uIndex i = 0; i < n_extra; i++) {
	  a[i] = gradient_multipass_b[statement.index][i];
	  gradient_multipass_b[statement.index][i] = 0.0;
	  if (a[i] != 0.0) {
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
	    i_non_zero[n_non_zero++] = i;
#else
	    n_non_zero = 1;
#endif
	  }
	}
	if (n_non_zero) {
	  for (uIndex iop = statement_[ist-1].end_plus_one;
	       iop < statement.end_plus_one; iop++) {
	    Real multiplier = multiplier_[iop];
	    Real* __restrict gradient_multipass 
	      = &(gradient_multipass_b[index_[iop]][0]);
	    //	    if (index_[iop] > max_gradient_-1
	    //		|| index_[iop] < 0) {
	    //	    std::cerr << "AAAAAA: iop=" << iop << " index_[iop]=" << index_[iop] << " max_gradient_=" << max_gradient_ << " ist=" << ist << "\n";
	      //	    }
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
	    for (uIndex i = 0; i < n_non_zero; i++) {
	      gradient_multipass[i_non_zero[i]] += multiplier*a[i_non_zero[i]];
	    }
#else
	    for (uIndex i = 0; i < n_extra; i++) {
	      //	      std::cerr << "BBBBB: i=" << i << " gradient_multipass[i]=" << gradient_multipass[i] << " multiplier=" << multiplier << " a[i]=" << a[i] << "\n";
	      gradient_multipass[i] += multiplier*a[i];
	    }
#endif
	  }
	}
      }
      for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	for (uIndex i = 0; i < n_extra; i++) {
	  jacobian_out[iindep*n_dependent()+i_dependent+i] 
	    = gradient_multipass_b[independent_index_[iindep]][i];
	}
      }
    }
  }

  // Compute the Jacobian matrix; note that jacobian_out must be
  // allocated to be of size m*n, where m is the number of dependent
  // variables and n is the number of independents. In the resulting
  // matrix, the "m" dimension of the matrix varies fastest. This is
  // implemented by calling one of jacobian_forward and
  // jacobian_reverse, whichever would be faster.
  void
  Stack::jacobian(Real* jacobian_out)
  {
    //    std::cout << ">>> Computing " << n_dependent() << "x" << n_independent()
    //	      << " Jacobian from " << n_statements_ << " statements, "
    //	      << n_operations() << " operations and " << max_gradient_ << " gradients\n";

    if (n_independent() <= n_dependent()) {
      jacobian_forward(jacobian_out);
    }
    else {
      jacobian_reverse(jacobian_out);
    }
  }
  
} // End namespace adept
