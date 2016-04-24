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

namespace adept {

  using namespace internal;


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
  Stack::jacobian_forward_openmp(Real* jacobian_out)
  {
    if (independent_index_.empty() || dependent_index_.empty()) {
      throw(dependents_or_independents_not_identified());
    }

    // Number of blocks to cycle through, including a possible last
    // block containing fewer than ADEPT_MULTIPASS_SIZE variables
    int n_block = (n_independent() + ADEPT_MULTIPASS_SIZE - 1)
      / ADEPT_MULTIPASS_SIZE;
    uIndex n_extra = n_independent() % ADEPT_MULTIPASS_SIZE;
    
    int iblock;
    
#pragma omp parallel
    {
      std::vector<Block<ADEPT_MULTIPASS_SIZE,Real> > 
	gradient_multipass_b(max_gradient_);
      
#pragma omp for
      for (iblock = 0; iblock < n_block; iblock++) {
	// Set the index to the dependent variables for this block
	uIndex i_independent =  ADEPT_MULTIPASS_SIZE * iblock;
	
	uIndex block_size = ADEPT_MULTIPASS_SIZE;
	// If this is the last iteration and the number of extra
	// elements is non-zero, then set the block size to the number
	// of extra elements. If the number of extra elements is zero,
	// then the number of independent variables is exactly divisible
	// by ADEPT_MULTIPASS_SIZE, so the last iteration will be the
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
	  gradient_multipass_b[independent_index_[i_independent+i]][i] = 1.0;
	}
	// Loop forward through the derivative statements
	for (uIndex ist = 1; ist < n_statements_; ist++) {
	  const Statement& statement = statement_[ist];
	  // We copy the LHS to "a" in case it appears on the RHS in any
	  // of the following statements
	  Block<ADEPT_MULTIPASS_SIZE,Real> a; // Initialized to zero
					      // automatically

	  // Loop through operations
	  for (uIndex iop = statement_[ist-1].end_plus_one;
	       iop < statement.end_plus_one; iop++) {
	    // Loop through columns within this block; we hope the
	    // compiler can optimize this loop. Note that it is faster
	    // to always use ADEPT_MULTIPASS_SIZE, always known at
	    // compile time, than to use block_size, which is not, even
	    // though in the last iteration this may involve redundant
	    // computations.
	    if (multiplier_[iop] == 1.0) {
	    //	    if (__builtin_expect(multiplier_[iop] == 1.0,0)) {
	      for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
		//	      for (uIndex i = 0; i < block_size; i++) {
		a[i] += gradient_multipass_b[index_[iop]][i];
	      }
	    }
	    else {
	      for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
		//	      for (uIndex i = 0; i < block_size; i++) {
		a[i] += multiplier_[iop]*gradient_multipass_b[index_[iop]][i];
	      }
	    }
	  }
	  // Copy the results
	  for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	    gradient_multipass_b[statement.index][i] = a[i];
	  }
	} // End of loop over statements
	// Copy the gradients corresponding to the dependent variables
	// into the Jacobian matrix
	for (uIndex idep = 0; idep < n_dependent(); idep++) {
	  for (uIndex i = 0; i < block_size; i++) {
	    jacobian_out[(i_independent+i)*n_dependent()+idep]
	      = gradient_multipass_b[dependent_index_[idep]][i];
	  }
	}
      } // End of loop over blocks
    } // End of parallel section
  } // End of jacobian function



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
  Stack::jacobian_reverse_openmp(Real* jacobian_out)
  {
    if (independent_index_.empty() || dependent_index_.empty()) {
      throw(dependents_or_independents_not_identified());
    }

    // Number of blocks to cycle through, including a possible last
    // block containing fewer than ADEPT_MULTIPASS_SIZE variables
    int n_block = (n_dependent() + ADEPT_MULTIPASS_SIZE - 1)
      / ADEPT_MULTIPASS_SIZE;
    uIndex n_extra = n_dependent() % ADEPT_MULTIPASS_SIZE;
    
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
      std::vector<Block<ADEPT_MULTIPASS_SIZE,Real> > 
	gradient_multipass_b(max_gradient_);
      
#pragma omp for
      for (iblock = 0; iblock < n_block; iblock++) {
	// Set the index to the dependent variables for this block
	uIndex i_dependent =  ADEPT_MULTIPASS_SIZE * iblock;
	
	uIndex block_size = ADEPT_MULTIPASS_SIZE;
	// If this is the last iteration and the number of extra
	// elements is non-zero, then set the block size to the number
	// of extra elements. If the number of extra elements is zero,
	// then the number of independent variables is exactly divisible
	// by ADEPT_MULTIPASS_SIZE, so the last iteration will be the
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
	  Real a[ADEPT_MULTIPASS_SIZE];
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	  // For large blocks, we only process the ones where a[i] is
	  // non-zero
	  uIndex i_non_zero[ADEPT_MULTIPASS_SIZE];
#endif
	  uIndex n_non_zero = 0;
	  for (uIndex i = 0; i < block_size; i++) {
	    a[i] = gradient_multipass_b[statement.index][i];
	    gradient_multipass_b[statement.index][i] = 0.0;
	    if (a[i] != 0.0) {
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
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
	      register Real multiplier = multiplier_[iop];
	      register Real* __restrict gradient_multipass 
		= &(gradient_multipass_b[index_[iop]][0]);
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	      // For large blocks, loop over only the indices
	      // corresponding to non-zero a
	      for (uIndex i = 0; i < n_non_zero; i++) {
		gradient_multipass[i_non_zero[i]] += multiplier*a[i_non_zero[i]];
	      }
#else
	      // For small blocks, do all indices
	      for (uIndex i = 0; i < block_size; i++) {
	      //	      for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
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
	&& n_independent() > ADEPT_MULTIPASS_SIZE
	&& omp_get_max_threads() > 1) {
      // Call the parallel version
      jacobian_forward_openmp(jacobian_out);
      return;
    }
#endif

    gradient_multipass_.resize(max_gradient_);
    // For optimization reasons, we process a block of
    // ADEPT_MULTIPASS_SIZE columns of the Jacobian at once; calculate
    // how many blocks are needed and how many extras will remain
    uIndex n_block = n_independent() / ADEPT_MULTIPASS_SIZE;
    uIndex n_extra = n_independent() % ADEPT_MULTIPASS_SIZE;
    uIndex i_independent = 0; // uIndex of first column in the block we
			      // are currently computing
    // Loop over blocks of ADEPT_MULTIPASS_SIZE columns
    for (uIndex iblock = 0; iblock < n_block; iblock++) {
      // Set the initial gradients all to zero
      zero_gradient_multipass();
      // Each seed vector has one non-zero entry of 1.0
      for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	gradient_multipass_[independent_index_[i_independent+i]][i] = 1.0;
      }
      // Loop forward through the derivative statements
      for (uIndex ist = 1; ist < n_statements_; ist++) {
	const Statement& statement = statement_[ist];
	// We copy the LHS to "a" in case it appears on the RHS in any
	// of the following statements
	Block<ADEPT_MULTIPASS_SIZE,Real> a; // Initialized to zero
					    // automatically
	// Loop through operations
	for (uIndex iop = statement_[ist-1].end_plus_one;
	     iop < statement.end_plus_one; iop++) {
	  if (multiplier_[iop] == 1.0) {
	    // Loop through columns within this block; we hope the
	    // compiler can optimize this loop
	    for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	      a[i] += gradient_multipass_[index_[iop]][i];
	    }
	  }
	  else {
	    for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	      a[i] += multiplier_[iop]*gradient_multipass_[index_[iop]][i];
	    }
	  }
	}
	// Copy the results
	for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	  gradient_multipass_[statement.index][i] = a[i];
	}
      } // End of loop over statements
      // Copy the gradients corresponding to the dependent variables
      // into the Jacobian matrix
      for (uIndex idep = 0; idep < n_dependent(); idep++) {
	for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	  jacobian_out[(i_independent+i)*n_dependent()+idep] 
	    = gradient_multipass_[dependent_index_[idep]][i];
	}
      }
      i_independent += ADEPT_MULTIPASS_SIZE;
    } // End of loop over blocks
    
    // Now do the same but for the remaining few columns in the matrix
    if (n_extra > 0) {
      zero_gradient_multipass();
      for (uIndex i = 0; i < n_extra; i++) {
	gradient_multipass_[independent_index_[i_independent+i]][i] = 1.0;
      }
      for (uIndex ist = 1; ist < n_statements_; ist++) {
	const Statement& statement = statement_[ist];
	Block<ADEPT_MULTIPASS_SIZE,Real> a;
	for (uIndex iop = statement_[ist-1].end_plus_one;
	     iop < statement.end_plus_one; iop++) {
	  if (multiplier_[iop] == 1.0) {
	    for (uIndex i = 0; i < n_extra; i++) {
	      a[i] += gradient_multipass_[index_[iop]][i];
	    }
	  }
	  else {
	    for (uIndex i = 0; i < n_extra; i++) {
	      a[i] += multiplier_[iop]*gradient_multipass_[index_[iop]][i];
	    }
	  }
	}
	for (uIndex i = 0; i < n_extra; i++) {
	  gradient_multipass_[statement.index][i] = a[i];
	}
      }
      for (uIndex idep = 0; idep < n_dependent(); idep++) {
	for (uIndex i = 0; i < n_extra; i++) {
	  jacobian_out[(i_independent+i)*n_dependent()+idep] 
	    = gradient_multipass_[dependent_index_[idep]][i];
	}
      }
    }
  }


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
	&& n_dependent() > ADEPT_MULTIPASS_SIZE
	&& omp_get_max_threads() > 1) {
      // Call the parallel version
      jacobian_reverse_openmp(jacobian_out);
      return;
    }
#endif

    gradient_multipass_.resize(max_gradient_);
    // For optimization reasons, we process a block of
    // ADEPT_MULTIPASS_SIZE rows of the Jacobian at once; calculate
    // how many blocks are needed and how many extras will remain
    uIndex n_block = n_dependent() / ADEPT_MULTIPASS_SIZE;
    uIndex n_extra = n_dependent() % ADEPT_MULTIPASS_SIZE;
    uIndex i_dependent = 0; // uIndex of first row in the block we are
			    // currently computing
    // Loop over the of ADEPT_MULTIPASS_SIZE rows
    for (uIndex iblock = 0; iblock < n_block; iblock++) {
      // Set the initial gradients all to zero
      zero_gradient_multipass();
      // Each seed vector has one non-zero entry of 1.0
      for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	gradient_multipass_[dependent_index_[i_dependent+i]][i] = 1.0;
      }
      // Loop backward through the derivative statements
      for (uIndex ist = n_statements_-1; ist > 0; ist--) {
	const Statement& statement = statement_[ist];
	// We copy the RHS to "a" in case it appears on the LHS in any
	// of the following statements
	Real a[ADEPT_MULTIPASS_SIZE];
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	// For large blocks, we only process the ones where a[i] is
	// non-zero
	uIndex i_non_zero[ADEPT_MULTIPASS_SIZE];
#endif
	uIndex n_non_zero = 0;
	for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	  a[i] = gradient_multipass_[statement.index][i];
	  gradient_multipass_[statement.index][i] = 0.0;
	  if (a[i] != 0.0) {
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
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
	    register Real multiplier = multiplier_[iop];
	    register Real* __restrict gradient_multipass 
	      = &(gradient_multipass_[index_[iop]][0]);
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	    // For large blocks, loop over only the indices
	    // corresponding to non-zero a
	    for (uIndex i = 0; i < n_non_zero; i++) {
	      gradient_multipass[i_non_zero[i]] += multiplier*a[i_non_zero[i]];
	    }
#else
	    // For small blocks, do all indices
	    for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	      gradient_multipass[i] += multiplier*a[i];
	    }
#endif
	  }
	}
      } // End of loop over statement
      // Copy the gradients corresponding to the independent variables
      // into the Jacobian matrix
      for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	for (uIndex i = 0; i < ADEPT_MULTIPASS_SIZE; i++) {
	  jacobian_out[iindep*n_dependent()+i_dependent+i] 
	    = gradient_multipass_[independent_index_[iindep]][i];
	}
      }
      i_dependent += ADEPT_MULTIPASS_SIZE;
    } // End of loop over blocks
    
    // Now do the same but for the remaining few rows in the matrix
    if (n_extra > 0) {
      zero_gradient_multipass();
      for (uIndex i = 0; i < n_extra; i++) {
	gradient_multipass_[dependent_index_[i_dependent+i]][i] = 1.0;
      }
      for (uIndex ist = n_statements_-1; ist > 0; ist--) {
	const Statement& statement = statement_[ist];
	Real a[ADEPT_MULTIPASS_SIZE];
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	uIndex i_non_zero[ADEPT_MULTIPASS_SIZE];
#endif
	uIndex n_non_zero = 0;
	for (uIndex i = 0; i < n_extra; i++) {
	  a[i] = gradient_multipass_[statement.index][i];
	  gradient_multipass_[statement.index][i] = 0.0;
	  if (a[i] != 0.0) {
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	    i_non_zero[n_non_zero++] = i;
#else
	    n_non_zero = 1;
#endif
	  }
	}
	if (n_non_zero) {
	  for (uIndex iop = statement_[ist-1].end_plus_one;
	       iop < statement.end_plus_one; iop++) {
	    register Real multiplier = multiplier_[iop];
	    register Real* __restrict gradient_multipass 
	      = &(gradient_multipass_[index_[iop]][0]);
#if ADEPT_MULTIPASS_SIZE > ADEPT_MULTIPASS_SIZE_ZERO_CHECK
	    for (uIndex i = 0; i < n_non_zero; i++) {
	      gradient_multipass[i_non_zero[i]] += multiplier*a[i_non_zero[i]];
	    }
#else
	    for (uIndex i = 0; i < n_extra; i++) {
	      gradient_multipass[i] += multiplier*a[i];
	    }
#endif
	  }
	}
      }
      for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	for (uIndex i = 0; i < n_extra; i++) {
	  jacobian_out[iindep*n_dependent()+i_dependent+i] 
	    = gradient_multipass_[independent_index_[iindep]][i];
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
