/* jacobian.cpp -- Computation of Jacobian matrix

    Copyright (C) 2012-2014 University of Reading
    Copyright (C) 2015-2020 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifdef _OPENMP
#include <omp.h>
#endif

#include <adept_arrays.h>

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
  // allocated to be at least of size m*n, where m is the number of
  // dependent variables and n is the number of independents. The
  // independents and dependents must have already been identified
  // with the functions "independent" and "dependent", otherwise this
  // function will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED. The
  // offsets in memory of the two dimensions are provided by
  // dep_offset and indep_offset. This is implemented using a forward
  // pass, appropriate for m>=n.
  void
  Stack::jacobian_forward_openmp(Real* jacobian_out,
				 Index dep_offset, Index indep_offset) const
  {

    // Number of blocks to cycle through, including a possible last
    // block containing fewer than MULTIPASS_SIZE variables
    int n_block = (n_independent() + MULTIPASS_SIZE - 1)
      / MULTIPASS_SIZE;
    uIndex n_extra = n_independent() % MULTIPASS_SIZE;
    
#pragma omp parallel
    {
      //      std::vector<Block<MULTIPASS_SIZE,Real> > 
      //	gradient_multipass_b(max_gradient_);
      uIndex gradient_multipass_size = max_gradient_*MULTIPASS_SIZE;
      Real* __restrict gradient_multipass_b 
	= alloc_aligned<Real>(gradient_multipass_size);
      
#pragma omp for schedule(static)
      for (int iblock = 0; iblock < n_block; iblock++) {
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
	for (uIndex i = 0; i < gradient_multipass_size; i++) {
	  gradient_multipass_b[i] = 0.0;
	}
	// Each seed vector has one non-zero entry of 1.0
	for (uIndex i = 0; i < block_size; i++) {
	  gradient_multipass_b[independent_index_[i_independent+i]*MULTIPASS_SIZE+i] = 1.0;
	}

	jacobian_forward_kernel(gradient_multipass_b);

	// Copy the gradients corresponding to the dependent variables
	// into the Jacobian matrix
	if (indep_offset == 1) {
	  for (uIndex idep = 0; idep < n_dependent(); idep++) {
	    for (uIndex i = 0; i < block_size; i++) {
	      jacobian_out[idep*dep_offset+i_independent+i]
		= gradient_multipass_b[dependent_index_[idep]*MULTIPASS_SIZE+i];
	    }
	  }
	}
	else {
	  for (uIndex idep = 0; idep < n_dependent(); idep++) {
	    for (uIndex i = 0; i < block_size; i++) {
	      jacobian_out[(i_independent+i)*indep_offset+idep*dep_offset]
		= gradient_multipass_b[dependent_index_[idep]*MULTIPASS_SIZE+i];
	    }
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
  // will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED. This is
  // implemented using a forward pass, appropriate for m>=n.
  void
  Stack::jacobian_forward(Real* jacobian_out,
			  Index dep_offset, Index indep_offset) const
  {
    if (independent_index_.empty() || dependent_index_.empty()) {
      throw(dependents_or_independents_not_identified());
    }

    // If either of the offsets are zero, set them to the size of the
    // other dimension, which assumes that the full Jacobian matrix is
    // contiguous in memory.
    if (dep_offset <= 0) {
      dep_offset = n_independent();
    }
    if (indep_offset <= 0) {
      indep_offset = n_dependent();
    }

#ifdef _OPENMP
    if (have_openmp_ 
	&& !openmp_manually_disabled_
	&& n_independent() > MULTIPASS_SIZE
	&& omp_get_max_threads() > 1) {
      // Call the parallel version
      jacobian_forward_openmp(jacobian_out, dep_offset, indep_offset);
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
      for (uIndex i = 0; i < gradient_multipass_size; i++) {
	gradient_multipass_b[i] = 0.0;
      }

      // Each seed vector has one non-zero entry of 1.0
      for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	gradient_multipass_b[independent_index_[i_independent+i]*MULTIPASS_SIZE+i] = 1.0;
      }

      jacobian_forward_kernel(gradient_multipass_b);

      // Copy the gradients corresponding to the dependent variables
      // into the Jacobian matrix
      if (indep_offset == 1) {
	for (uIndex idep = 0; idep < n_dependent(); idep++) {
	  for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	    jacobian_out[idep*dep_offset+i_independent+i]
	      = gradient_multipass_b[dependent_index_[idep]*MULTIPASS_SIZE+i];
	  }
	}
      }
      else {
	for (uIndex idep = 0; idep < n_dependent(); idep++) {
	  for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	    jacobian_out[(i_independent+i)*indep_offset+idep*dep_offset] 
	      = gradient_multipass_b[dependent_index_[idep]*MULTIPASS_SIZE+i];
	  }
	}
      }
    } // End of loop over blocks
    
    // Now do the same but for the remaining few columns in the matrix
    if (n_extra > 0) {
      uIndex i_independent =  MULTIPASS_SIZE * n_block;
      ///zero_gradient_multipass();
      for (uIndex i = 0; i < gradient_multipass_size; i++) {
	gradient_multipass_b[i] = 0.0;
      }

      for (uIndex i = 0; i < n_extra; i++) {
	gradient_multipass_b[independent_index_[i_independent+i]*MULTIPASS_SIZE+i] = 1.0;
      }

      jacobian_forward_kernel_extra(gradient_multipass_b, n_extra);

      if (indep_offset == 1) {
	for (uIndex idep = 0; idep < n_dependent(); idep++) {
	  for (uIndex i = 0; i < n_extra; i++) {
	    jacobian_out[idep*dep_offset+i_independent+i]
	      = gradient_multipass_b[dependent_index_[idep]*MULTIPASS_SIZE+i];
	  }
	}
      }
      else {
	for (uIndex idep = 0; idep < n_dependent(); idep++) {
	  for (uIndex i = 0; i < n_extra; i++) {
	    jacobian_out[(i_independent+i)*indep_offset+idep*dep_offset] 
	      = gradient_multipass_b[dependent_index_[idep]*MULTIPASS_SIZE+i];
	  }
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
  // allocated to be at least of size m*n, where m is the number of
  // dependent variables and n is the number of independents. The
  // independents and dependents must have already been identified
  // with the functions "independent" and "dependent", otherwise this
  // function will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED. The
  // offsets in memory of the two dimensions are provided by
  // dep_offset and indep_offset.  This is implemented using a reverse
  // pass, appropriate for m<n.
  void
  Stack::jacobian_reverse_openmp(Real* jacobian_out,
				 Index dep_offset, Index indep_offset) const
  {

    // Number of blocks to cycle through, including a possible last
    // block containing fewer than MULTIPASS_SIZE variables
    int n_block = (n_dependent() + MULTIPASS_SIZE - 1)
      / MULTIPASS_SIZE;
    uIndex n_extra = n_dependent() % MULTIPASS_SIZE;
    
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
      for (int iblock = 0; iblock < n_block; iblock++) {
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
	if (dep_offset == 1) {
	  for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	    for (uIndex i = 0; i < block_size; i++) {
	      jacobian_out[iindep*indep_offset+i_dependent+i] 
		= gradient_multipass_b[independent_index_[iindep]][i];
	    }
	  }
	}
	else {
	  for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	    for (uIndex i = 0; i < block_size; i++) {
	      jacobian_out[iindep*indep_offset+(i_dependent+i)*dep_offset] 
		= gradient_multipass_b[independent_index_[iindep]][i];
	    }
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
  // will fail with FAILURE_XXDEPENDENT_NOT_IDENTIFIED.  This is
  // implemented using a reverse pass, appropriate for m<n.
  void
  Stack::jacobian_reverse(Real* jacobian_out,
			  Index dep_offset, Index indep_offset) const
  {
    if (independent_index_.empty() || dependent_index_.empty()) {
      throw(dependents_or_independents_not_identified());
    }

    // If either of the offsets are zero, set them to the size of the
    // other dimension, which assumes that the full Jacobian matrix is
    // contiguous in memory.
    if (dep_offset <= 0) {
      dep_offset = n_independent();
    }
    if (indep_offset <= 0) {
      indep_offset = n_dependent();
    }

#ifdef _OPENMP
    if (have_openmp_ 
	&& !openmp_manually_disabled_
	&& n_dependent() > MULTIPASS_SIZE
	&& omp_get_max_threads() > 1) {
      // Call the parallel version
      jacobian_reverse_openmp(jacobian_out,
			      dep_offset, indep_offset);
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
      if (dep_offset == 1) {
	for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	  for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	    jacobian_out[iindep*indep_offset+i_dependent+i] 
	      = gradient_multipass_b[independent_index_[iindep]][i];
	  }
	}
      }
      else {
	for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	  for (uIndex i = 0; i < MULTIPASS_SIZE; i++) {
	    jacobian_out[iindep*indep_offset+(i_dependent+i)*dep_offset] 
	      = gradient_multipass_b[independent_index_[iindep]][i];
	  }
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
#if MULTIPASS_SIZE > MULTIPASS_SIZE_ZERO_CHECK
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
      if (dep_offset == 1) {
	for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	  for (uIndex i = 0; i < n_extra; i++) {
	    jacobian_out[iindep*indep_offset+i_dependent+i] 
	      = gradient_multipass_b[independent_index_[iindep]][i];
	  }
	}
      }
      else {
	for (uIndex iindep = 0; iindep < n_independent(); iindep++) {
	  for (uIndex i = 0; i < n_extra; i++) {
	    jacobian_out[iindep*indep_offset+(i_dependent+i)*dep_offset] 
	      = gradient_multipass_b[independent_index_[iindep]][i];
	  }
	}
      }
    }
  }
  
  // Return the Jacobian matrix in the matrix "jac", using the forward
  // or reverse method depending which would be faster
  void Stack::jacobian(Array<2,Real,false> jac) const {
    if (jac.dimension(0) != n_dependent()
	|| jac.dimension(1) != n_independent()) {
      throw size_mismatch("Jacobian matrix has wrong size");
    }
    if (n_independent() <= n_dependent()) {
      jacobian_forward(jac.data(), jac.offset(0), jac.offset(1));
    }
    else {
      jacobian_reverse(jac.data(), jac.offset(0), jac.offset(1));
    }
  }

  // Return the Jacobian matrix in the matrix "jac", explicitly
  // specifying whether to use the forward or reverse method
  void Stack::jacobian_forward(Array<2,Real,false> jac) const {
    if (jac.dimension(0) != n_dependent()
	|| jac.dimension(1) != n_independent()) {
      throw size_mismatch("Jacobian matrix has wrong size");
    }
    jacobian_forward(jac.data(), jac.offset(0), jac.offset(1));
  }

  void Stack::jacobian_reverse(Array<2,Real,false> jac) const {
    if (jac.dimension(0) != n_dependent()
	|| jac.dimension(1) != n_independent()) {
      throw size_mismatch("Jacobian matrix has wrong size");
    }
    jacobian_reverse(jac.data(), jac.offset(0), jac.offset(1));
  }

  // Return the Jacobian matrix using the forward or reverse method
  // depending which would be faster
  Array<2,Real,false> Stack::jacobian() const {
    Array<2,Real,false> jac(n_dependent(), n_independent());
    if (n_independent() <= n_dependent()) {
      jacobian_forward(jac.data(), jac.offset(0), jac.offset(1));
    }
    else {
      jacobian_reverse(jac.data(), jac.offset(0), jac.offset(1));
    }
    return jac;
  }

  // Return the Jacobian matrix, explicitly specifying whether to use
  // the forward or reverse method
  Array<2,Real,false> Stack::jacobian_forward() const {
    Array<2,Real,false> jac(n_dependent(), n_independent());
    jacobian_forward(jac.data(), jac.offset(0), jac.offset(1));
    return jac;
  }

  Array<2,Real,false> Stack::jacobian_reverse() const {
    Array<2,Real,false> jac(n_dependent(), n_independent());
    jacobian_reverse(jac.data(), jac.offset(0), jac.offset(1));
    return jac;
  }

} // End namespace adept
