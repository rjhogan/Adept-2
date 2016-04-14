/* contiguous_matrix.h -- Return matrix with contiguous storage

    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/

#ifndef AdeptContiguousMatrix_H
#define AdeptContiguousMatrix_H 1

#include <adept/Array.h>

namespace adept {
  namespace internal {
    
    // If for input into BLAS or LAPACK a matrix is required to have
    // one dimension contiguous and increasing in memory, then call
    // this function: if the matrix has this property then the
    // returned matrix in "out" will be linked to the input matrix;
    // otherwise, "out" will be a copy of "in" but satisfying this
    // condition. The returned "order" is ROW_MAJOR or COL_MAJOR
    // stating the storage type of the returned matrix.
    template <typename T, bool IsActive>
    MatrixStorageOrder contiguous_matrix(Array<2,T,IsActive>& in, 
					 Array<2,T,IsActive>& out,
					 Index& stride) {
      MatrixStorageOrder order = ROW_MAJOR;
      if (in.empty()) {
	throw(invalid_operation("Input matrix must not be empty"));
      }
      if (in.dimension(1) == 1) {
	out.link(in);
	stride = in.offset(0);
      }
      else if (in.dimension(0) == 1) {
	order = COL_MAJOR;
	out.link(in);
	stride = in.offset(1);
      }
      else {
	out.resize_row_major(in.dimensions());
	out = in;
	stride = in.offset(0);
      }
      return order;
    }

    // As contiguous_matrix but checks that the input matrix is square
    template <typename T, bool IsActive>
    MatrixStorageOrder contiguous_square_matrix(Array<2,T,IsActive>& in, 
						Array<2,T,IsActive>& out,
						Index& stride) {
      if (in.dimension(0) != in.dimension(1)) {
	throw(invalid_operation("Square matrix required"));
      }
      return contiguous_matrix(in, out, stride);
    }

  }
}


#endif
