/* VectorOrientation.h -- 

    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/


namespace adept {
  enum VectorOrientation {
    UNSPECIFIED_VECTOR_ORIENTATION,
    ROW_VECTOR_ORIENTATION,
    COL_VECTOR_ORIENTATION
  };

  namespace internal {
    template <VectorOrientation L, VectorOrientation R>
    struct combined_orientation {
      static const bool is_legal = true;
      static const VectorOrientation value = L;
      static const VectorOrientation value_matmul = L;
    };

    template <VectorOrientation R>
    struct combined_orientation<UNSPECIFIED_VECTOR_ORIENTATION,R> {
      static const bool is_legal = true;
      static const VectorOrientation value = R;
      static const VectorOrientation value_matmul = R;
    };

    template <>
    struct combined_orientation<ROW_VECTOR_ORIENTATION,COL_VECTOR_ORIENTATION> {
      static const bool is_legal = false;
      //      static const VectorOrientation value = CANNOT_COMBINE_ROW_AND_COLUMN_VECTOR_EXPRESSIONS;
      //      static const VectorOrientation value_matmul = USED_ROW_VECTOR_WHERE_COL_VECTOR_REQUIRED;
    };

    template <>
    struct combined_orientation<COL_VECTOR_ORIENTATION,ROW_VECTOR_ORIENTATION> {
      static const bool is_legal = false;
      //      static const VectorOrientation value = CANNOT_COMBINE_COLUMN_AND_ROW_VECTOR_EXPRESSIONS;
      //      static const VectorOrientation value_matmul = USED_COL_VECTOR_WHERE_ROW_VECTOR_REQUIRED;
    };


    template <int LRank, VectorOrientation L,
	      int RRank, VectorOrientation R>
    struct matmul_props_ {
      static const int rank = 2;
      static const VectorOrientation orientation
        = UNSPECIFIED_VECTOR_ORIENTATION;
    };
    
    template <VectorOrientation L, VectorOrientation R>
    struct matmul_props_<2,L,1,R> {
      static const int rank = 1;
      static const VectorOrientation orientation
        = combined_orientation<R,COL_VECTOR_ORIENTATION>::value_matmul;
    };

    template <VectorOrientation L, VectorOrientation R>
    struct matmul_props_<1,L,2,R> {
      static const int rank = 1;
      static const VectorOrientation orientation
        = combined_orientation<L,ROW_VECTOR_ORIENTATION>::value_matmul;
    };

    template <VectorOrientation LL, VectorOrientation RR>
    struct matmul_result_ { static const int rank = -1; /*ERROR_ORIENTATION_OF_VECTORS_AMBIGUOUS;*/ };
    
    template <VectorOrientation LL>
    struct matmul_result_<LL,ROW_VECTOR_ORIENTATION> { static const int rank = 2; };
    
    template <VectorOrientation LL>
    struct matmul_result_<LL,COL_VECTOR_ORIENTATION> { static const int rank = 0; };

    template <VectorOrientation RR>
    struct matmul_result_<ROW_VECTOR_ORIENTATION,RR> { static const int rank = 0; };
    
    template <VectorOrientation RR>
    struct matmul_result_<COL_VECTOR_ORIENTATION,RR> { static const int rank = 2; };
    
    template <>
      struct matmul_result_<ROW_VECTOR_ORIENTATION,ROW_VECTOR_ORIENTATION>
    { static const int rank = -1; /*ERROR_CANNOT_MATMUL_TWO_ROW_VECTORS;*/ };
    
    template <>
    struct matmul_result_<COL_VECTOR_ORIENTATION,COL_VECTOR_ORIENTATION>
    { static const int rank = -1; /*ERROR_CANNOT_MATMUL_TWO_COL_VECTORS;*/ };
    
    template <VectorOrientation L, VectorOrientation R>
    struct matmul_props_<1,L,1,R> {
      static const int rank = matmul_result_<L,R>::rank;
      static const VectorOrientation orientation = UNSPECIFIED_VECTOR_ORIENTATION;
      static const int ERROR_IN_VECTOR_MATMUL = 1 / (rank+1);
    };

    template <class L, class R>
    struct matmul_props {
      static const int rank
        = matmul_props_<L::rank,L::vector_orientation,
			      R::rank,R::vector_orientation>::rank;
      static const VectorOrientation orientation
        = matmul_props_<L::rank,L::vector_orientation,
			      R::rank,R::vector_orientation>::orientation;
    };
    
  }
};

