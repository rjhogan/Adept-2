/* Statement.h -- Original method to store statement & operation stacks

    Copyright (C) 2012-2014 University of Reading

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#ifndef AdeptStatement_H
#define AdeptStatement_H 1

#include <adept/base.h>

namespace adept {
  namespace internal {

    // Structure describing the LHS of a derivative expression.  For dx
    // = z dy + y dz, "index" would be the location of dx in the
    // gradient list, and "end_plus_one" would be one plus the location
    // of the final operation (multiplier-derivative pair) on the RHS,
    // in this case y dz.
    struct Statement {
      Statement() { }
      Statement(uIndex index_, uIndex end_plus_one_)
	: index(index_), end_plus_one(end_plus_one_) { }
      uIndex index;
      uIndex end_plus_one;
    };
 
  }
}

#endif
