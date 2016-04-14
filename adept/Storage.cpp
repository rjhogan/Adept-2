/* Storage.cpp -- Global variables recording use of Storage objects

    Copyright (C) 2015 European Centre for Medium-Range Weather Forecasts

    Author: Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.

*/

#include <adept/Storage.h>

namespace adept {
  namespace internal {
    Index n_storage_objects_created_;
    Index n_storage_objects_deleted_;
  }
}
