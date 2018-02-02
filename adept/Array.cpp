/* Array.cpp -- Functions and global variables controlling array behaviour

    Copyright (C) 2015-2016 European Centre for Medium-Range Weather Forecasts

    Robin Hogan <r.j.hogan@ecmwf.int>

    This file is part of the Adept library.
*/


#include <adept/Array.h>

namespace adept {
  namespace internal {
    bool array_row_major_order = true;
    //    bool array_print_curly_brackets = true;

    // Variables describing how arrays are written to a stream
    ArrayPrintStyle array_print_style = PRINT_STYLE_CURLY;
    std::string vector_separator = ", ";
    std::string vector_print_before = "{";
    std::string vector_print_after = "}";
    std::string array_opening_bracket = "{";
    std::string array_closing_bracket = "}";
    std::string array_contiguous_separator = ", ";
    std::string array_non_contiguous_separator = ",\n";
    std::string array_print_before = "\n{";
    std::string array_print_after = "}";
    std::string array_print_empty_before = "(empty rank-";
    std::string array_print_empty_after = " array)";
    bool array_print_indent = true;
    bool array_print_empty_rank = true;
  }

  void set_array_print_style(ArrayPrintStyle ps) {
    using namespace internal;
    switch (ps) {
    case PRINT_STYLE_PLAIN:
       vector_separator = " ";
       vector_print_before = "";
       vector_print_after = "";
       array_opening_bracket = "";
       array_closing_bracket = "";
       array_contiguous_separator = " ";
       array_non_contiguous_separator = "\n";
       array_print_before = "";
       array_print_after = "";
       array_print_empty_before = "(empty rank-";
       array_print_empty_after = " array)";
       array_print_indent = false;
       array_print_empty_rank = true;
       break;
    case PRINT_STYLE_CSV:
       vector_separator = ", ";
       vector_print_before = "";
       vector_print_after = "";
       array_opening_bracket = "";
       array_closing_bracket = "";
       array_contiguous_separator = ", ";
       array_non_contiguous_separator = "\n";
       array_print_before = "";
       array_print_after = "";
       array_print_empty_before = "empty";
       array_print_empty_after = "";
       array_print_indent = false;
       array_print_empty_rank = false;
       break;
    case PRINT_STYLE_MATLAB:
       vector_separator = " ";
       vector_print_before = "[";
       vector_print_after = "]";
       array_opening_bracket = "";
       array_closing_bracket = "";
       array_contiguous_separator = " ";
       array_non_contiguous_separator = ";\n";
       array_print_before = "[";
       array_print_after = "]";
       array_print_empty_before = "[";
       array_print_empty_after = "]";
       array_print_indent = true;
       array_print_empty_rank = false;
       break;
    case PRINT_STYLE_CURLY:
       vector_separator = ", ";
       vector_print_before = "{";
       vector_print_after = "}";
       array_opening_bracket = "{";
       array_closing_bracket = "}";
       array_contiguous_separator = ", ";
       array_non_contiguous_separator = ",\n";
       array_print_before = "\n{";
       array_print_after = "}";
       array_print_empty_before = "(empty rank-";
       array_print_empty_after = " array)";
       array_print_indent = true;
       array_print_empty_rank = true;
       break;
    default:
      throw invalid_operation("Array print style not understood");
    }
    array_print_style = ps;
  }

}
