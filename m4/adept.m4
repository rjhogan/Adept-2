# ---------------------------------------------------------------------------
# FILE         : adept.m4
# COPYRIGHT    : 2018- ECMWF
# AUTHOR       : Alessio Bozzo
# LICENSE      : Apache License Version 2.0
# ----------------------------------------------------------------------------
#
# This software is licensed under the terms of the Apache Licence
# Version 2.0 which can be obtained at
# http://www.apache.org/licenses/LICENSE-2.0. In applying this
# licence, ECMWF does not waive the privileges and immunities granted
# to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#
# ----------------------------------------------------------------------------
#
# This file contains a macro processor (m4 file) to enable autotools
# to locate the Adept C++ library (version 2.0 or greater).  The file
# should be placed in the m4 directory of your package. If you have
# aclocal.m4 in your top-level directory then it will be found
# automatically; otherwise you will need the following in your
# configure.ac file:
#
#   m4_include([m4/adept.m4])
#
# Usage is then as follows in the configure.ac file
#
#   AX_CHECK_ADEPT([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
#
# for example:
#
#   AX_CHECK_ADEPT([have_adept=yes], [have_adept=no])
#
# This creates variables ADEPT_LDFLAGS and ADEPT_CPPFLAGS, and adds
# them to LDFLAGS and CPPFLAGS.
#
# The macro looks for the Adept library in system directories, but the
# user can specify another location by passing an argument to the
# configure script as follows:
#
#   ./configure --with-adept=/home/me/apps/adept-2.1
#
# ----------------------------------------------------------------------------

dnl defines a custom macro
AC_DEFUN([AX_CHECK_ADEPT], [

      dnl provides a framework to handle the --with-{arg} values passed to configure on the command line      
      AC_ARG_WITH([adept],
            [AS_HELP_STRING([--with-adept=DIR], [use Adept Library from directory DIR])],
            adept_prefix="$with_adept"
            []
            )
      
      AS_IF([test x$adept_prefix != x],
            [AS_IF([test -d "$adept_prefix/lib"],
                  [ADEPT_LDFLAGS="-L$adept_prefix/lib -Wl,-rpath,$adept_prefix/lib -ladept"
                  ADEPT_CPPFLAGS="-I$adept_prefix/include"],
		  [test -d "$adept_prefix/lib64"],
                  [ADEPT_LDFLAGS="-L$adept_prefix/lib64 -Wl,-rpath,$adept_prefix/lib64 -ladept"
                  ADEPT_CPPFLAGS="-I$adept_prefix/include"],
                  [AC_MSG_ERROR([
  -----------------------------------------------------------------------------
     --with-adept=$adept_prefix is not a valid directory
  -----------------------------------------------------------------------------])])],
      [AC_MSG_WARN([
  -----------------------------------------------------------------------------
   Missing option `--with-adept=DIR`. Looking for Adept Library
   into Linux default library search paths
  -----------------------------------------------------------------------------])]
           )
     
      LDFLAGS="$ADEPT_LDFLAGS $LDFLAGS"
      CPPFLAGS="$ADEPT_CPPFLAGS $CPPFLAGS"
      ax_have_adept=yes
      dnl checks for ADEPT
      AC_MSG_CHECKING([for Adept >= 2.0.4: including adept_arrays.h and linking via -ladept])
      AC_LINK_IFELSE([AC_LANG_PROGRAM([#include <adept_arrays.h>
      #include <string>
      #if ADEPT_VERSION < 20004
      #error "Adept version >= 2.0.4 required"
      #endif],[std::string test = adept::compiler_version()])],AC_MSG_RESULT([yes]),AC_MSG_RESULT([no])
      AC_MSG_ERROR([Unable to find Adept library version >= 2.0.4]))

      AS_IF([test "x$ax_have_adept" = xyes],
            dnl outputing Adept Library
            [AC_SUBST([ADEPT_LDFLAGS])
            AC_SUBST([ADEPT_CPPFLAGS])
            $1],
            [$2])
      ]
)
dnl vim:set softtabstop=4 shiftwidth=4 expandtab:
