# Adept 2: Combined array and automatic differentiation library in C++

## Introduction

The Adept version 2.1 software library provides three different
functionalities:

* Its automatic differentiation capability enables algorithms written
  in C++ to be differentiated with little code modification, very
  useful for a wide range of applications that involve mathematical
  optimization. It is backwards compatible with and as fast as Adept
  1.1. The name "Adept" refers to "Automatic Differentiation using
  Expression Templates".

* Its array capability provides support for vectors, matrices, arrays
  of up to 7 dimensions and linear algebra. Adept 2 uses a single
  expression-template framework under the hood to enable array
  operations to be differentiated with very good computational
  performance.

* Its optimization capability provides the various minimization
  algorithms (Levenberg, Levenberg-Marquardt, Conjugate Gradient and
  Limited Memory BFGS) each of which can be used with or without box
  constraints on the state variables. The interface to the
  optimization functionality is in terms of Adept vectors and matrices.

If you are not interested in the array or optimization capabilities of
Adept 2 then Adept 1.1 may be more to your liking as a very
lightweight library that has virtually all the
automatic-differentiation capabilities of version 2.


## Documentation and links

* The [Adept web site](http://www.met.reading.ac.uk/clouds/adept/) for formal Adept releases
* The [Adept-2 GitHub page](https://github.com/rjhogan/Adept-2) for the latest snapshot
* The [Adept-1.1 GitHub page](https://github.com/rjhogan/Adept) for the older (scalar) library
* A detailed [User Guide](http://www.met.reading.ac.uk/clouds/adept/adept_documentation.pdf)
* A paper describing the automatic differentiation capability: [Hogan, R. J., 2014: Fast reversemode automatic differentiation using expression templates in C++. *ACM Trans. Math. Softw.* **40,** 26:1-26:16](http://www.met.reading.ac.uk/~swrhgnrj/publications/adept.pdf)
* The [Adept Wikipedia page](https://en.wikipedia.org/wiki/Adept_(C++_library))
* Bug fixes, and queries not answered by the documentation, should be addressed to Robin Hogan (r.j.hogan at ecmwf.int)

## Installation

To build Adept from a GitHub snapshot, first do the following to
recreate the configure script:

    autoreconf -i

Formal release packages already contain a configure script. The normal
build sequence is then:

    ./configure
    make
    make check
    make install

Please consult the User Guide for further installation options; in
particular, if you plan to make serious us of matrix multiplication
and linear algebra then you should compile Adept to use an optimized
BLAS library such as OpenBLAS.


## License and copyright

The code in this package has a mix of copyright owners:

Copyright (C) 2012-2015 University of Reading

Copyright (C) 2015-     European Centre for Medium-Range Weather Forecasts

Two licenses are used for the code in this package:

* The files that form the Adept library are distributed under the
  conditions of the Apache License, Version 2 - see the COPYING file
  for details.  This is a permissive free-software license but one
  that does impose a few conditions if you intend to distribute
  derivative works.  The files this license applies to are those in
  the include/ and adept/ directories, and the subdirectories below
  them.

* All code in the test/ and benchmark/ directories is subject to the
  terms of the GNU all-permissive license, given at the top of those
  files - basically you can do what you like with the code from these
  files.

If you use Adept in published scientific work then it is requested
that you cite the Hogan (2014) paper above, but this is not a
condition of the license.
