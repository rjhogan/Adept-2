# Adept 2
## Combined array and automatic differentiation library in C++

The Adept version 2 software library provides two different
functionalities:

* Its array capability provides support for vectors, matrices, arrays
  of up to 7 dimensions and linear algebra.

* Its automatic differentiation capability enables algorithms written
  in C++ to be differentiated with little code modification, very
  useful for a wide range of applications that involve mathematical
  optimization. It is backwards compatible with and as fast as Adept
  1.1.

If you are not interested in the array capability of Adept 2.0 then
Adept 1.1 may be more to your liking as a very lightweight library
that has virtually all the automatic-differentiation capabilities of
version 2.0.

For further information see:
* The [Adept-2 web site] (http://www.met.reading.ac.uk/clouds/adept2/)
* A detailed [User Guide] (http://www.met.reading.ac.uk/clouds/adept2/adept_documentation_v2.pdf)
* [A paper published in ACM TOMS] (http://www.met.reading.ac.uk/~swrhgnrj/publications/adept.pdf) describing the automatic differentiation capability

To build Adept from a GitHub snapshot, do the following:

    autoreconf -i

Then the normal make sequence:

    ./configure
    make
    make install