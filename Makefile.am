pkgdata_DATA = README COPYING ChangeLog NEWS AUTHORS
SUBDIRS = adept include benchmark test
# The test/ directory does not use automake so we need to specify the
# files that will be included in the distribution
EXTRA_DIST = test/Makefile test/README test/*.cpp test/*.h test/run_tests.sh \
	doc/Makefile doc/README doc/COPYING doc/*.tex 
ACLOCAL_AMFLAGS = -I m4
