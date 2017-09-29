check_PROGRAMS = autodiff_benchmark animate matrix_benchmark
autodiff_benchmark_SOURCES = autodiff_benchmark.cpp \
	differentiator.h advection_schemes.h \
	advection_schemes_AD.h advection_schemes_K.h nx.h

autodiff_benchmark_CPPFLAGS = -I../include
autodiff_benchmark_LDFLAGS = -static -no-install -L../adept/.libs
autodiff_benchmark_LDADD = -ladept

animate_SOURCES = animate.cpp
animate_CPPFLAGS = -I../include

matrix_benchmark_SOURCES = matrix_benchmark.cpp
matrix_benchmark_CPPFLAGS = -I../include
matrix_benchmark_LDFLAGS = -static -no-install -L../adept/.libs
matrix_benchmark_LDADD = -ladept
