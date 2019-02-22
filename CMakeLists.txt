cmake_minimum_required(VERSION 3.6)
project(Adept2)

set(CMAKE_CXX_STANDARD 14)

set(ADEPT_HEADERS
        adept/cpplapack.h
        include/adept/Active.h
        include/adept/ActiveConstReference.h
        include/adept/ActiveReference.h
        include/adept/Allocator.h
        include/adept/Array.h
        include/adept/ArrayWrapper.h
        include/adept/BinaryOperation.h
        include/adept/Expression.h
        include/adept/ExpressionSize.h
        include/adept/FixedArray.h
        include/adept/IndexedArray.h
        include/adept/Packet.h
        include/adept/RangeIndex.h
        include/adept/ScratchVector.h
        include/adept/SpecialMatrix.h
        include/adept/Stack.h
        include/adept/StackStorage.h
        include/adept/StackStorageOrig.h
        include/adept/StackStorageOrigStl.h
        include/adept/Statement.h
        include/adept/Storage.h
        include/adept/UnaryOperation.h
        include/adept/array_shortcuts.h
        include/adept/base.h
        include/adept/contiguous_matrix.h
        include/adept/cppblas.h
        include/adept/eval.h
        include/adept/exception.h
        include/adept/interp.h
        include/adept/inv.h
        include/adept/matmul.h
        include/adept/noalias.h
        include/adept/outer_product.h
        include/adept/reduce.h
        include/adept/scalar_shortcuts.h
        include/adept/settings.h
        include/adept/solve.h
        include/adept/spread.h
        include/adept/store_transpose.h
        include/adept/traits.h
        include/adept/vector_utilities.h
        include/adept/where.h
        )

set(ADEPT_SOURCES
        adept/Array.cpp
        adept/Stack.cpp
        adept/StackStorageOrig.cpp
        adept/Storage.cpp
        adept/cppblas.cpp
        adept/index.cpp
        adept/inv.cpp
        adept/jacobian.cpp
        adept/settings.cpp
        adept/solve.cpp
        adept/vector_utilities.cpp
        )

add_library(adept STATIC ${ADEPT_SOURCES} ${ADEPT_HEADERS})
add_library(adept::adept ALIAS adept)
target_include_directories(adept PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        )

install(TARGETS adept
        EXPORT adept
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        )

install(FILES ${HEADER_FILES} DESTINATION include/adept)

install(EXPORT adept
        NAMESPACE adept::
        DESTINATION cmake)

