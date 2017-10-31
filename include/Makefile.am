include_HEADERS = adept.h adept_arrays.h

pkginclude_HEADERS = adept/Active.h adept/ActiveReference.h adept/Allocator.h \
	adept/Array.h adept/Expression.h adept/ExpressionSize.h \
	adept/IndexedArray.h adept/matmul.h adept/RangeIndex.h \
	adept/ScratchVector.h adept/SpecialMatrix.h adept/Stack.h \
	adept/StackStorage.h adept/StackStorageOrig.h \
	adept/StackStorageOrigStl.h adept/Statement.h adept/Storage.h \
	adept/array_shortcuts.h adept/base.h adept/reduce.h \
	adept/contiguous_matrix.h adept/exception.h adept/settings.h \
	adept/interp.h adept/ActiveConstReference.h adept/cppblas.h \
	adept/scalar_shortcuts.h adept/solve.h adept/traits.h adept/where.h \
	adept/vector_utilities.h adept/FixedArray.h adept/Packet.h \
	adept/UnaryOperation.h adept/BinaryOperation.h adept/ArrayWrapper.h \
	adept/outer_product.h adept/spread.h adept/inv.h adept/eval.h \
	adept/noalias.h adept/store_transpose.h

EXTRA_DIST = Timer.h create_adept_source_header adept_source.h

adept_source.h: @top_srcdir@/adept/*.h @top_srcdir@/adept/*.cpp @srcdir@/create_adept_source_header
	@srcdir@/create_adept_source_header
all-local: adept_source.h
