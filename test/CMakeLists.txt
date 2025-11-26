function(add_adept_test TESTNAME)
    set(TEST_SOURCES
            algorithm.cpp)
    set(TEST_HEADERS
            algorithm.h
            algorithm_with_and_without_ad.h)
    add_executable(${TESTNAME} ${TESTNAME}.cpp ${TEST_SOURCES} ${TEST_HEADERS})
    target_link_libraries(${TESTNAME}
            PRIVATE
            adept::adept)
    enable_testing()
    add_test(NAME ${TESTNAME} COMMAND ./${TESTNAME})
endfunction()

add_adept_test(test_adept)
#add_adept_test(test_adept_with_and_without_ad)
add_adept_test(test_array_derivatives)
add_adept_test(test_array_speed)
add_adept_test(test_arrays)
add_adept_test(test_checkpoint)
add_adept_test(test_constructors)
add_adept_test(test_derivatives)
add_adept_test(test_fixed_arrays)
#add_adept_test(test_gsl_interface)
add_adept_test(test_misc)
#add_adept_test(test_no_lib)
#add_adept_test(test_radiances)
#add_adept_test(test_radiances_array)
add_adept_test(test_thread_safe)
add_adept_test(test_thread_safe_arrays)
