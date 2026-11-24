###############################################################
# Unit tests (GoogleTest), registered with CTest.
#
#   cmake -DENABLE_TESTING=ON ..   &&   make   &&   ctest
#
# An installed GoogleTest is used when CMake finds one; otherwise the pinned
# version below is downloaded at configure time.

find_package(GTest CONFIG QUIET)
if(GTest_FOUND)
  message(STATUS "Unit tests: using installed GoogleTest ${GTest_VERSION}")
else()
  message(STATUS "Unit tests: downloading GoogleTest")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/b796f7d44681514f58a683a3a71ff17c94edb0c1.zip
    URL_HASH SHA256=983a7f2f4cc2a4d75d94ee06300c46a657291fba965e355d11ab3b6965a7b0e5
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

# ccake_add_unit_test(<name> [OWN_MAIN] [libraries...])
#   Builds <name>.cpp, links GoogleTest and the libraries given, and registers
#   every TEST in it with CTest under the label "unit".
#   OWN_MAIN: the file defines main() itself (to start Kokkos once for all its
#   tests); otherwise GoogleTest's main() is linked.
#   Each test executable runs in its own directory, and its tests never run at
#   the same time as each other, because some of them write scratch files.
#
#   Known issues: a TEST in the suite "KnownIssue" documents a problem found in
#   the code that is reported but not fixed yet. It gets the label "quarantine"
#   instead of "unit", so that CI runs it and shows the result without letting
#   it block (ctest -LE quarantine / ctest -L quarantine). When the problem is
#   fixed, rename the suite and the test joins the others.
function(ccake_add_unit_test name)
  set(_libs ${ARGN})
  set(_gtest GTest::gtest_main)
  if("OWN_MAIN" IN_LIST _libs)
    list(REMOVE_ITEM _libs OWN_MAIN)
    set(_gtest GTest::gtest)
  endif()
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE ${_gtest} ${_libs})
  set(_workdir ${CMAKE_CURRENT_BINARY_DIR}/work/${name})
  file(MAKE_DIRECTORY ${_workdir})
  set(_discovery)
  if(NOT CMAKE_VERSION VERSION_LESS 3.18)
    # Look for the tests when ctest runs, not while building.
    set(_discovery DISCOVERY_MODE PRE_TEST)
  endif()
  # "threadsafe" death tests start a fresh process instead of forking one that
  # already runs OpenMP threads, which would deadlock.
  # DISCOVERY_TIMEOUT: listing the tests means starting the program, which in a
  # coverage build also writes its coverage data on exit; the default of 5 s is
  # too short for that on a slow file system, and a timeout aborts the whole run.
  set(_common
    DISCOVERY_TIMEOUT 120
    WORKING_DIRECTORY ${_workdir}
    ${_discovery}
  )
  set(_properties
    RESOURCE_LOCK ${name}_workdir
    ENVIRONMENT GTEST_DEATH_TEST_STYLE=threadsafe
    TIMEOUT 300
  )
  if(CMAKE_VERSION VERSION_LESS 3.22)
    # no TEST_FILTER: everything, known issues included, is a unit test
    gtest_discover_tests(${name} ${_common} PROPERTIES LABELS unit ${_properties})
  else()
    gtest_discover_tests(${name} ${_common} TEST_FILTER "-KnownIssue.*"
                         PROPERTIES LABELS unit ${_properties})
    gtest_discover_tests(${name} ${_common} TEST_FILTER "KnownIssue.*"
                         PROPERTIES LABELS quarantine ${_properties})
  endif()
endfunction()

# 'src' is the object library with all of CCAKE except main().
ccake_add_unit_test(vector_test)
ccake_add_unit_test(legacy_checks_test)
ccake_add_unit_test(kernel_test src)
ccake_add_unit_test(causality_test src)
ccake_add_unit_test(smoothing_test OWN_MAIN src)
ccake_add_unit_test(grad_smoothing_test OWN_MAIN src)
ccake_add_unit_test(input_test OWN_MAIN src)
ccake_add_unit_test(eom_default_test OWN_MAIN src)
ccake_add_unit_test(evolver_test OWN_MAIN src)
ccake_add_unit_test(eos_interpolator_test OWN_MAIN src)
ccake_add_unit_test(particle_test OWN_MAIN src)
ccake_add_unit_test(milne_test src)
ccake_add_unit_test(eos_test src)
ccake_add_unit_test(output_test OWN_MAIN src)
ccake_add_unit_test(source_test OWN_MAIN src)

# eom_default_test.cpp ends with a regression snapshot of the viscous time
# derivatives: read the comment there before changing the equations of motion.
