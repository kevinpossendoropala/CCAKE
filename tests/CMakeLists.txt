###############################################################
# Tests that need Python: the end-to-end smoke run and the tests of the scripts.
#
#   ctest -L smoke     one short run of the ccake executable
#   ctest -L python    pytest on tests/python
#
# Both are skipped, with a message, when no suitable Python is found.

# Use the Python that comes first on the PATH (an activated virtual or conda
# environment puts itself there), not the newest one CMake can find anywhere on
# the machine. Override with -DPython3_EXECUTABLE=/path/to/python.
set(Python3_FIND_STRATEGY LOCATION)
set(Python3_FIND_VIRTUALENV STANDARD)
set(Python3_FIND_UNVERSIONED_NAMES FIRST)
find_package(Python3 COMPONENTS Interpreter)
if(NOT Python3_FOUND)
  message(STATUS "Python 3 not found: smoke run and Python tests are not registered")
  return()
endif()

# Keep the source tree clean: no bytecode or pytest cache next to the sources.
set(_python_env "PYTHONDONTWRITEBYTECODE=1")

# A short ideal 1+1D run. Checks that ccake runs and that its output is well
# formed; it does not judge the physics.
add_test(
  NAME smoke_long_1d
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/regression/run_smoke.py
          --ccake $<TARGET_FILE:${PROJECT_NAME}>
          --source-dir ${PROJECT_SOURCE_DIR}
          --work-dir ${CMAKE_CURRENT_BINARY_DIR}/smoke_long_1d
          --steps 20
)
set_tests_properties(smoke_long_1d PROPERTIES LABELS smoke TIMEOUT 600 ENVIRONMENT "${_python_env}")

execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "import pytest"
  RESULT_VARIABLE _pytest_missing OUTPUT_QUIET ERROR_QUIET
)
if(_pytest_missing)
  message(STATUS "pytest not found for ${Python3_EXECUTABLE}: Python tests are not registered")
else()
  add_test(
    NAME python_tests
    COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/python
            -q -p no:cacheprovider
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  )
  set_tests_properties(python_tests PROPERTIES LABELS python TIMEOUT 600 ENVIRONMENT "${_python_env}")
endif()
