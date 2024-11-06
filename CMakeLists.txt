# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 3.27
cmake_minimum_required(VERSION 3.15...3.27)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)


# Find the module development requirements (requires FindPython from 3.17 or
# scikit-build-core's built-in backport)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Add a library using FindPython's tooling (pybind11 also provides a helper like
# this)
python_add_library(_core MODULE src/ffi/main.cpp WITH_SOABI)
target_link_libraries(_core PRIVATE pybind11::headers)

python_add_library(_gpu_stable MODULE src/ffi/GPU_stable.cu WITH_SOABI)
add_library(German STATIC src/ffi/shared/German.cc)
target_include_directories(_gpu_stable PUBLIC src/ffi/shared)
target_link_libraries(_gpu_stable PRIVATE pybind11::headers CUDA::curand German)



# This is passing in the version as a define just as an example
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})

# The install directory is the output (wheel) directory
install(TARGETS _core _gpu_stable DESTINATION bang)
# install(TARGETS _gpu_stable DESTINATION bang)