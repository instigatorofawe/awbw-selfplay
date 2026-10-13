cmake_minimum_required(VERSION 3.27...4.4)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AWBW_NATIVE "Optimize for the build machine's CPU (not for distributed wheels)" OFF)

find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(nanobind CONFIG REQUIRED)

# Engine core: plain C++20, no Python dependency.
add_library(awbw_core INTERFACE)
target_include_directories(awbw_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if(AWBW_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native AWBW_HAS_MARCH_NATIVE)
  if(AWBW_HAS_MARCH_NATIVE)
    target_compile_options(awbw_core INTERFACE -march=native)
  else()
    target_compile_options(awbw_core INTERFACE -mcpu=native)
  endif()
endif()

nanobind_add_module(_core NB_STATIC STABLE_ABI LTO src/bindings/module.cpp)
target_link_libraries(_core PRIVATE awbw_core)
install(TARGETS _core LIBRARY DESTINATION awbw)

# The stub is written into the source tree and committed, so type checkers see
# awbw._core without a build; it ships with the package from there. CI fails
# if the committed copy drifts from what the build generates.
nanobind_add_stub(_core_stub
  MODULE _core
  OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/awbw/_core.pyi
  PYTHON_PATH $<TARGET_FILE_DIR:_core>
  DEPENDS _core)
