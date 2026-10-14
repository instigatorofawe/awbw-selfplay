cmake_minimum_required(VERSION 3.28...4.4)
project(awbw_selfplay LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# scikit-build-core builds the Python extension; a plain CMake build is for
# the C++ tests and benchmarks:
#   cmake -S . -B build/cpp -DAWBW_BUILD_TESTS=ON -DAWBW_BUILD_BENCHMARKS=ON
#   cmake --build build/cpp && ctest --test-dir build/cpp
if(DEFINED SKBUILD)
  set(_awbw_python_default ON)
else()
  set(_awbw_python_default OFF)
endif()
option(AWBW_BUILD_PYTHON "Build the nanobind extension" ${_awbw_python_default})
option(AWBW_BUILD_TESTS "Build the C++ unit tests (googletest)" OFF)
option(AWBW_BUILD_BENCHMARKS "Build the C++ micro-benchmarks (Google Benchmark)" OFF)
option(AWBW_NATIVE "Optimize for the build machine's CPU (not for distributed wheels)" OFF)
option(AWBW_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(AWBW_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all)
  add_link_options(-fsanitize=address,undefined)
endif()

# Engine core: plain C++20, no Python dependency.
add_library(awbw_core STATIC
  src/core/actions.cpp
  src/core/belief.cpp
  src/core/combat.cpp
  src/core/encode.cpp
  src/core/hash.cpp
  src/core/legal.cpp
  src/core/map.cpp
  src/core/movegen.cpp
  src/core/snapshot.cpp
  src/core/state.cpp)
target_include_directories(awbw_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
set_target_properties(awbw_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(awbw_core PRIVATE -Wall -Wextra)
endif()

if(AWBW_NATIVE)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-march=native AWBW_HAS_MARCH_NATIVE)
  if(AWBW_HAS_MARCH_NATIVE)
    target_compile_options(awbw_core PUBLIC -march=native)
  else()
    target_compile_options(awbw_core PUBLIC -mcpu=native)
  endif()
endif()

if(AWBW_BUILD_PYTHON)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_core NB_STATIC STABLE_ABI LTO src/bindings/module.cpp)
  target_link_libraries(_core PRIVATE awbw_core)
  install(TARGETS _core LIBRARY DESTINATION awbw)

  # The stub is written into the source tree and committed, so type checkers
  # see awbw._core without a build; it ships with the package from there. CI
  # fails if the committed copy drifts from what the build generates.
  nanobind_add_stub(_core_stub
    MODULE _core
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/awbw/_core.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:_core>
    DEPENDS _core)
endif()

if(AWBW_BUILD_TESTS OR AWBW_BUILD_BENCHMARKS)
  include(FetchContent)
endif()

if(AWBW_BUILD_TESTS)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.18.0.tar.gz
    URL_HASH SHA256=6e3191c1455468b3fc35a417fb565c1c5071aee1b7e7f85e30cf48a98d37d8b5)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  enable_testing()
  add_executable(awbw_tests
    tests/cpp/test_actions.cpp
    tests/cpp/test_belief.cpp
    tests/cpp/test_combat.cpp
    tests/cpp/test_encode.cpp
    tests/cpp/test_hash.cpp
    tests/cpp/test_hidden_hp.cpp
    tests/cpp/test_legal.cpp
    tests/cpp/test_map.cpp
    tests/cpp/test_movegen.cpp
    tests/cpp/test_powers.cpp
    tests/cpp/test_snapshot.cpp
    tests/cpp/test_state.cpp)
  target_link_libraries(awbw_tests PRIVATE awbw_core GTest::gtest_main)
  target_include_directories(awbw_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
  include(GoogleTest)
  gtest_discover_tests(awbw_tests)
endif()

if(AWBW_BUILD_BENCHMARKS)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz
    URL_HASH SHA256=9631341c82bac4a288bef951f8b26b41f69021794184ece969f8473977eaa340)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)

  # Abseil's Swiss tables, measured against std::unordered_map as the
  # transposition table search will use (BM_TableFind).
  FetchContent_Declare(abseil
    URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20260817.0.tar.gz
    URL_HASH SHA256=f7e05179df39c45434cad433f5783840bb3788ef322976f9138bc6b72b3a107d
    EXCLUDE_FROM_ALL)  # build only the libraries linked
  set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)
  set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
  set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(abseil)

  add_executable(awbw_bench bench/bench_core.cpp)
  target_link_libraries(awbw_bench PRIVATE awbw_core benchmark::benchmark_main absl::flat_hash_map absl::hash)
  target_include_directories(awbw_bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp)
endif()
