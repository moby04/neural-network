cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NeuralNetwork)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the executable
file(GLOB SOURCES "src/*.cpp" "src/*/*.cpp")
list(REMOVE_ITEM SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")
add_executable(NeuralNetwork ${SOURCES} src/main.cpp)

# Enable testing
enable_testing()

# Add GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add test executable
file(GLOB TEST_SOURCES "tests/test_*.cpp" "src/*.cpp" "src/*/*.cpp")
list(REMOVE_ITEM TEST_SOURCES "${CMAKE_SOURCE_DIR}/src/main.cpp")
add_executable(NeuralNetworkTests ${TEST_SOURCES})

# Link test executable against gtest & gtest_main
target_link_libraries(NeuralNetworkTests gtest_main)

# Discover tests
include(GoogleTest)
gtest_discover_tests(NeuralNetworkTests)
