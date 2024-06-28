# CMakeLists.txt

# Specify project make details
cmake_minimum_required(VERSION 3.22)
project(hadolibrary)

# Set minimum C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -O0 -g -fsanitize=address")

# Add testing directory
enable_testing()
add_subdirectory(test)