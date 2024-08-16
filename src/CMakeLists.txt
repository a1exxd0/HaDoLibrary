# src/CMakeLists.txt

# Include the main source directory to access headers
include_directories(${PROJECT_SOURCE_DIR}/)

# Specify test directory
set(SRC_FILES_DIRECTORY ${PROJECT_SOURCE_DIR}/src)

# Add main
add_executable(main ${SRC_FILES_DIRECTORY}/main.cpp)

# Include files
include_directories(main ${SRC_FILES_DIRECTORY})