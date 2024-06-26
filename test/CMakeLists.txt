# test/CMakeLists.txt

# Include the main source directory to access headers
include_directories(${PROJECT_SOURCE_DIR}/)

# Set memory leak checkers
include(CTest)
set(CTEST_MEMORYCHECK_COMMAND "valgrind")
set(CTEST_MEMORYCHECK_COMMAND_OPTIONS "--leak-check=full --suppressions=${PROJECT_SOURCE_DIR}/.github/workflows/valgrind.supp")

# Find the testing framework
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Specify test directory 
set(TEST_FILES_DIRECTORY ${PROJECT_SOURCE_DIR}/test)

# Recursively get all .cpp files in test/
file(GLOB_RECURSE TEST_FILES ${TEST_FILES_DIRECTORY} *.cpp)

# Iterate over each file
foreach(TEST_FILE ${TEST_FILES})
    # Get the file name without extension
    get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
    
    # Create an executable for each test
    add_executable(${TEST_NAME} ${SOURCES} ${TEST_FILE})
    
    # Link the GTest and pthread libraries
    target_link_libraries(${TEST_NAME} ${GTEST_LIBRARIES} pthread)
    
    # Add a test with the same name as the executable
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
endforeach()