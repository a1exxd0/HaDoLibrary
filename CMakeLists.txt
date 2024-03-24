cmake_minimum_required(VERSION 3.5)
project(HaDo VERSION 1.0)

# Add your source files
set(SOURCES
    src/init.cpp
    # Add more source files here if needed
)

# Add your header files
set(HEADERS
    include/init.hpp
    # Add more header files here if needed
)

# Add your library target
add_library(hado ${SOURCES} ${HEADERS})

# Set include directories
target_include_directories(hado
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Set any additional compiler options
target_compile_options(hado
    PUBLIC
    -Wall
    -Wextra
    # Add more compiler options here if needed
)

# Set C++ standard
target_compile_features(hado PUBLIC cxx_std_20)

# Optionally, you can install the library
# install(TARGETS your_library_name
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
#     RUNTIME DESTINATION bin
# )

# Optionally, you can install the headers
# install(FILES ${HEADERS} DESTINATION include/your_library)
