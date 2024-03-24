# Compiler
CXX = g++
# Compiler flags
CXXFLAGS = -Wall -Wextra -pedantic -std=c++11
# Source directory
SRC_DIR = src
# Object directory
OBJ_DIR = obj
# Source files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
# Object files
OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))
# Library name
LIBRARY = hado.a

# Default target
all: $(LIBRARY)

# Compile source files into object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
    @mkdir -p $(OBJ_DIR)
    $(CXX) $(CXXFLAGS) -c $< -o $@

# Create the library
$(LIBRARY): $(OBJS)
    ar rcs $@ $(OBJS)

# Clean up
clean:
    rm -rf $(OBJ_DIR) $(LIBRARY)
