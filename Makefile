# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -pedantic -fopenmp

# Directories
SRCDIR = src
INCDIR = include
OBJDIR = obj
BINDIR = bin
EIGENDIR = Eigen

# Files
SRCFILES = $(wildcard $(SRCDIR)/*.cpp)
EIGENFILES = $(wildcard $(EIGENDIR)/*)
STBFILES = $(wildcard stb/*.h)
JSONFILES = $(wildcard json/*.hpp)
OBJFILES = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCFILES))
LIBRARY = $(BINDIR)/libhado.a
EXECUTABLE = $(BINDIR)/main

# Targets
all: lib $(EXECUTABLE)

# Compile all non-main.cpp files into a library
lib: $(LIBRARY)

$(LIBRARY): $(OBJFILES)
	@mkdir -p $(BINDIR)
	ar rcs $@ $^

# Compile each source file into an object file
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(EIGENFILES) $(STBFILES) $(JSONFILES)
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -I$(INCDIR) -I$(INCDIR)/base -I$(INCDIR)/layers -I$(INCDIR)/pipeline -I$(INCDIR)/image -I$(INCDIR)/errors -I ./ -o $@

# Compile the executable
$(EXECUTABLE): $(SRCDIR)/main.cpp $(LIBRARY)
	$(CXX) $(CXXFLAGS) $< -I$(INCDIR) -I$(INCDIR)/base -I$(INCDIR)/layers -I$(INCDIR)/pipeline -I$(INCDIR)/image -I$(INCDIR)/errors -I ./ -L$(BINDIR) -lhado -o $@

# Clean
clean:
	$(RM) -r $(OBJDIR) $(BINDIR)

.PHONY: all lib clean
