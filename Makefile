# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -pedantic

# Directories
SRCDIR = src
INCDIR = HaDo
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
INCLUDES = -I ./ -I$(INCDIR) -I$(INCDIR)/base -I$(INCDIR)/layers -I$(INCDIR)/pipeline -I$(INCDIR)/image -I$(INCDIR)/errors -I$(INCDIR)/util

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
	$(CXX) $(CXXFLAGS) -c $< $(INCLUDES) -o $@

# Compile the executable without openmp
$(EXECUTABLE): $(SRCDIR)/main.cpp $(LIBRARY)
	$(CXX) $(CXXFLAGS) $< $(INCLUDES) -L$(BINDIR) -lhado -o $@

# Compile the executable with openmp
omp: CXXFLAGS += -fopenmp
omp: $(EXECUTABLE)

# Clean
clean:
	$(RM) -r $(OBJDIR) $(BINDIR)

.PHONY: all lib omp clean
