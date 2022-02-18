EXEC = tarantula
CXX = g++      
OPTS = -O2 -Wall -I
#CFLAGS = $(OPTS) -larmadillo -fopenmp 
CFLAGS = $(OPTS) /home/oxygen/FOUDAAE/armadillo-install/include -fopenmp
CXXFLAGS = -std=c++11
LDFLAGS = -fopenmp 

INCDIR =./include
OBJDIR = ./obj
BINDIR = ./bin
SRCDIR = ./src

CFLAGS += -I$(INCDIR) -I$(SRCDIR)

SOURCES = read_and_write.cpp pulse_interaction.cpp main.cpp rk4.cpp

_OBJ = $(SOURCES:.cpp=.o)
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

all: $(BINDIR)/$(EXEC)

$(BINDIR)/$(EXEC): $(OBJ)
	$(CXX) -o $(BINDIR)/$(EXEC) $(OBJ) $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) -c -o $@ $< $(CFLAGS)

clean:
	rm -vf $(BINDIR)/$(EXEC) $(OBJ)
