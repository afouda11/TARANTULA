EXEC = tarantula
CXX = g++-10             
OPTS = -O2 -Wall -g -I
#CFLAGS = $(OPTS) -larmadillo -fopenmp 
CFLAGS = $(OPTS) /Users/afouda/armadillo-9.900.2/include -fopenmp -framework Accelerate
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
