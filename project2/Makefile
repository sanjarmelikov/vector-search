CXX = g++
CXXFLAGS = -std=c++17 -g -Wall -O2 

PROG ?= main

# Core objects
CORE_OBJS = \
	Item.o \
	Compare.o \
	ItemGenerator.o \

# Main program objects
MAIN_OBJS = main.o

OBJS = $(MAIN_OBJS) $(CORE_OBJS) $(TEST_OBJS)

mainprog: $(PROG)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(PROG): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

clean:
	rm -rf $(PROG) *.o *.out \
		*.o \
		*/*.o 

rebuild: clean main