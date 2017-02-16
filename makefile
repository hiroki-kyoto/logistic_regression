CC = g++
TARGETS = lr.o
$(TARGETS) :
	$(CC) -o lr.o lr.cpp
clean :
	rm *.o
run : $(TARGETS)
	./lr.o
