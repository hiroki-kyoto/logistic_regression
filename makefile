CC = gcc
TARGETS = main.o
SOURCES = main.c
$(TARGETS) : $(SOURCES)
	$(CC) -o $(TARGETS) $(SOURCES)
clean :
	rm *.o
run : $(TARGETS)
	./main.o
