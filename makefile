CC = gcc
TARGETS = main.o
SOURCES = main.c
release : $(SOURCES)
	$(CC) -o $(TARGETS) $(SOURCES)
debug : $(SOURCES)
	$(CC) -D __DEBUG__ -o $(TARGETS) $(SOURCES)

main.c : read_data.h

read_data.h : def.h

clean :
	rm *.o
run : $(TARGETS)
	./$(TARGETS)
