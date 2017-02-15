CC = g++
TARGETS = lr.o
$(TARGETS) :
	$(CC) -o lr.o lr.cpp
clean :
	@rm *.o && echo "object files cleaned!"
run : $(TARGETS)
	@./lr.o
