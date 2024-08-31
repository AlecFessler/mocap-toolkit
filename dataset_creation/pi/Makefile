CC=g++
INCLUDES=-I./include `pkg-config --cflags --libs libcamera`
CFLAGS=-Wall -Wextra -g $(INCLUDES)
LDFLAGS=-pthread -lrt `pkg-config --cflags --libs libcamera`

CPPFILES=$(wildcard src/*.cpp)
OBJFILES=$(CPPFILES:src/%.cpp=obj/%.o)

BINARY=bin/framecap

all: $(BINARY)

$(BINARY): $(OBJFILES)
	@mkdir -p $(dir $(BINARY))
	$(CC) $(OBJFILES) -o $@ $(LDFLAGS)
	sudo setcap cap_sys_nice+ep $(BINARY)

obj/%.o: src/%.cpp
	@mkdir -p obj
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJFILES) $(BINARY)
