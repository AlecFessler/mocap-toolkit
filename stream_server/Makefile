CC=gcc
INCLUDES=-I./include
CFLAGS=-Wall -Wextra -g $(INCLUDES)
LDFLAGS=-pthread -lyaml

CFILES=$(wildcard src/*.c)
OBJFILES=$(CFILES:src/%.c=obj/%.o)
BINARY=bin/collect_frames

all: $(BINARY)

$(BINARY): $(OBJFILES)
	@mkdir -p $(dir $(BINARY))
	$(CC) $(OBJFILES) -o $@ $(LDFLAGS)

obj/%.o: src/%.c
	@mkdir -p obj
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJFILES) $(BINARY)
