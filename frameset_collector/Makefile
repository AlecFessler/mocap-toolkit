CC=gcc
PKG_AVCODEC=$(shell pkg-config --cflags libavcodec libavutil)
INCLUDES=-I./include $(PKG_AVCODEC)
CFLAGS=-Wall -Wextra -g $(INCLUDES)

PKG_LIBS_AVCODEC=$(shell pkg-config --libs libavcodec libavutil)
LDFLAGS=-pthread -latomic -lyaml $(PKG_LIBS_AVCODEC)

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
