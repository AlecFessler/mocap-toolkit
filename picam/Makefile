CC=g++
CXXFLAGS=-std=c++20
PKG_CAMERA=$(shell pkg-config --cflags libcamera)
PKG_AVCODEC=$(shell pkg-config --cflags libavcodec libavutil)
INCLUDES=-I./include $(PKG_CAMERA) $(PKG_AVCODEC)
CFLAGS=$(CXXFLAGS) -Wall -Wextra -g $(INCLUDES)
PKG_LIBS_CAMERA=$(shell pkg-config --libs libcamera)
PKG_LIBS_AVCODEC=$(shell pkg-config --libs libavcodec libavutil)
LDFLAGS=-pthread $(PKG_LIBS_CAMERA) $(PKG_LIBS_AVCODEC) -lrt -latomic
CPPFILES=$(wildcard src/*.cpp)
OBJFILES=$(CPPFILES:src/%.cpp=obj/%.o)
BINARY=bin/picam
PREFIX=/usr/local

.PHONY: all clean install uninstall

all: $(BINARY)

$(BINARY): $(OBJFILES)
	@mkdir -p $(dir $(BINARY))
	$(CC) $(OBJFILES) -o $@ $(LDFLAGS)
	sudo setcap cap_sys_nice+ep $(BINARY)

obj/%.o: src/%.cpp
	@mkdir -p obj
	$(CC) $(CFLAGS) -c -o $@ src/$*.cpp

install: $(BINARY)
	install -d $(DESTDIR)$(PREFIX)/bin
	install -m 755 $(BINARY) $(DESTDIR)$(PREFIX)/bin/picam
	sudo setcap cap_sys_nice+ep $(DESTDIR)$(PREFIX)/bin/picam

uninstall:
	rm -f $(DESTDIR)$(PREFIX)/bin/picam

clean:
	rm -f $(OBJFILES) $(BINARY)
