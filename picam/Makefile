CC=g++
PKG_CAMERA=$(shell pkg-config --cflags libcamera)
PKG_AVCODEC=$(shell pkg-config --cflags libavcodec libavutil)
INCLUDES=-I./include $(PKG_CAMERA) $(PKG_AVCODEC)
CFLAGS=-Wall -Wextra -g $(INCLUDES)

PKG_LIBS_CAMERA=$(shell pkg-config --libs libcamera)
PKG_LIBS_AVCODEC=$(shell pkg-config --libs libavcodec libavutil)
LDFLAGS=-pthread $(PKG_LIBS_CAMERA) $(PKG_LIBS_AVCODEC) -lrt -latomic

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
	$(CC) $(CFLAGS) -c -o $@ src/$*.cpp

clean:
	rm -f $(OBJFILES) $(BINARY)
