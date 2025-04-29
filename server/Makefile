CC=gcc
PKG_AVCODEC=$(shell pkg-config --cflags libavcodec libavutil)
INCLUDES=-I./include $(PKG_AVCODEC) -I/opt/cuda/include
CFLAGS=-Wall -Wextra -O2 $(INCLUDES)

PKG_LIBS_AVCODEC=$(shell pkg-config --libs libavcodec libavutil)
LDFLAGS=-pthread -latomic -lyaml -L/opt/cuda/lib64 -lcudart $(PKG_LIBS_AVCODEC)

CFILES=$(wildcard src/*.c)
OBJFILES=$(CFILES:src/%.c=obj/%.o)
BINARY=bin/mocap-toolkit-server
INSTALL_PATH=/usr/local/bin/mocap-toolkit-server

all: $(BINARY)

$(BINARY): $(OBJFILES)
	@mkdir -p $(dir $(BINARY))
	$(CC) $(OBJFILES) -o $@ $(LDFLAGS)

obj/%.o: src/%.c
	@mkdir -p obj
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJFILES) $(BINARY)

install: $(BINARY)
	@echo "Installing mocap-toolkit-server to $(INSTALL_PATH)"
	@sudo install -m 755 $(BINARY) $(INSTALL_PATH)

uninstall:
	@echo "Removing mocap-toolkit-server from $(INSTALL_PATH)"
	@sudo rm -f $(INSTALL_PATH)
