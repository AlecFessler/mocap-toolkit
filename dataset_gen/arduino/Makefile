MCU=atmega328p
TARGET=pulse_generator
SRC=$(TARGET).c
CC=avr-gcc
PROGRAMMER=arduino
PORT=/dev/ttyACM0
BAUD=115200
CFLAGS=-mmcu=$(MCU) -Wall

build: $(TARGET).hex

$(TARGET).elf: $(SRC)
	$(CC) $(CFLAGS) -o $@ $<

$(TARGET).hex: $(TARGET).elf
	avr-objcopy -O ihex -R .eeprom $< $@

burn: $(TARGET).hex
	avrdude -F -V -c $(PROGRAMMER) -p m328p -P $(PORT) -b $(BAUD) -U flash:w:$<

clean:
	rm -f $(TARGET).hex $(TARGET).elf
