obj-m := gpio17_interrupt_driver.o

KDIR := /lib/modules/$(shell uname -r)/build

PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install: all
	sudo mkdir -p /lib/modules/$(shell uname -r)/extra
	sudo cp gpio17_interrupt_driver.ko /lib/modules/$(shell uname -r)/extra/
	sudo depmod -a

uninstall:
	sudo rm -f /lib/modules/$(shell uname -r)/extra/gpio17_interrupt_driver.ko
	sudo depmod -a
