# Raspberry Pi PREEMPT-RT Kernel Installation Guide

## Step 1: Install Required Packages
First, install the necessary packages for building the kernel:
```bash
sudo apt update
sudo apt install -y git bc bison flex libssl-dev make libc6-dev libncurses5-dev crossbuild-essential-arm64
```

## Step 2: Download the Kernel Source
Download the Raspberry Pi kernel source for version 6.6:

```bash
cd ~
git clone --depth=1 --branch rpi-6.6.y https://github.com/raspberrypi/linux
cd linux
```

## Step 3: Download PREEMPT-RT Patch
Download the appropriate PREEMPT-RT patch for the 6.6 kernel series:

```bash
wget https://cdn.kernel.org/pub/linux/kernel/projects/rt/6.6/patch-6.6.44-rt39.patch.xz
unxz patch-6.6.44-rt39.patch.xz
patch -p1 < patch-6.6.44-rt39.patch
```

## Step 4: Configure the Kernel
Configure the kernel for the Raspberry Pi with PREEMPT-RT:

```bash
KERNEL=kernel8
make bcm2711_defconfig
make menuconfig

# In the menuconfig, navigate to:

# General Features -> Preemption Model (Voluntary Kernel Preemption (Voluntary)) and select Fully Preemptible Kernel (RT).

# Kernel Features -> Timer frequency and set it to 1000 HZ.

# Save and exit the menuconfig.
```

## Step 5: Build the Kernel
Build the kernel and modules:

```bash
make -j$(nproc) Image.gz modules dtbs
```

## Step 6: Install the Kernel and Modules
Install the kernel and modules:

```bash
sudo make modules_install
sudo cp arch/arm64/boot/Image.gz /boot/kernel8.img
sudo cp arch/arm64/boot/dts/broadcom/*.dtb /boot/
sudo cp arch/arm64/boot/dts/overlays/*.dtb* /boot/overlays/
sudo cp arch/arm64/boot/dts/overlays/README /boot/overlays/
```

## Step 7: Update the Boot Configuration
Edit the boot configuration file to use the new kernel:

```bash
sudo nano /boot/firmware/config.txt

# Ensure the following line is present:
# kernel=kernel8.img
```

## Step 8: Reboot and Verify
Reboot the Raspberry Pi:

```bash
sudo reboot
```

After the system reboots, verify that the PREEMPT-RT kernel is running:

```bash
uname -a
```
