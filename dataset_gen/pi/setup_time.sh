#!/bin/bash
# Setup script for Pi time synchronization
# This configures each Pi to:
# 1. Get time directly from the PC via NTP
# 2. Participate in PTP synchronization according to its role (master or slave)

# First argument determines if this is the grandmaster (gm) or slave
if [ "$1" != "gm" ] && [ "$1" != "slave" ]; then
    echo "Usage: $0 [gm|slave]"
    echo "Example: $0 gm     # Set up as PTP grandmaster"
    echo "Example: $0 slave  # Set up as PTP slave"
    exit 1
fi

# Add NTP server configuration to sync with PC
# We append to the existing config rather than replacing it
sudo tee -a /etc/chrony/chrony.conf << 'EOF'

# Get time from PC at 192.168.86.100
server 192.168.86.100 iburst minpoll 0 maxpoll 5 prefer
EOF

# Configure PTP based on role
if [ "$1" = "gm" ]; then
    # Grandmaster PTP configuration
    sudo tee /etc/linuxptp/ptp4l.conf << 'EOF'
[global]
priority1               127
priority2               128
domainNumber            0
slaveOnly               0
time_stamping           hardware
verbose                 1
logging_level           6
message_tag             master
[eth0]
network_transport       UDPv4
delay_mechanism         E2E
delay_filter            moving_median
delay_filter_length     10
EOF
else
    # Slave PTP configuration
    sudo tee /etc/linuxptp/ptp4l.conf << 'EOF'
[global]
priority1               255
priority2               255
domainNumber            0
slaveOnly               1
time_stamping           hardware
verbose                 1
logging_level           6
message_tag             slave
[eth0]
network_transport       UDPv4
delay_mechanism         E2E
delay_filter            moving_median
delay_filter_length     10
EOF
fi

# Create time initialization service
# This ensures we get a rough sync via NTP before starting PTP
sudo tee /etc/systemd/system/time-init.service << 'EOF'
[Unit]
Description=Initial Time Synchronization
Before=ptp4l.service
After=network-online.target chrony.service
Wants=network-online.target
Requires=chrony.service

[Service]
Type=oneshot
ExecStart=/usr/bin/chronyc waitsync 30 1.0
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Create PTP service that waits for initial time sync
sudo tee /etc/systemd/system/ptp4l.service << 'EOF'
[Unit]
Description=Precision Time Protocol (PTP) Service
After=network-online.target time-init.service
Wants=network-online.target
Requires=time-init.service

[Service]
Type=simple
ExecStart=/usr/sbin/ptp4l -f /etc/linuxptp/ptp4l.conf -i eth0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and restart services to apply changes
sudo systemctl daemon-reload
sudo systemctl enable chrony
sudo systemctl restart chrony
sudo systemctl enable time-init.service
sudo systemctl start time-init.service
sudo systemctl enable ptp4l
sudo systemctl restart ptp4l

echo "Pi time synchronization setup complete as $1"
