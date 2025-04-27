#!/bin/bash
# Setup script for Pi time synchronization
# This configures each Pi to participate in PTP synchronization
# using Best Master Clock Algorithm (BMCA) for automatic master selection

# Add NTP server configuration to sync with PC
sudo tee /etc/chrony/chrony.conf << 'EOF'
# Get time from PC with frequent polling
server 192.168.1.100 iburst minpoll 0 maxpoll 5 prefer

# Allow stepping the clock for large offsets
makestep 10.0 3

# Save drift for better startup accuracy
driftfile /var/lib/chrony/drift

# Disable server functionality
port 0
EOF

# Configure PTP with BMCA-friendly settings
sudo tee /etc/linuxptp/ptp4l.conf << 'EOF'
[global]
# Default priorities - let BMCA choose the best master
priority1               128
priority2               128
domainNumber           0
# Don't force slave mode, allow BMCA to work
slaveOnly              0
time_stamping          hardware
verbose                1
logging_level          6
# Set clock class to indicate we're happy to be master or slave
clockClass             128
# Additional BMCA parameters for fine-tuning
G.8275.defaultDS.localPriority     128
# Increase announce interval for better stability
logAnnounceInterval    1
# Add quality metrics that BMCA can use
clockAccuracy         0xFE
offsetScaledLogVariance 0xFFFF
[eth0]
network_transport      UDPv4
delay_mechanism       E2E
delay_filter         moving_median
delay_filter_length  10
EOF

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

echo "Pi time synchronization setup complete with BMCA"
