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

sudo tee /etc/systemd/system/ptp4l.service << 'EOF'
[Unit]
Description=Precision Time Protocol (PTP) Slave service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/sbin/ptp4l -f /etc/linuxptp/ptp4l.conf -i eth0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ptp4l
sudo systemctl start ptp4l
