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

sudo tee /etc/systemd/system/ptp4l.service << 'EOF'
[Unit]
Description=Precision Time Protocol (PTP) Master service
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
