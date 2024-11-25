#!/bin/bash
# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

PORTS=(12345 12346 12347)
COUNTER_FILE=".video_counter.txt"
LOG_FILE="/var/log/ffmpeg_mgr.log"
VIDEO_BASE_PATH="/home/alecfessler/Documents/mimic/data"

log() {
    local level="$1"
    local message="$2"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [$level] $message" | tee -a "$LOG_FILE"
}

if [ ! -f "$COUNTER_FILE" ]; then
    log "INFO" "Initializing counter file"
    echo "0" > "$COUNTER_FILE"
fi

next_counter() {
    counter=$(cat "$COUNTER_FILE")
    next_counter=$((counter + 1))
    echo "$next_counter" > "$COUNTER_FILE"
    echo "$next_counter"
}

write_config() {
    local port="$1"
    local counter="$2"
    local filename="port${port}_vid${counter}"
    {
        echo "VIDEO_BASE_PATH=${VIDEO_BASE_PATH}"
        echo "FILENAME=${VIDEO_BASE_PATH}/${filename}_%03d.mp4"
    } > /etc/default/ffmpeg-stream
}

case $1 in
    start)
        log "INFO" "Starting FFmpeg instances..."
        counter=$(next_counter)
        for port in "${PORTS[@]}"; do
            filename="port${port}_vid${counter}"
            write_config "$port" "$counter"
            if ! sudo systemctl start ffmpeg-stream@"$port".service; then
                log "ERROR" "Failed to start ffmpeg-stream@$port.service"
                continue
            fi
            log "INFO" "Started ffmpeg-stream@$port.service with filename $filename"
        done
        log "INFO" "Starting watchdogs..."
        for port in "${PORTS[@]}"; do
            ./ffmpeg_watchdog.sh "$port" &
            echo $! > "/tmp/ffmpeg_watchdog_${port}.pid"
            log "INFO" "Started watchdog for port $port with PID $(cat /tmp/ffmpeg_watchdog_${port}.pid)"
        done
        ;;
    stop)
        log "INFO" "Stopping FFmpeg instances..."
        for port in "${PORTS[@]}"; do
            if ! sudo systemctl stop ffmpeg-stream@"$port".service; then
                log "ERROR" "Failed to stop ffmpeg-stream@$port.service"
                continue
            fi
            rm -f "/tmp/recording_active_${port}"
            log "INFO" "Stopped ffmpeg-stream@$port.service"
        done
        log "INFO" "Stopping watchdogs..."
        for port in "${PORTS[@]}"; do
            if [ -f "/tmp/ffmpeg_watchdog_${port}.pid" ]; then
                kill $(cat "/tmp/ffmpeg_watchdog_${port}.pid")
                rm "/tmp/ffmpeg_watchdog_${port}.pid"
                log "INFO" "Stopped watchdog for port $port"
            fi
        done
        ;;
    status)
        log "INFO" "Checking status of FFmpeg instances..."
        for port in "${PORTS[@]}"; do
            log "DEBUG" "Checking status of ffmpeg-stream@$port.service"
            sudo systemctl status ffmpeg-stream@"$port".service
        done
        ;;
    restart)
        log "INFO" "Restarting FFmpeg instances..."
        counter=$(next_counter)
        for port in "${PORTS[@]}"; do
            filename="port${port}_vid${counter}"
            write_config "$port" "$counter"
            if ! sudo systemctl restart ffmpeg-stream@"$port".service; then
                log "ERROR" "Failed to restart ffmpeg-stream@$port.service"
                continue
            fi
            log "INFO" "Restarted ffmpeg-stream@$port.service with filename $filename"
        done
        ;;
    *)
        log "ERROR" "Invalid command: $1"
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
